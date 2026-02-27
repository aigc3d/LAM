# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


import math
from typing import Optional, Literal, Dict, List
from glob import glob
import concurrent.futures
import multiprocessing
from copy import deepcopy
import yaml
import json
import tyro
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
# from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from vhap.config.base import DataConfig, ModelConfig, import_module
from vhap.data.nerf_dataset import NeRFDataset
from vhap.model.flame import FlameHead
from vhap.util.mesh import get_obj_content
from vhap.util.render_nvdiffrast import NVDiffRenderer

# to prevent "OSError: [Errno 24] Too many open files"
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


max_threads = min(multiprocessing.cpu_count(), 8)


class NeRFDatasetWriter:
    def __init__(self, cfg_data: DataConfig, tgt_folder: Path, subset:Optional[str]=None, scale_factor: Optional[float]=None, background_color: Optional[str]=None):
        self.cfg_data = cfg_data
        self.tgt_folder = tgt_folder

        print("==== Config: data ====")
        print(tyro.to_yaml(cfg_data))

        cfg_data.target_extrinsic_type = 'c2w'
        cfg_data.background_color = 'white'
        cfg_data.use_alpha_map = True
        dataset = import_module(cfg_data._target)(cfg=cfg_data)
        self.dataloader = DataLoader(dataset, shuffle=False, batch_size=None, collate_fn=lambda x: x, num_workers=0)

    def write(self):
        if not self.tgt_folder.exists():
            self.tgt_folder.mkdir(parents=True)
        
        db = {
            "frames": [],
        }
        
        print(f"Writing images to {self.tgt_folder}")
        worker_args = []
        timestep_indices = set()
        camera_indices = set()
        for i, item in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            # print(item.keys())

            timestep_indices.add(item['timestep_index'])
            camera_indices.add(item['camera_index'])

            extrinsic = item['extrinsic']
            transform_matrix = torch.cat([extrinsic, torch.tensor([[0,0,0,1]])], dim=0).numpy()

            intrinsic = item['intrinsic'].double().numpy()

            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
            fl_x = intrinsic[0, 0]
            fl_y = intrinsic[1, 1]
            h = item['rgb'].shape[0]
            w = item['rgb'].shape[1]
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2

            frame_item = {
                "timestep_index": item['timestep_index'],
                "timestep_index_original": item['timestep_index_original'],
                "timestep_id": item['timestep_id'],
                "camera_index": item['camera_index'],
                "camera_id": item['camera_id'],

                "cx": cx,
                "cy": cy,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "h": h,
                "w": w,
                "camera_angle_x": angle_x,
                "camera_angle_y": angle_y,
                
                "transform_matrix": transform_matrix.tolist(),

                "file_path": f"images/{item['timestep_index']:05d}_{item['camera_index']:02d}.png",
            }

            path2data = {
                str(self.tgt_folder / frame_item['file_path']): item['rgb'],
            }

            if 'alpha_map' in item:
                frame_item['fg_mask_path'] = f"fg_masks/{item['timestep_index']:05d}_{item['camera_index']:02d}.png"
                path2data[str(self.tgt_folder / frame_item['fg_mask_path'])] = item['alpha_map']

            db['frames'].append(frame_item)
            worker_args.append([path2data])
            
            #--- no threading
            # if len(worker_args) > 0:
            #     write_data(path2data)

            #--- threading
            if len(worker_args) == max_threads or i == len(self.dataloader)-1:
                with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                    futures = [executor.submit(write_data, *args) for args in worker_args]
                    concurrent.futures.wait(futures)
                worker_args = []
            
        # add shared intrinsic parameters to be compatible with other nerf libraries
        db.update({
            "cx": cx,
            "cy": cy,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "h": h,
            "w": w,
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y
        })

        # add indices to ease filtering
        db['timestep_indices'] = sorted(list(timestep_indices))
        db['camera_indices'] = sorted(list(camera_indices))
        
        write_json(db, self.tgt_folder)
        write_json(db, self.tgt_folder, division='backup')


class TrackedFLAMEDatasetWriter:
    def __init__(self, cfg_model: ModelConfig, src_folder: Path, tgt_folder: Path, mode: Literal['mesh', 'param'], epoch: int = -1):
        print("---- Config: model ----")
        print(tyro.to_yaml(cfg_model))

        self.cfg_model = cfg_model
        self.src_folder = src_folder
        self.tgt_folder = tgt_folder
        self.mode = mode

        db_backup_path = tgt_folder / "transforms_backup.json"
        assert db_backup_path.exists(), f"Could not find {db_backup_path}"
        print(f"Loading database from: {db_backup_path}")
        self.db = json.load(open(db_backup_path, "r"))
        
        paths = [Path(p) for p in glob(str(src_folder / "tracked_flame_params*.npz"))]
        epochs = [int(p.stem.split('_')[-1]) for p in paths]
        if epoch == -1:
            index = np.argmax(epochs)
        else:
            index = epochs.index(epoch)
        flame_params_path = paths[index]

        assert flame_params_path.exists(), f"Could not find {flame_params_path}"
        print(f"Loading FLAME parameters from: {flame_params_path}")
        self.flame_params = dict(np.load(flame_params_path))

        if "focal_length" in self.flame_params:
            self.focal_length = self.flame_params['focal_length'].item()
        else:
            self.focal_length = None

        # Relocate FLAME to the origin and return the transformation matrix to modify camera poses.
        self.M = self.relocate_flame_meshes(self.flame_params)

        print("Initializing FLAME model...")
        self.flame_model = FlameHead(cfg_model.n_shape, cfg_model.n_expr, add_teeth=True)
    
    def relocate_flame_meshes(self, flame_param):
        """ Relocate FLAME to the origin and return the transformation matrix to modify camera poses. """
        # Rs = torch.tensor(flame_param['rotation'])
        Ts = torch.tensor(flame_param['translation'])

        # R_mean = axis_angle_to_matrix(Rs.mean(0))
        T_mean = Ts.mean(0)
        M = torch.eye(4)
        # M[:3, :3] = R_mean.transpose(-1, -2)
        M[:3, 3] = -T_mean
        
        # flame_param['rotation'] = (matrix_to_axis_angle(M[None, :3, :3] @ axis_angle_to_matrix(Rs))).numpy()
        flame_param['translation'] = (M[:3, 3] + Ts).numpy()
        return M.numpy()

    def replace_cam_params(self, item):
        c2w = np.eye(4)
        c2w[2, 3] = 1  # place the camera at (0, 0, 1) in the world coordinate by default
        item['transform_matrix'] = c2w

        h = item['h']
        w = item['w']
        fl_x = self.focal_length * max(h, w)
        fl_y = self.focal_length * max(h, w)
        angle_x = math.atan(w / (fl_x * 2)) * 2
        angle_y = math.atan(h / (fl_y * 2)) * 2

        item.update({
            "cx": w / 2,
            "cy": h / 2,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            
            "transform_matrix": c2w.tolist(),
        })

    def write(self):
        if self.mode == 'mesh':
            self.write_canonical_mesh()
            indices = self.db['timestep_indices']
            verts = infer_flame_params(self.flame_model, self.flame_params, indices)

            print(f"Writing FLAME expressions and meshes to: {self.tgt_folder}")
        elif self.mode == 'param':
            self.write_canonical_flame_param()
            print(f"Writing FLAME parameters to: {self.tgt_folder}")
        
        saved = [False] * len(self.db['timestep_indices'])  # avoid writing the same mesh multiple times
        num_processes = 0
        worker_args = []
        for i, frame in tqdm(enumerate(self.db['frames']), total=len(self.db['frames'])):
            if self.focal_length is not None:
                self.replace_cam_params(frame)
            # modify the camera extrinsics to place the tracked FLAME at the origin
            frame['transform_matrix'] = (self.M @ np.array(frame['transform_matrix'])).tolist()

            ti_orig = frame['timestep_index_original']  # use ti_orig when loading FLAME parameters
            ti = frame['timestep_index']  # use ti when saving files

            # write FLAME mesh or parameters
            if self.mode == 'mesh':
                frame['exp_path'] = f"flame/exp/{ti:05d}.txt"
                frame['mesh_path'] = f"meshes/{ti:05d}.obj"
                if not saved[ti]:
                    worker_args.append([self.tgt_folder, frame['exp_path'], self.flame_params['expr'][ti_orig], frame['mesh_path'], verts[ti_orig], self.flame_model.faces])
                    saved[ti] = True
                    func = self.write_expr_and_mesh
            elif self.mode == 'param':
                frame['flame_param_path'] = f"flame_param/{ti:05d}.npz"
                if not saved[ti]:
                    worker_args.append([self.tgt_folder, frame['flame_param_path'], self.flame_params, ti_orig])
                    saved[ti] = True
                    func = self.write_flame_param
            #--- no multiprocessing
            if len(worker_args) > 0:
                func(*worker_args.pop())
            #--- multiprocessing
            # if len(worker_args) == num_processes or i == len(self.db['frames'])-1:
            #     pool = multiprocessing.Pool(processes=num_processes)
            #     pool.starmap(func, worker_args)
            #     pool.close()
            #     pool.join()
            #     worker_args = []

        write_json(self.db, self.tgt_folder)
        write_json(self.db, self.tgt_folder, division='backup_flame')
    
    def write_canonical_mesh(self):
        print(f"Inferencing FLAME in the canonical space...")
        if 'static_offset' in self.flame_params:
            static_offset = torch.tensor(self.flame_params['static_offset'])
        else:
            static_offset = None
        with torch.no_grad():
            ret = self.flame_model(
                torch.tensor(self.flame_params['shape'])[None, ...],
                torch.zeros(*self.flame_params['expr'][:1].shape),
                torch.zeros(*self.flame_params['rotation'][:1].shape),
                torch.zeros(*self.flame_params['neck_pose'][:1].shape),
                torch.tensor([[0.3, 0, 0]]),
                torch.zeros(*self.flame_params['eyes_pose'][:1].shape),
                torch.zeros(*self.flame_params['translation'][:1].shape),
                return_verts_cano=False,
                static_offset=static_offset,
            )
        verts = ret[0]

        cano_mesh_path = self.tgt_folder / 'canonical.obj'
        print(f"Writing canonical mesh to: {cano_mesh_path}")
        obj_data = get_obj_content(verts[0], self.flame_model.faces)
        write_data({cano_mesh_path: obj_data})
    
    @staticmethod
    def write_expr_and_mesh(tgt_folder, exp_path, expr, mesh_path, verts, faces):
        path2data = {}

        expr_data = '\n'.join([str(n) for n in expr])
        path2data[tgt_folder / exp_path] = expr_data

        obj_data = get_obj_content(verts, faces)
        path2data[tgt_folder / mesh_path] = obj_data
        write_data(path2data)
    
    def write_canonical_flame_param(self):
        flame_param = {
            'translation': np.zeros_like(self.flame_params['translation'][:1]),
            'rotation': np.zeros_like(self.flame_params['rotation'][:1]),
            'neck_pose': np.zeros_like(self.flame_params['neck_pose'][:1]),
            'jaw_pose': np.array([[0.3, 0, 0]]),  # open mouth
            'eyes_pose': np.zeros_like(self.flame_params['eyes_pose'][:1]),
            'shape': self.flame_params['shape'],
            'expr': np.zeros_like(self.flame_params['expr'][:1]),
        }
        if 'static_offset' in self.flame_params:
            flame_param['static_offset'] = self.flame_params['static_offset']

        cano_flame_param_path = self.tgt_folder / 'canonical_flame_param.npz'
        print(f"Writing canonical FLAME parameters to: {cano_flame_param_path}")
        write_data({cano_flame_param_path: flame_param})

    @staticmethod
    def write_flame_param(tgt_folder, flame_param_path, flame_params, tid):
        params = {
            'translation': flame_params['translation'][[tid]],
            'rotation': flame_params['rotation'][[tid]],
            'neck_pose': flame_params['neck_pose'][[tid]],
            'jaw_pose': flame_params['jaw_pose'][[tid]],
            'eyes_pose': flame_params['eyes_pose'][[tid]],
            'shape': flame_params['shape'],
            'expr': flame_params['expr'][[tid]],
        }

        if 'static_offset' in flame_params:
            params['static_offset'] = flame_params['static_offset']
        if 'dynamic_offset' in flame_params:
            params['dynamic_offset'] = flame_params['dynamic_offset'][[tid]]

        path2data = {tgt_folder / flame_param_path: params}
        write_data(path2data)

class MaskFromFLAME:
    def __init__(self, cfg_model: ModelConfig, tgt_folder, background_color: str) -> None:
        background_color = self.cfg_data.background_color if background_color is None else background_color
        if background_color == 'white':
            self.background_tensor = torch.tensor([255, 255, 255]).byte()
        elif background_color == 'black':
            self.background_tensor = torch.tensor([0, 0, 0]).byte()
        else:
            raise ValueError(f"Unknown background color: {background_color}")

        dataset = NeRFDataset(
            root_folder=tgt_folder,
            division=None,
            camera_convention_conversion=None,
            target_extrinsic_type='w2c',
            use_fg_mask=True,
            use_flame_param=True,
        )
        self.dataloader = DataLoader(dataset, shuffle=False, batch_size=None, collate_fn=None, num_workers=0)

        self.flame_model = FlameHead(cfg_model.n_shape, cfg_model.n_expr, add_teeth=True)

        self.mesh_renderer = NVDiffRenderer(use_opengl=False)

    @torch.no_grad()
    def write(self):
        t2verts = {}
        worker_args = []
        print(f"Generating masks from FLAME...")
        for i, frame in enumerate(tqdm(self.dataloader)):

            # get FLAME vertices
            timestep = frame['timestep_index']
            if timestep not in t2verts:
                t2verts[timestep] = infer_flame_params(self.flame_model, frame['flame_param'], [0]).cuda()
            verts = t2verts[timestep]

            # render to get forground mask
            RT = frame['extrinsics'].cuda()[None]
            K = frame['intrinsics'].cuda()[None]
            h = frame['image_height']
            w = frame['image_width']

            # mask = self.get_mask(verts, RT, K, h, w)
            mask = self.get_mask_tilted_line(verts, RT, K, h, w)

            # edit the image and mask with dilated FLAME mask
            img = frame['image'].cuda()
            img = img * mask[:, :, None] + self.background_tensor.cuda()[None, None, :] * (1-mask)[:, :, None]

            # overwrite the original images
            path2data = {
                str(frame['image_path']): img.byte().cpu().numpy(),
            }

            if 'fg_mask_path' in frame and 'fg_mask' in frame:
                fg_mask = frame['fg_mask'].cuda()
                fg_mask = fg_mask * mask

                # overwrite the original masks
                path2data.update({
                    str(frame['fg_mask_path']): fg_mask.byte().cpu().numpy(),
                })

                #  # write to new folder
                # path2data.update({
                #     str(frame['fg_mask_path']).replace('fg_masks', 'fg_masks_'): fg_mask.byte().cpu().numpy(),
                # })

            write_data(path2data)
            worker_args.append([path2data])

            #--- no threading
            # if len(worker_args) > 0:
            #     write_data(path2data)

            #--- threading
            if len(worker_args) == max_threads or i == len(self.dataloader)-1:
                with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                    futures = [executor.submit(write_data, *args) for args in worker_args]
                    concurrent.futures.wait(futures)
                worker_args = []

    def get_mask(self, verts, RT, K, h, w):
        faces = self.flame_model.faces.cuda()
        out_dict = self.mesh_renderer.render_without_texture(verts, faces, RT, K, (h, w))

        rgba_mesh = out_dict['rgba'].squeeze(0)  # (H, W, C)
        mask_mesh = rgba_mesh[..., 3]  # (H, W)
        
        # get the bottom line of the neck and disable mask for the upper part
        verts_clip = out_dict['verts_clip'][0]
        verts_ndc = verts_clip[:, :3] / verts_clip[:, -1:]
        xy = verts_ndc[:, :2]
        xy[:, 1] = -xy[:, 1]
        xy = (xy * 0.5 + 0.5) * torch.tensor([[h, w]]).cuda()
        vid_ring = self.flame_model.mask.get_vid_by_region(['neck_top'])
        xy_ring = xy[vid_ring]
        bottom_line = int(xy_ring[:, 1].min().item())

        mask = mask_mesh.clone()
        mask[:bottom_line] = 1

        # anti-aliasing with gaussian kernel
        k = int(0.02 * w)//2 * 2 + 1
        blur = torchvision.transforms.GaussianBlur(k, sigma=k)
        mask = blur(mask[None])[0]  #.clamp(0, 1)
        return mask

    def get_mask_tilted_line(self, verts, RT, K, h, w):
        verts_ndc = self.mesh_renderer.world_to_ndc(verts, RT, K, (h, w), flip_y=True) 

        verts_xy = verts_ndc[0, :, :2]
        verts_xy = (verts_xy * 0.5 + 0.5) * torch.tensor([w, h]).cuda()

        verts_xy_left = verts_xy[self.flame_model.mask.get_vid_by_region(['neck_right_point'])]
        verts_xy_right = verts_xy[self.flame_model.mask.get_vid_by_region(['neck_left_point'])]
        verts_xy_bottom = verts_xy[self.flame_model.mask.get_vid_by_region(['front_middle_bottom_point_boundary'])]
        
        delta_xy = verts_xy_left - verts_xy_right
        assert (delta_xy[:, 0] != 0).all()
        k = delta_xy[:, 1] / delta_xy[:, 0]
        b = verts_xy_bottom[:, 1] - k * verts_xy_bottom[:, 0]

        x = torch.arange(w).cuda()
        y = torch.arange(h).cuda()
        yx = torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1)

        mask = ((k * yx[:, :, 1] + b - yx[:, :, 0]) > 0).float()

        # anti-aliasing with gaussian kernel
        k = int(0.03 * w)//2 * 2 + 1
        blur = torchvision.transforms.GaussianBlur(k, sigma=k)
        mask = blur(mask[None])[0]  #.clamp(0, 1)
        return mask

def infer_flame_params(flame_model: FlameHead, flame_params: Dict, indices:List):
    if 'static_offset' in flame_params:
        static_offset = flame_params['static_offset']
        if isinstance(static_offset, np.ndarray):
            static_offset = torch.tensor(static_offset)
    else:
        static_offset = None
    for k in flame_params:
        if isinstance(flame_params[k], np.ndarray):
            flame_params[k] = torch.tensor(flame_params[k])
    with torch.no_grad():
        ret = flame_model(
            flame_params['shape'][None, ...].expand(len(indices), -1),
            flame_params['expr'][indices],
            flame_params['rotation'][indices],
            flame_params['neck_pose'][indices],
            flame_params['jaw_pose'][indices],
            flame_params['eyes_pose'][indices],
            flame_params['translation'][indices],
            return_verts_cano=False,
            static_offset=static_offset,
        )
    verts = ret[0]
    return verts


    
def write_json(db, tgt_folder, division=None):
    fname = "transforms.json" if division is None else f"transforms_{division}.json"
    json_path = tgt_folder / fname
    print(f"Writing database: {json_path}")
    with open(json_path, "w") as f:
        json.dump(db, f, indent=4)

def write_data(path2data):
    for path, data in path2data.items():
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")

def split_json(tgt_folder: Path, train_ratio=0.7):
    db = json.load(open(tgt_folder / "transforms.json", "r"))
    
    # init db for each division
    db_train = {k: v for k, v in db.items() if k not in ['frames', 'timestep_indices', 'camera_indices']}
    db_train['frames'] = []
    db_val = deepcopy(db_train)
    db_test = deepcopy(db_train)

    # divide timesteps
    nt = len(db['timestep_indices'])
    assert 0 < train_ratio <= 1
    nt_train = int(np.ceil(nt * train_ratio))
    nt_test = nt - nt_train

    # record number of timesteps
    timestep_indices = sorted(db['timestep_indices'])
    db_train['timestep_indices'] = timestep_indices[:nt_train]
    db_val['timestep_indices'] = timestep_indices[:nt_train]  # validation set share the same timesteps with training set
    db_test['timestep_indices'] = timestep_indices[nt_train:]

    if len(db['camera_indices']) > 1:
        # when having multiple cameras, leave one camera for validation (novel-view sythesis)
        if 8 in db['camera_indices']:
            # use camera 8 for validation (front-view of the NeRSemble dataset)
            db_train['camera_indices'] = [i for i in db['camera_indices'] if i != 8]
            db_val['camera_indices'] = [8]
            db_test['camera_indices'] = db['camera_indices']
        else:
            # use the last camera for validation
            db_train['camera_indices'] = db['camera_indices'][:-1]
            db_val['camera_indices'] = [db['camera_indices'][-1]]
            db_test['camera_indices'] = db['camera_indices']
    else:
        # when only having one camera, we create an empty validation set
        db_train['camera_indices'] = db['camera_indices']
        db_val['camera_indices'] = []
        db_test['camera_indices'] = db['camera_indices']

    # fill data by timestep index
    range_train = range(db_train['timestep_indices'][0], db_train['timestep_indices'][-1]+1) if nt_train > 0 else []
    range_test = range(db_test['timestep_indices'][0], db_test['timestep_indices'][-1]+1) if nt_test > 0 else []
    for f in db['frames']:
        if f['timestep_index'] in range_train:
            if f['camera_index'] in db_train['camera_indices']:
                db_train['frames'].append(f)
            elif f['camera_index'] in db_val['camera_indices']:
                db_val['frames'].append(f)
            else:
                raise ValueError(f"Unknown camera index: {f['camera_index']}")
        elif f['timestep_index'] in range_test:
            db_test['frames'].append(f)
            assert f['camera_index'] in db_test['camera_indices'], f"Unknown camera index: {f['camera_index']}"
        else:
            raise ValueError(f"Unknown timestep index: {f['timestep_index']}")
    
    write_json(db_train, tgt_folder, division='train')
    write_json(db_val, tgt_folder, division='val')
    write_json(db_test, tgt_folder, division='test')

def load_config(src_folder: Path):
    config_path = src_folder / "config.yml"
    if not config_path.exists():
        src_folder = sorted(src_folder.iterdir())[-1]
        config_path = src_folder / "config.yml"
    assert config_path.exists(), f"File not found: {config_path}"

    cfg = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    # assert isinstance(cfg, BaseTrackingConfig)
    return src_folder, cfg

def check_epoch(src_folder: Path, epoch: int):
    paths = [Path(p) for p in glob(str(src_folder / "tracked_flame_params*.npz"))]
    epochs = [int(p.stem.split('_')[-1]) for p in paths]
    if epoch == -1:
        index = np.argmax(epochs)
    else:
        try:
            index = epochs.index(epoch)
        except ValueError:
            raise ValueError(f"Could not find epoch {epoch} in {src_folder}")

def main(
        src_folder: Path, 
        tgt_folder: Path,
        subset: Optional[str]=None,
        scale_factor: Optional[float]=None,
        background_color: Optional[str]=None,
        flame_mode: Literal['mesh', 'param']='param',
        create_mask_from_mesh: bool=False,
        epoch: int=-1,
    ):
    print(f"Begin exportation from {src_folder}")
    assert src_folder.exists(), f"Folder not found: {src_folder}"
    src_folder, cfg = load_config(src_folder)

    check_epoch(src_folder, epoch)

    if epoch != -1:
        tgt_folder = Path(str(tgt_folder) + f"_epoch{epoch}")
    
    nerf_dataset_writer = NeRFDatasetWriter(cfg.data, tgt_folder, subset, scale_factor, background_color)
    nerf_dataset_writer.write()

    flame_dataset_writer = TrackedFLAMEDatasetWriter(cfg.model, src_folder, tgt_folder, mode=flame_mode, epoch=epoch)
    flame_dataset_writer.write()

    if create_mask_from_mesh:
        mask_generator = MaskFromFLAME(cfg.model, tgt_folder, background_color)
        mask_generator.write()

    split_json(tgt_folder)
    
    print("Finshed!")


if __name__ == "__main__":
    tyro.cli(main)