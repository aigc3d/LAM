# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


from typing import Optional, Literal, List
from copy import deepcopy
import json
import tyro
from pathlib import Path
import shutil
import random


class NeRFDatasetAssembler:
    def __init__(self, src_folders: List[Path], tgt_folder: Path, division_mode: Literal['random_single', 'random_group', 'last']='random_group'):
        self.src_folders = src_folders
        self.tgt_folder = tgt_folder
        self.num_timestep = 0

        # use the subject name as the random seed to sample the test sequence
        subjects = [sf.name.split('_')[0] for sf in src_folders]
        for s in subjects:
            assert s == subjects[0], f"Cannot combine datasets from different subjects: {subjects}"
        subject = subjects[0]
        random.seed(subject)

        if division_mode == 'random_single':
            self.src_folders_test = [self.src_folders.pop(int(random.uniform(0, 1) * len(src_folders)))]
        elif division_mode == 'random_group':
            # sample one sequence as the test sequence every `group_size` sequences
            self.src_folders_test = []
            num_all = len(self.src_folders)
            group_size = 10
            num_test = max(1, num_all // group_size)
            indices_test  = []
            for gi in range(num_test):
                idx = min(num_all - 1, random.randint(0, group_size - 1) + gi * group_size)
                indices_test.append(idx)

            for idx in indices_test:
                self.src_folders_test.append(self.src_folders.pop(idx))
        elif division_mode == 'last':
            self.src_folders_test = [self.src_folders.pop(-1)]
        else:
            raise ValueError(f"Unknown division mode: {division_mode}")

        self.src_folders_train = self.src_folders

    def write(self):
        self.combine_dbs(self.src_folders_train, division='train')
        self.combine_dbs(self.src_folders_test, division='test')

    def combine_dbs(self, src_folders, division: Optional[Literal['train', 'test']] = None):
        db = None
        for i, src_folder in enumerate(src_folders):
            dbi_path = src_folder / "transforms.json"
            assert dbi_path.exists(), f"Could not find {dbi_path}"
            # print(f"Loading database: {dbi_path}")
            dbi = json.load(open(dbi_path, "r"))
           
            dbi['timestep_indices'] = [t + self.num_timestep for t in dbi['timestep_indices']]
            self.num_timestep += len(dbi['timestep_indices'])
            for frame in dbi['frames']:
                # drop keys that are irrelevant for a combined dataset
                frame.pop('timestep_index_original')
                frame.pop('timestep_id')

                # accumulate timestep indices
                frame['timestep_index'] = dbi['timestep_indices'][frame['timestep_index']]

                # complement the parent folder
                frame['file_path'] = str(Path('..') / Path(src_folder.name) / frame['file_path'])
                frame['flame_param_path'] = str(Path('..') / Path(src_folder.name) / frame['flame_param_path'])
                frame['fg_mask_path'] = str(Path('..') / Path(src_folder.name) / frame['fg_mask_path'])
            
            if db is None:
                db = dbi
            else:
                db['frames'] += dbi['frames']
                db['timestep_indices'] += dbi['timestep_indices']
            
        if not self.tgt_folder.exists():
            self.tgt_folder.mkdir(parents=True)
        
        if division == 'train':
            # copy the canonical flame param
            cano_flame_param_path = src_folders[0] / "canonical_flame_param.npz"
            tgt_flame_param_path = self.tgt_folder / f"canonical_flame_param.npz"
            print(f"Copying canonical flame param: {tgt_flame_param_path}")
            shutil.copy(cano_flame_param_path, tgt_flame_param_path)

            # leave one camera for validation
            db_train = {k: v for k, v in db.items() if k not in ['frames', 'camera_indices']}
            db_train['frames'] = []
            db_val = deepcopy(db_train)

            if len(db['camera_indices']) > 1:
                # when having multiple cameras, leave one camera for validation (novel-view sythesis)
                if 8 in db['camera_indices']:
                    # use camera 8 for validation (front-view of the NeRSemble dataset)
                    db_train['camera_indices'] = [i for i in db['camera_indices'] if i != 8]
                    db_val['camera_indices'] = [8]
                else:
                    # use the last camera for validation
                    db_train['camera_indices'] = db['camera_indices'][:-1]
                    db_val['camera_indices'] = [db['camera_indices'][-1]]
            else:
                # when only having one camera, we create an empty validation set
                db_train['camera_indices'] = db['camera_indices']
                db_val['camera_indices'] = []

            for frame in db['frames']:
                if frame['camera_index'] in db_train['camera_indices']:
                    db_train['frames'].append(frame)
                elif frame['camera_index'] in db_val['camera_indices']:
                    db_val['frames'].append(frame)
                else:
                    raise ValueError(f"Unknown camera index: {frame['camera_index']}")
                
            write_json(db_train, self.tgt_folder, 'train')
            write_json(db_val, self.tgt_folder, 'val')

            with open(self.tgt_folder / 'sequences_trainval.txt', 'w') as f:
                for folder in src_folders:
                    f.write(folder.name + '\n')
        else:
            db['timestep_indices'] = sorted(db['timestep_indices'])
            write_json(db, self.tgt_folder, division)

            with open(self.tgt_folder / f'sequences_{division}.txt', 'w') as f:
                for folder in src_folders:
                    f.write(folder.name + '\n')

    
def write_json(db, tgt_folder, division=None):
    fname = "transforms.json" if division is None else f"transforms_{division}.json"
    json_path = tgt_folder / fname
    print(f"Writing database: {json_path}")
    with open(json_path, "w") as f:
        json.dump(db, f, indent=4)
    
def main(
        src_folders: List[Path],
        tgt_folder: Path,
        division_mode: Literal['random_single', 'random_group', 'last']='random_group',
    ):
    incomplete = False
    print("==== Begin assembling datasets ====")
    print(f"Division mode: {division_mode}")
    for src_folder in src_folders:
        try:
            assert src_folder.exists(), f"Error: could not find {src_folder}"
            assert src_folder.parent == tgt_folder.parent, "All source folders must be in the same parent folder as the target folder"
            # print(src_folder)
        except AssertionError as e:
            print(e)
            incomplete = True

    if incomplete:
        return

    nerf_dataset_assembler = NeRFDatasetAssembler(src_folders, tgt_folder, division_mode)
    nerf_dataset_assembler.write()

    print("Done!")


if __name__ == "__main__":
    tyro.cli(main)
