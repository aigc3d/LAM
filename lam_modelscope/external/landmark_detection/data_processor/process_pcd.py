import os
import cv2
import numpy as np
import open3d as o3d
# import pyrender
# from pyrender import mesh, DirectionalLight, Material, PerspectiveCamera

os.environ['__GL_THREADED_OPTIMIZATIONS'] = '1'

cord_list = []
with open('./cord.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        m = line.split()
        x = int(m[0])
        y = int(m[1])

        x = 1000 - x
        y = 1000 - y

        cord_list.append([x, y])


# 假设TXT文件的路径
output_folder = '/media/gyalex/Data/face_det_dataset/rgbd_data/rgbd'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for idx in range(32, 33):
    txt_file_path = '/media/gyalex/Data/face_det_dataset/rgbd_data/PointImage'+ str(idx) + '.txt'
    _, name = os.path.split(txt_file_path)
    print(txt_file_path)

    with open(txt_file_path, 'r') as file:
        points   = []
        rgb_list = []
        ori_rgb_list = []
        normal_list = []

        # 逐行读取数据
        for line in file:
            # 去除行尾的换行符并分割字符串
            x, y, z, r, g, b, nx, ny, nz, w = line.split()
            # 将字符串转换为浮点数
            x = float(x)
            y = float(y)
            z = float(z)
            r = float(r)
            g = float(g)
            b = float(b)
            nx = float(nx)
            ny = float(ny)
            nz = float(nz)
            # 将点添加到列表中
            points.append((x, y, z))
            rgb_list.append((r/255.0, g/255.0 , b/255.0))
            normal_list.append((nx, ny, nz))

            ori_r = int(r)
            ori_g = int(g)
            ori_b = int(b)
            ori_rgb_list.append((ori_r, ori_g , ori_b))

    np_points  = np.asarray(points)

    np_points_a = np_points

    np_colors  = np.asarray(rgb_list)
    np_normals = np.asarray(normal_list)

    np_colors_ori = np.asarray(ori_rgb_list)

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(np_points)
    pcd.colors  = o3d.utility.Vector3dVector(np_colors)
    pcd.normals = o3d.utility.Vector3dVector(np_normals)

    map_dict = {}
    
    image = np.ones((1000, 1000, 3),dtype=np.uint8)*255
    for i in range(np.array(pcd.points).shape[0]):
        x = np.array(pcd.points)[i,0]+400
        y = np.array(pcd.points)[i,1]+400

        image[int(x),int(y),:] = (np.array(pcd.colors)[i,:]*255).astype(np.uint8)
        image[int(x+1),int(y),:] = (np.array(pcd.colors)[i,:]*255).astype(np.uint8)
        image[int(x),int(y+1),:] = (np.array(pcd.colors)[i,:]*255).astype(np.uint8)
        image[int(x-1),int(y),:] = (np.array(pcd.colors)[i,:]*255).astype(np.uint8)
        image[int(x),int(y-1),:] = (np.array(pcd.colors)[i,:]*255).astype(np.uint8)

        map_dict[str(int(x)) + '_' + str(int(y))] = i
        map_dict[str(int(x+1)) + '_' + str(int(y))] = i
        map_dict[str(int(x)) + '_' + str(int(y+1))] = i
        map_dict[str(int(x-1)) + '_' + str(int(y))] = i
        map_dict[str(int(x)) + '_' + str(int(y-1))] = i

        # if [int(y), int(x)] in cord_list:
        #     image[int(x),int(y),:] = np.array([0, 255, 0])

        # if [int(y), int(x+1)] in cord_list:
        #     image[int(x+1),int(y),:] = np.array([0, 255, 0])

        # if [int(y+1), int(x)] in cord_list:
        #     image[int(x),int(y+1),:] = np.array([0, 255, 0])

        # if [int(y), int(x-1)] in cord_list:
        #     image[int(x-1),int(y),:] = np.array([0, 255, 0])

        # if [int(y-1), int(x)] in cord_list:
        #     image[int(x),int(y-1),:] = np.array([0, 255, 0])

        # if [int(y-1), int(x-1)] in cord_list:
        #     image[int(x-1),int(y-1),:] = np.array([0, 255, 0])

        # if [int(y+1), int(x+1)] in cord_list:
        #     image[int(x+1),int(y+1),:] = np.array([0, 255, 0])

    h_list = []
    for m in cord_list:
        a, b = m[0], m[1]
        c = image[int(b),int(a),:][0]

        flag = False

        if image[int(b),int(a),:][1] != 255:
            h_list.append(str(int(b))+'_'+str(int(a)))
            flag = True
        else:
            if image[int(b)-2,int(a)-2,:][1] != 255:
                h_list.append(str(int(b)-2)+'_'+str(int(a)-2))
                flag = True
            elif image[int(b)+2,int(a)+2,:][1] != 255:
                h_list.append(str(int(b)+2)+'_'+str(int(a)+2))
                flag = True
            elif image[int(b),int(a)-3,:][1] != 255:
                h_list.append(str(int(b))+'_'+str(int(a)-3))
                flag = True
            
        # if flag == False:
        #     cc = image[int(b),int(a),:][1]
    
    # cv2.circle(image, (465,505), 2, (0, 255, 0), -1)
    
    # cv2.imshow('win', image)
    # cv2.waitKey(0)

    with open('pid.txt', 'w') as f:
        for h in h_list:
            pid = map_dict[h]
            s = str(pid) + '\n'
            f.write(s)

            np_colors[pid,:] = np.array([0, 255, 0])

    f.close()

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points  = o3d.utility.Vector3dVector(np_points)
    pcd0.colors  = o3d.utility.Vector3dVector(np_colors)
    pcd0.normals = o3d.utility.Vector3dVector(np_normals)

    o3d.io.write_point_cloud('aa.ply', pcd0)


    mm = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image3 = cv2.flip(mm, -1)

    # cv2.imwrite('./rgb.png', image3)

with open('./cord.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        m = line.split()
        x = int(m[0])
        y = int(m[1])

        x = 1000 - x
        y = 1000 - y

        cv2.circle(image, (x,y), 2, (0, 255, 0), -1)

        idx = map_dict[str(x)+'_'+str(y)]

        a = 0

# cv2.imshow("win", image)
# cv2.waitKey(0)


    
    










    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.show()

    # save_pcd_path = os.path.join(output_folder, name[:-3]+'ply')
    # # o3d.io.write_point_cloud(save_pcd_path, pcd)

    # # render
    # import trimesh
    # # fuze_trimesh = trimesh.load('/home/gyalex/Desktop/PointImage32.obj')
    # # mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
    # mesh = pyrender.Mesh.from_points(np_points, np_colors_ori, np_normals)
    
    # import math
    # camera = PerspectiveCamera(yfov=math.pi / 3, aspectRatio=1.0)
    # camera_pose = np.array([[-1.0, 0.0, 0.0, 0], \
    #                         [0.0, 1.0, 0.0,  0],  \
    #                         [0.0, 0.0, -1.0, 0], \
    #                         [0.0, 0.0, 0.0,  1.0]])

    # # 创建场景
    # scene = pyrender.Scene()
    # scene.add(mesh)
    # scene.add(camera, pose=camera_pose)
    
    # # light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
    # # scene.add(light, pose=camera_pose)
    
    # # 渲染场景
    # renderer = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=1024)
    # color, depth = renderer.render(scene)
    
    # # # 设置场景和光源
    # # scene = pyrender.Scene()
    # # scene.add(point_cloud_mesh, 'point_cloud')
    # # camera = PerspectiveCamera(yfov=45.0, aspectRatio=1.0)
    # # scene.add(camera)

    # # # 渲染场景
    # # renderer = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=1024)
    # # color, depth = renderer.render(scene)
    
    # # 保存渲染结果为图片
    # import cv2
    # cv2.imshow('win', color)
    
    # rgb_img = cv2.imread('/media/gyalex/Data/face_det_dataset/rgbd_data/color_32.bmp')
    # cv2.imshow('win0', rgb_img)
    # cv2.waitKey(0)