import json
import glob
import sys
import os

data_root = sys.argv[1]
save_path = sys.argv[2]

all_img_list = []
for hid in os.listdir(data_root):
    all_view_imgs_dir = os.path.join(data_root, hid, "kinect_color")
    if not os.path.exists(all_view_imgs_dir):
        continue
    
    for view_id in os.listdir(all_view_imgs_dir):
        imgs_dir = os.path.join(all_view_imgs_dir, view_id)
        for img_path in glob.glob(os.path.join(imgs_dir, "*.png")):
            all_img_list.append(img_path.replace(data_root + "/", ""))

print(f"len:{len(all_img_list)}")
print(all_img_list[:3])
with open(save_path, 'w') as fp:
    json.dump(all_img_list, fp, indent=4)