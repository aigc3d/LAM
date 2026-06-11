import csv
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm

tsv_file = '/apdcephfs/share_1134483/charlinzhou/datas/ADNet/WFLW/test.tsv'
save_folder = '/apdcephfs/share_1134483/charlinzhou/datas/ADNet/_WFLW/'

save_tags = ['largepose', 'expression', 'illumination', 'makeup', 'occlusion', 'blur']
save_tags = ['test_{}_metadata.tsv'.format(t) for t in save_tags]
save_files = [osp.join(save_folder, t) for t in save_tags]
save_files = [open(f, 'w', newline='') for f in save_files]

landmark_num = 98
items = pd.read_csv(tsv_file, sep="\t")

items_num = len(items)
for index in tqdm(range(items_num)):
    image_path = items.iloc[index, 0]
    landmarks_5pts = items.iloc[index, 1]
    # landmarks_5pts = np.array(list(map(float, landmarks_5pts.split(","))), dtype=np.float32).reshape(5, 2)
    landmarks_target = items.iloc[index, 2]
    # landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(landmark_num, 2)
    scale = items.iloc[index, 3]
    center_w, center_h = items.iloc[index, 4], items.iloc[index, 5]
    if len(items.iloc[index]) > 6:
        tags = np.array(list(map(lambda x: int(float(x)), items.iloc[index, 6].split(","))))
    else:
        tags = np.array([])
    assert len(tags) == 6, '{} v.s. 6'.format(len(tags))
    for k, tag in enumerate(tags):
        if tag == 1:
            save_file = save_files[k]
            tsv_w = csv.writer(save_file, delimiter='\t')
            tsv_w.writerow([image_path, landmarks_5pts, landmarks_target, scale, center_w, center_h])

print('Done!')
