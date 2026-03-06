import os

import cv2
import numpy as np
from PIL import Image

selected_indices_old = [
    2311,
    2416,
    2437,
    2460,
    2495,
    2518,
    2520,
    2627,
    4285,
    4315,
    6223,
    6457,
    6597,
    6642,
    6974,
    7054,
    7064,
    7182,
    7303,
    7334,
    7351,
    7368,
    7374,
    7493,
    7503,
    7626,
    8443,
    8562,
    8597,
    8701,
    8817,
    8953,
    11213,
    11261,
    11317,
    11384,
    11600,
    11755,
    11852,
    11891,
    11945,
    12010,
    12354,
    12534,
    12736,
    12880,
    12892,
    13004,
    13323,
    13371,
    13534,
    13575,
    14874,
    14949,
    14977,
    15052,
    15076,
    15291,
    15620,
    15758,
    16309,
    16325,
    16348,
    16390,
    16489,
    16665,
    16891,
    17147,
    17183,
    17488,
    17549,
    17657,
    17932,
    19661,
    20162,
    20200,
    20238,
    20286,
    20432,
    20834,
    20954,
    21015,
    21036,
    21117,
    21299,
    21611,
    21632,
    21649,
    22722,
    22759,
    22873,
    23028,
    23033,
    23082,
    23187,
    23232,
    23302,
    23413,
    23430,
    23446,
    23457,
    23548,
    23636,
    32060,
    32245,
]

selected_indices = list()
with open('/home/gyalex/Desktop/face_anno.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        hh = line.strip().split()
        if len(hh) > 0:
            pid = hh[0].find('.')
            if pid != -1:
                s = hh[0][pid+1:len(hh[0])]
                print(s)
                selected_indices.append(int(s))
            
f.close()

dir = '/media/gyalex/Data/face_ldk_dataset/MHC_LightingPreset_Portrait_RT_0_19/MHC_LightingPreset_Portrait_RT_seq_000015'

for idx in range(500):
    img = os.path.join(dir, "view_1/MHC_LightingPreset_Portrait_RT_seq_000015_FinalImage_" + str(idx).zfill(4) + ".jpeg")
    lmd = os.path.join(dir, "mesh/mesh_screen" + str(idx+5).zfill(7) + ".npy")

    img = cv2.imread(img)
    # c = 511 / 2
    # lmd = np.load(lmd) * c + c
    # lmd[:, 1] = 511 - lmd[:, 1]
    lmd = np.load(lmd)[selected_indices]
    for i in range(lmd.shape[0]):
        p = lmd[i]
        x, y = round(float(p[0])), round(float(p[1]))
        print(p)
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow('win', img)
    cv2.waitKey(0)