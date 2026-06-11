import json
import glob
import sys
import os

data_root = sys.argv[1]
save_path = sys.argv[2]

all_hid_list = []
for hid in os.listdir(data_root):
    if hid.startswith("p"):
        hid = os.path.join(data_root, hid)
        all_hid_list.append(hid.replace(data_root + "/", ""))

print(f"len:{len(all_hid_list)}")
print(all_hid_list[:3])
with open(save_path, 'w') as fp:
    json.dump(all_hid_list, fp, indent=4)