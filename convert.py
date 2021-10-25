"""Script to convert HICO object indices to MS COCO 80 format"""

import enum
import json
from tqdm import tqdm
from copy import deepcopy
import numpy as np

fname = 'instances_train2015.json'

with open(fname, 'r') as f:
    a = json.load(f)

with open('coco80tohico80.json', 'r') as f:
    b = json.load(f)

coco = np.asarray([int(k) for k in b.keys()]) - 1
hico = np.asarray([v for v in b.values()])

order = np.argsort(coco)
convert = hico[order]

# Re-order object names
objects = a['objects'].copy()
objects = [objects[k] for k in convert]
a['objects'] = objects

order = np.argsort(hico)
convert = coco[order].tolist()

# Update correspondence table
corr = deepcopy(a['correspondence'])
for v in corr:
    v[1] = convert[v[1]]
a['correspondence'] = corr

annotation = a['annotation']
for v in tqdm(annotation):
    v['object'] = [convert[o] for o in v['object']]

with open('new_' + fname, 'w') as f:
    json.dump(a, f)
