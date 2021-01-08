# Documentation

### __`CLASS`__ pocket.data.HICODet(_root: str, anno_file: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None_)

HICO-DET dataset for human-object interaction detection. *\_\_len\_\_()* returns the number of images and *\_\_getitem\_\_()* fetches an image and the corresponding annotations. For string representations, *\_\_str\_\_()* returns the dataset information, and *\_\_repr\_\_()* returns instantiation arguments. Images without bounding box annotations will be skipped automatically during indexing.

`Parameters:`
* **root**: Root directory where images are downloaded to
* **anno_file**: Path to json annotation file
* **transform**: A function/transform that takes in an PIL image and returns a transformed version
* **target_transform**: A function/transform that takes in the target and transforms it
* **transforms**: A function/transform that takes input sample and its target as entry and returns a transformed version

`Methods`:
* \_\_getitem\_\_(_i: int_) -> tuple: Return a tuple of the transformed image and annotations. The annotations are formatted in the form of a Python dict with the following keys
    * boxes_h: List[list]
    * boxes_o: List[list]
    * hoi: List[int]
    * verb: List[int]
    * object: List[int]
* split(_ratio: float_) -> Tuple[HICODetSubset, HICODetSubset]: Split the dataset according to given ratio. 
    * __ratio__: The percentage of training set between 0 and 1
* filename(_idx: int_) -> str: Return the image file name given the index
* image_size(self, idx: int) -> Tuple[int, int]: Return the size (width, height) of an image

`Properties:`
* annotations -> List[dict]: All annotations for the dataset
* class_corr -> List[Tuple[int, int, int]]: Class correspondence matrix in zero-based index in the order of [*hoi_idx*, *obj_idx*, *verb_idx*]
* object_n_verb_to_interaction -> List[list]: The interaction classes corresponding to an object-verb pair. An interaction class index can be found by the object index and verb index (in the same order). Invalid combinations will return None.
* object_to_interaction -> List[list]: The interaction classes that involve each object type
* object_to_verb -> List[list]: The valid verbs for each object type
* anno_interaction -> List[int]: Number of annotated box pairs for each interaction class
* anno_object -> List[int]: Number of annotated box pairs for each object class
* anno_action -> List[int]: Number of annotated box pairs for each action class
* objects -> List[str]: Object names
* verbs -> List[str]: Verb (action) names
* interactions -> List[str]: Interaction names

`Examples:`
```python
>>> from pocket.data import HICODet
>>> # Instantiate the dataset by passing the directory of images and the path to the annotation file
>>> trainset = HICODet(root='./hico_20160224_det/images/train2015', anno_file='./instances_train2015.json')
>>> # Print the number of images with bounding box annotations
>>> len(trainset)
37633
>>> # Load an image (PIL) and its annotations (dict)
>>> image, annotation = trainset[0]
>>> image.show()
>>> # The annotation dict contains five keys
>>> # boxes_h: List[N] Human boxes in each of the N pairs
>>> # boxes_o: List[N] Object boxes in each of the N pairs
>>> # hoi: List[N] Index of the HOI in each of the N pairs
>>> # object: List[N] Index of the object in each of the N pairs
>>> # verb: List[N] Index of the verb in each of the N pairs
>>> annotation
{'boxes_h': [[208.0, 33.0, 427.0, 300.0], [213.0, 20.0, 438.0, 357.0], [206.0, 33.0, 427.0, 306.0], [209.0, 26.0, 444.0, 317.0]], 'boxes_o': [[59.0, 98.0, 572.0, 405.0], [77.0, 115.0, 583.0, 396.0], [61.0, 100.0, 571.0, 401.0], [59.0, 99.0, 579.0, 395.0]], 'hoi': [152, 153, 154, 155], 'object': [44, 44, 44, 44], 'verb': [72, 76, 87, 98]}
>>> # Print the natural language descriptions of some HOIs, objects and actions
>>> trainset.interactions[152]
'race motorcycle'
>>> trainset.object[44]
'motorcycle'
>>> trainset.verbs[72]
'race'
>>> # Visualise a box pair
>>> from pocket.utils import draw_box_pairs
>>> i = 1
>>> draw_box_pairs(image, annotation['boxes_h'][i], annotation['boxes_o'][i], width=4)
>>> # Split the dataset into subsets
>>> train, val = trainset.split(0.8)
>>> len(train)
30106
>>> len(val)
7527
```
