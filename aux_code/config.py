import os.path 

# Paths for UCF_Crimes dataset.
ucf_crimes_path = '/path/to/datasets/UCF_Crimes'
action_splits_path = os.path.join(ucf_crimes_path, 'Action_Regnition_splits')
class_idx_path = os.path.join(action_splits_path, 'ClassIDs.txt')

# Paths for XD dataset.
xd_path = '/path/to/datasets/XD_Violence'

# Paths for ShanghaiTech dataset.
shanghai_path = '/path/to/datasets/shanghaitech'

# Paths for VISPR dataset.
vispr_path = '/path/to/datasets/vispr_resized'

# Paths for UCF101 dataset.
ucf101_path = '/path/to/datasets/UCF101'
ucf101_class_mapping = os.path.join(ucf101_path, 'ucfTrainTestlist', 'action_classes.json')

# General paths.
saved_models_dir = os.path.join('saved_models')
logs = os.path.join('logs')

