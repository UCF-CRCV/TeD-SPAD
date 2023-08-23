import numpy as np
import os
import glob
import numpy as np
from scipy.io import loadmat
from os import walk


'''
FOR XD Violence
'''

rgb_list_file ='xd-i3d-test.list'
file_list = list(open(rgb_list_file))
gt_file = '/path/to/XD-Violence/annotations.txt'
temporal_anno = open(gt_file, 'r').read().splitlines()
# print(temporal_anno)

num_frame = 0
gt = []
for idx, file in enumerate(file_list):

    features = np.load(file.strip('\n').replace('_mgfn', '_ours'), allow_pickle=True)
    # features = [t.cpu().detach().numpy() for t in features]
    features = np.array(features, dtype=np.float32)
    num_frame = features.shape[0] * 32 # HERE CHANGE THE LENGTH OF CLIP
    count = 0
    if idx < 300: # if it's normal
        # print('hello')
        for i in range(0, num_frame):
            gt.append(0.0)
            count+=1

    else: #if it's abnormal # get the name from temporal file
        anno = [x for x in temporal_anno if os.path.basename(file)[:-5] in x][0]
        anom = anno.split(' ')[1:]
        anomalies = []
        for i in range(int(len(anom)/2)):
            anomalies.append(anom[i*2] + ' ' + anom[i*2+1])

        start_idx = 0
        for anomaly in anomalies:
            anomaly_start, anomaly_end = anomaly.split(' ')
            for i in range(start_idx, int(anomaly_start)):
                gt.append(0.0)
                count += 1
            if int(anomaly_end) > num_frame:
                anomaly_end = num_frame
            for i in range(int(anomaly_start), int(anomaly_end)):
                gt.append(1.0)
                count += 1
            start_idx = int(anomaly_end)

        for i in range(int(anomaly_end), num_frame):
            gt.append(0.0)
            count += 1


    if count != num_frame:
        print(file)
        print('Num of frames is not correct!!')
        remainder = count - num_frame
        print(remainder)
        if remainder > 0 and remainder < 16:
            print('Fixing...')
            del gt[-remainder:]
            continue

        exit(1)


output_file = 'gt-xd-ours.npy'
gt = np.array(gt, dtype=float)
np.save(output_file, gt)
print(len(gt))
