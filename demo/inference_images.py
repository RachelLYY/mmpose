
from re import A
import mmcv
import sys 
sys.path.append('/research/d4/rshr/yyliu/code/mmpose/')
from mmpose.apis.inference import init_pose_model, inference_top_down_pose_model,vis_pose_result
import numpy as np
import cv2
import os
import json

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='sample images from one path to another')
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--src',default='/research/d4/rshr/yyliu/code/movenet/data/forjx/testimages/vip011@vip.com/20220704_221926_2101_R_0',
        type=str, help='src image file path')
    parser.add_argument('--dst', type=str,help='dst image file path')
    #
    # parser.add_argument('--dst_json', type=str,help='dst json file path')
    #parser.add_argument('--ratio',default=0.15,type=float,help='ratio to sample')
    args = parser.parse_args()
    return args
def main(): # /research/d4/rshr/yyliu/code/Lite-HRNet/configs/top_down/lite_hrnet/coco/hrnet_w48_coco_384x288.py
    config_path='/research/d4/rshr/yyliu/code/Lite-HRNet/configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192.py'
    # /research/d4/rshr/yyliu/code/Lite-HRNet/work_dirs/326_18_256_192_augmentation90/latest.pth
    # /research/d4/rshr/yyliu/code/Lite-HRNet/weights/hrnet_w48-8ef0771d.pth
    ckpt_path='/research/d4/rshr/yyliu/code/Lite-HRNet/work_dirs/705alldata/latest.pth'

    args = parse_args()
    inpath=args.src
    out=args.dst
    #dst_json=args.dst_json
    device='cpu'
    model = init_pose_model(config_path, ckpt_path)
    files=os.listdir(inpath)

    example_json = '/research/d4/rshr/yyliu/data/coco/annotations/person_keypoints_val2017.json'
    #lxjson='/research/d4/rshr/yyliu/code/movenet/data/nlx_add_aist/annotations/nlx_add_aist_val.json'
    with open(example_json) as f:
        example = json.load(f)
    #with open(lxjson) as f1:
    #    lx=json.load(f1)

    new_train={}
    new_train['info'] = example['info']
    new_train['licenses'] = example['licenses']
    new_train['categories'] = example['categories']
    new_train['images'] = []
    new_train['annotations'] = []

    idx=616640
    outputs=[]
    for i in files:
        img_path=os.path.join(inpath,i)
        name=i.split('.')[0]
        #img_path='/research/d4/rshr/yyliu/data/active/images/01_01_20210309170634.jpg'
        results= inference_top_down_pose_model(model, img_path) # results[0][0].keys() dict_keys(['bbox', 'keypoints'])
        results[0][0]['file_name']=i
        outputs.append(results[0][0])
        out_img=vis_pose_result(model,img_path,results[0])
        # results[0][0]
        a1={}
        a1['id']=idx
        a1['image_id']=idx
        a1['category_id']=1
        a1['area']=467218.2078865287
        a1['bbox']=results[0][0]['bbox'].tolist()
        kpt=[]
        for i in results[0][0]['keypoints']:
            ni=i.tolist()
            ni[2]=2.0
            kpt.append(ni[0])
            kpt.append(ni[1])
            kpt.append(ni[2])
        a1['keypoints']=kpt
        a1['num_keypoints']=17
        new_train['annotations'].append(a1)
        
        cur_img_anno={}
        # 'id': 608290, 'file_name': 'lx_76.jpg', 'width': 360, 'height': 480
        cur_img_anno['id']=idx
        cur_img_anno['file_name']=name+'.jpg'
        cur_img_anno['width']=360
        cur_img_anno['height']=480
        new_train['images'].append(cur_img_anno)
        idx+=1
        cv2.imwrite(os.path.join(out,name+'.jpg'),out_img)
        idx+=1
    '''with open('/research/d4/rshr/yyliu/code/movenet/data/LXtest/612_1coco_nlx_sampled_lied.json','w') as f1:
        json.dump(new_train,f1)'''

    #print(f'\nwriting results to {args.out}')
    
    #mmcv.dump(outputs, dst_json)
if __name__ == '__main__':
    main() 
## Visualize Results
'''hms =heatmaps[0]['heatmap']

result = results[0]
keypoints = ([np.array([v[0],v[1]]) for v in result['keypoints']])

#Plot image and keypoints
plt.figure()
plt.scatter(*zip(*keypoints))
#plt.savefig('./data/scatter.jpg')
plt.imshow(result['image'])
plt.show()

#Plot heatmaps in a grid
n_hms = np.shape(hms)[1]
f, axarr = plt.subplots(3, 4, figsize=(15,15))
this_col=0
for idx in range(n_hms):
    this_hm = hms[0,idx,:,:]
    row = idx % 4
    this_ax = axarr[this_col, row]
    this_ax.set_title(f'{idx}')
    hm_display = this_ax.imshow(this_hm, cmap='jet', vmin=0, vmax=1)
    if row == 3:
        this_col += 1

cb=f.colorbar(hm_display, ax=axarr)'''

