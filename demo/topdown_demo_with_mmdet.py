# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
import tempfile
from argparse import ArgumentParser

import json_tricks as json
import mmcv
import mmengine
import numpy as np
import sys 
sys.path.append('/research/d4/rshr/yyliu/code/mmpose/')
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def convtojson(id,subid,name,pred_instances):
    kpts=pred_instances['keypoints'][0].tolist()
    scores=pred_instances['keypoint_scores'][0].tolist()
    bbox=pred_instances['bboxes'][0].tolist()
    area=bbox[2]*bbox[3]
    curimg={'id':id,'file_name':name,'width':640,'height':480}
    curanno={'id':subid,'image_id':id,'category_id': 1,'num_keypoints': 17,\
    'area':area,'bbox':bbox}
    # 多人场景?
    '''kptformat=[]
    for k in range(len(kpts)):
        v=kpts[k]
        v.append(scores[k])
        kptformat.append(v)'''
    kptformat=[kpts[k].append(scores[k]) for k in range(len(kpts))]
    curanno['keypoints']=kpts
    new_train['images'].append(curimg)
    new_train['annotations'].append(curanno)
    #return pred_instances
def process_one_image(args, img_path, detector, pose_estimator, visualizer,
                      show_interval):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img_path)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img_path, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    img = mmcv.imread(img_path, channel_order='rgb')

    out_file = None
    if args.output_root:
        out_file = f'{args.output_root}/{os.path.basename(img_path)}'

    visualizer.add_datasample(
        'result',
        img,
        data_sample=data_samples,
        draw_gt=False,
        draw_heatmap=args.draw_heatmap,
        draw_bbox=args.draw_bbox,
        show_kpt_idx=args.show_kpt_idx,
        show=args.show,
        wait_time=show_interval,
        out_file=out_file,
        kpt_score_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)

def main_parse_args():
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    #assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'
    
    return args
def main(args,imgsavepath, predsavepath):
    """Visualize the demo images.

    Using mmdet to detect the human.
    """

    
    # build pose estimator
    
    input_type = mimetypes.guess_type(args.input)[0].split('/')[0]
    if input_type == 'image':
        pred_instances = process_one_image(
            args,
            imgsavepath,
            detector,
            pose_estimator,
            visualizer,
            show_interval=0)
        for ipred in pred_instances:
            global subidx
            global idx
            convtojson(idx,subidx,os.path.basename(imgsavepath),ipred)
            subidx+=1
        pred_instances_list = split_instances(pred_instances)

    elif input_type == 'video':
        tmp_folder = tempfile.TemporaryDirectory()
        video = mmcv.VideoReader(args.input)
        progressbar = mmengine.ProgressBar(len(video))
        video.cvt2frames(tmp_folder.name, show_progress=False)
        output_root = args.output_root
        args.output_root = tmp_folder.name
        pred_instances_list = []

        for frame_id, img_fname in enumerate(os.listdir(tmp_folder.name)):
            pred_instances = process_one_image(
                args,
                f'{tmp_folder.name}/{img_fname}',
                detector,
                pose_estimator,
                visualizer,
                show_interval=1)

            progressbar.update()
            pred_instances_list.append(
                dict(
                    frame_id=frame_id,
                    instances=split_instances(pred_instances)))

        if output_root:
            mmcv.frames2video(
                tmp_folder.name,
                f'{output_root}/{os.path.basename(args.input)}',
                fps=video.fps,
                fourcc='mp4v',
                show_progress=False)
        tmp_folder.cleanup()

    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:
        '''with open(predsavepath, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')'''
        print(f'predictions have been saved at {predsavepath}')


if __name__ == '__main__':
    #main()
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
    idx=700000
    subidx=700000
    inpath='/research/d4/rshr/yyliu/code/movenet/data/23321testdata/videoImages'
    files=os.listdir(inpath)
    args=main_parse_args()
    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # init visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)
    for i in files:
        name=i.split('.')[0]
        imgpath='/research/d4/rshr/yyliu/code/movenet/data/23220minhaodata/220newdata/152.jpg'
        main(args,os.path.join(inpath,i),os.path.join(args.output_root,'result_'+name+'.json'))
        print(os.path.join(inpath,i))
        #main(idx,subidx,args,imgpath,os.path.join(args.output_root,'result_'+name+'.json'))
        idx+=1
        subidx+=1
    with open('/research/d4/rshr/yyliu/code/movenet/data/23321testdata/litehrnet.json','w') as f:
        json.dump(new_train,f)
        