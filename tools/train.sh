#!/bin/bash
#SBATCH --mail-user=yanyanliu@cuhk.edu.hk
#SBATCH --mail-type=END
#SBATCH --output=/research/d4/rshr/yyliu/code/mmpose/exp/0321train.log
#SBATCH --job-name=0321
#SBATCH --gres=gpu:2

set -x

# activate your environment
# submit job sbatch -q ex_batch -p batch_72h -c 40 -C highcpucount tools/train.sh 297966

source ~/.bash_profile
conda activate mmdeploy

# cd to your working directory (absolute path)
cd /research/d4/rshr/yyliu/code/mmpose
# nvidia-smi
# projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-t_8xb256-420e_coco-256x192.py
# command to run your code
CONFIG=${1:-"projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-t_0117alldata.py"}

#GPU=${2:-"train_rle"}
WORKDIR=${2:-"/research/d4/rshr/yyliu/code/mmpose/work_dirs/0321train"}
#RESUME=${3:-"/research/d4/rshr/yyliu/code/Lite-HRNet/weights/litehrnet_18_coco_384x288.pth"}

HOST=$(hostname -i)

/bin/bash /research/d4/rshr/yyliu/code/mmpose/tools/dist_train.sh ${CONFIG} 2 \
    --work-dir ${WORKDIR}
    #--resume-from ${RESUME}

# python /research/d4/rshr/yyliu/code/mmpose/tools/train.py projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-t_0117alldata.py --work-dir /research/d4/rshr/yyliu/code/mmpose/work_dirs/0321traintest