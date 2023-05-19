#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --time=80:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=dm2_quat_1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xc2627@nyu.edu
#SBATCH --output=dm_1.out
#SBATCH --gres=gpu:v100:1

#!/bin/bash
cd /scratch/xc2627/dm2_quat_1/script/


# # dadtaset type
DATASET=KITTI
# path to dataset
DATA_DIR=/scratch/xc2627/dm2_quat_1/data/KITTI
# trajectiory file name
TRAJ=0018
# path to init pose
INIT=$DATA_DIR/$TRAJ/prior/init_pose.npy
# path to pairwise pose
PAIRWISE=$DATA_DIR/$TRAJ/prior/pairwise_pose.npy
# experiment name
NAME=KITTI_0018
# training epochs
EPOCH=160
# loss function
LOSS=bce_ch_eu
# number of points sampled from line-of-sight
N=10
# logging interval
LOG=1
# subsample rate
VOXEL=1
# goupr size
G_SIZE=8
# learning rate
LR=0.00005
# chamfer loss weight
ALPHA=0.1
# euclidean loss weight
BETA=0.1


singularity exec --nv --overlay /scratch/xc2627/overlay-15GB-500K.ext3:rw /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /scratch/xc2627/env.sh;
conda activate /scratch/xc2627/miniconda3/envs/dm2_v1;
CUDA_VISIBLE_DEVICES=0 python train.py --alpha $ALPHA --beta $BETA --lr $LR --name $NAME -d $DATA_DIR -t ${TRAJ} -i $INIT -p $PAIRWISE -e $EPOCH -l $LOSS -n $N -v $VOXEL --log_interval $LOG  --group_size $G_SIZE --dataset $DATASET"