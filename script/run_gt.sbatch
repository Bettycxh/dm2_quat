#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=vis_gt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xc2627@nyu.edu
#SBATCH --output=vis_gt.out
#SBATCH --gres=gpu:v100:1

#!/bin/bash
cd /scratch/xc2627/dm2_quat_1/script/
singularity exec --nv --overlay /scratch/xc2627/overlay-7.5GB-300K.ext3:rw /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /scratch/xc2627/env.sh;
conda activate /scratch/xc2627/miniconda3/envs/dm2_v1;
CUDA_VISIBLE_DEVICES=0 python vis_gt.py"