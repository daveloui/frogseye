#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1                # the maximum number of CPUs on V100 (in Beluga)
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=24000M              # the max amount of memory in the node (both gpus and cpus in the node?)
#SBATCH --time=11:59:00
#SBATCH --account=rrg-mbowling-ad_gpu
#SBATCH --output=FrogsEye_output_logs/%N-%j.out
#SBATCH --array=1-30


echo This script is running on
hostname

module load nixpkgs/16.09 python/3.6 gcc/7.3.0 cuda/10.2 cudacore/.10.1.243 cudnn/7.6.5
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_PATH
source ~/jax_env/bin/activate


# Fetch command on line $SLURM_ARRAY_TASK_ID from test file
EXE=`cat FrogEye_parameter_sweep.txt | head -n $SLURM_ARRAY_TASK_ID | tail -n 1`
command="python main_parser.py $EXE"
eval $command