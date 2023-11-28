#!/bin/bash 
#SBATCH -J BTO
#SBATCH --ntasks-per-node=96 ## Number of tasks per one node   
#SBATCH -A project01273 
#SBATCH --export=ALL
#SBATCH --mem-per-cpu=3800
#SBATCH -C avx512
#SBATCH -t 00:30:00


module purge
###module load gcc intel/mpi
module load intel/2020.4 intelmpi/2020.4

export SMPD_OPTION_NO_DYNAMIC_HOSTS=1
export OMP_NUM_THREADS=1

#python parameter.py


echo $SLURM_JOB_ID > jobid

#####choose the vasp version
#srun -K /work/projects/Projects-da_tmm/Apps/vasp.5.4.4/vasp.5.4.4/bin/vasp_std >& out
srun -K /work/projects/Projects-da_tmm/Apps/vasp.6.1.1/vasp-std  >& out
