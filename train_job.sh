#!/bin/bash    
#PBS -N bin_exp
#PBS -l select=1:ncpus=2:mem=16gb:scratch_local=50gb:ngpus=1:cl_adan=True    
#PBS -q gpu    
#PBS -l walltime=10:00:00    
#PBS -m ae


# define BASE TODO: alter..
export BASE=/storage/brno2/home/xsarva00/

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually     
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt    
     
# test if scratch directory is set    
# if scratch directory is not set, issue error message and exit    
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }    


# load modules    
module add anaconda3-4.0.0    
module add conda-modules-py37    
    
# activate (or create) the conda environment    
conda activate torch_env || {    
  conda create -n torch_env python=3.7;    
  conda activate torch_env;    
  conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch;
  conda install scipy;    
  conda install -c conda-forge matplotlib;
}

    
# move into scratch directory
cd $SCRATCHDIR

# copy the repository..
cp -r $BASE/vut-fit-bin .
cd vut-fit-bin
mkdir data     

# run the training..
bash run_meta.sh

# clean the SCRATCH directory and exit :)
clean_scratch
