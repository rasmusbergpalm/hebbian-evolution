#!/bin/bash -ex
REVISION=$(git rev-parse --short HEAD)
MESSAGE=$(git show -s --format=%s)
SCRIPT=$1
DIR=${PWD##*/}
ssh -A -q hpc.itu.dk << EOF
    set -ex
    cd ${DIR}
    git fetch
    git checkout ${REVISION}
    cd ..
    mkdir -p runs/${REVISION}
    cd runs/${REVISION}
    cat <<EOT>run.job
#!/bin/bash
#SBATCH --job-name=${REVISION}
#SBATCH --output=job.%j.out
#SBATCH --cpus-per-task=32
#SBATCH --time=23:00:00
#SBATCH --partition=red
export REVISION=${REVISION}
export MESSAGE="${MESSAGE}"
export SCRIPT=${SCRIPT}
export DIR=${DIR}
export OMP_NUM_THREADS=1
export PYTHONPATH="${PYTHONPATH}:../../${DIR}"
module load singularity/3.4.1
singularity exec /home/enaj/img.sif xvfb-run -a python -u /home/enaj/hebbian-evolution/main.py
EOT
    sbatch run.job
EOF
