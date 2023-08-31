#! /bin/sh

#SBATCH --job-name="Earth-Mover"
#SBATCH --account=hpcbigdata2
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-core=1
#SBATCH --cores-per-socket=16
#SBATCH --mem-per-cpu=30G
#SBATCH --partition=largemem_q
#SBATCH --time=0-1:00:00
#SBATCH --output=slurm-%A.out

# get the path to this script (method depending on whether using Slurm)
if [ -n "${SLURM_JOB_ID}" ] ; then
    SCRIPTPATH=$(scontrol show job ${SLURM_JOB_ID} | grep -Po "(?<=Command=).*$")
else
    SCRIPTPATH=$(realpath $0)
fi

# get the path to the directory based on where this script is in
export ROOT=$(dirname ${SCRIPTPATH})

# get current time (seconds since epich)
export TIME=$(date +"%s")

# load micromamba and the Python environment
module load mambaforge
mamba activate scoring-test

echo "Current epoch time: ${TIME}"
echo "Case folder: ${ROOT}"
echo "Job script: ${SCRIPTPATH}"
echo "Number of total tasks: ${SLURM_NTASKS}"
echo "Which python: $(which python)"

echo ""
echo "==============================================================="
lscpu
echo "==============================================================="

echo "Start the run"

cd ${ROOT}
mpiexec -n ${SLURM_NTASKS} python ${ROOT}/main.py
