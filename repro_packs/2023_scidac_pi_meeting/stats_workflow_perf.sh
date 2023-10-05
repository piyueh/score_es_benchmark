#! /bin/sh

#SBATCH --job-name="Stats-Workflow"
#SBATCH --account=hpcbigdata2
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=largemem_q
#SBATCH --time=0-12:00:00
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
srun -N 1 -n 1 -c 1 --mem 10G --exclusive python ${ROOT}/stats_workflow_perf.py small numpy &
srun -N 1 -n 1 -c 1 --mem 10G --exclusive python ${ROOT}/stats_workflow_perf.py small c &
srun -N 1 -n 1 -c 1 --mem 10G --exclusive python ${ROOT}/stats_workflow_perf.py medium numpy &
srun -N 1 -n 1 -c 1 --mem 10G --exclusive python ${ROOT}/stats_workflow_perf.py medium c &
srun -N 1 -n 1 -c 1 --mem 10G --exclusive python ${ROOT}/stats_workflow_perf.py large c &
srun -N 1 -n 1 -c 1 --mem 100G --exclusive python ${ROOT}/stats_workflow_perf.py large numpy &
wait
