#!/bin/bash
#SBATCH --job-name=run_all_docker_jobs
#SBATCH --partition=its-2a30-01-part
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50GB
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1 
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/mra23/ips_MaxRiffiAslett
#SBATCH -e /home/mra23/ips_MaxRiffiAslett/output/run_all_docker_jobs_%j.err
#SBATCH -o /home/mra23/ips_MaxRiffiAslett/output/run_all_docker_jobs_%j.out

echo "Starting job: $(date)"

# Start rootless Docker daemon
module load rootless-docker
start_rootless_docker.sh --quiet

# Check if Docker started successfully
if [ $? -ne 0 ]; then
  echo "Failed to start Docker daemon"
  exit 1
fi

# Variables
IMAGE_NAME="my-custom-image"
REPO_DIR="$(pwd)"
RESULTS_DIR="$REPO_DIR/results"
SCRIPT_DIR="/app/ips_MaxRiffiAslett"
MAIN_SCRIPT_PATH="$SCRIPT_DIR/main.py"
DATA_SCRIPT_PATH="$SCRIPT_DIR/data/megapixel_mnist/PineneedleMegaMNIST_200.py"
DATA_DIR="$SCRIPT_DIR/data/megapixel_mnist/dsets/megapixel_mnist_1500"
OUTPUT_FILE="/app/results/results_28_28_3000_3000_150n.txt"

echo "Repository directory: $REPO_DIR"
echo "Results directory: $RESULTS_DIR"

# Ensure the repository and results directories exist
if [ ! -d "$REPO_DIR" ]; then
  echo "Repository directory $REPO_DIR does not exist. Please clone it before running this script."
  exit 1
fi

mkdir -p "$RESULTS_DIR"

# Optionally clear previous data
echo "Clearing previous data"
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

# 1. Build the Docker image with no cache to force rebuild
echo "Building Docker image: $IMAGE_NAME"
docker build --no-cache -t $IMAGE_NAME .
if [ $? -ne 0 ]; then
  echo "Docker image build failed"
  exit 1
fi

# 2. Run the Docker container and mount the repository and results directory
echo "Running Docker container"
docker run --gpus all --shm-size=4g --rm -v "$REPO_DIR:/app/ips_MaxRiffiAslett" -v "$RESULTS_DIR:/app/results" $IMAGE_NAME bash -c "
  cd /app/ips_MaxRiffiAslett
  
  python3 $DATA_SCRIPT_PATH --width 28 --height 3000 --width 3000 -- $DATA_DIR

  # 4. Run the main scripts sequentially and capture the output
  unbuffer python3 $MAIN_SCRIPT_PATH | tee $OUTPUT_FILE
"

if [ $? -ne 0 ]; then
  echo "Docker run failed"
  exit 1
fi

echo "Job completed successfully: $(date)"
