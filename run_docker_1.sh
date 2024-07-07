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

# Start rootless Docker daemon
module load rootless-docker
start_rootless_docker.sh --quiet

# Variables
IMAGE_NAME="my-custom-image"
REPO_DIR="$(pwd)"
RESULTS_DIR="$REPO_DIR/results"
SCRIPT_DIR="/app/ips_MaxRiffiAslett"
MAIN_SCRIPT_PATH="$SCRIPT_DIR/main.py"
DATA_SCRIPT_PATH="$SCRIPT_DIR/data/megapixel_mnist/PineneedleMegaMNIST_150.py"
DATA_DIR="$SCRIPT_DIR/data/megapixel_mnist/dsets/megapixel_mnist_1500"
OUTPUT_FILE="/app/results/results_28_28_3000_3000_150n.txt"
DOCKERFILE_PATH="$REPO_DIR/Dockerfile.txt"

# Ensure the repository and results directories exist
if [ ! -d "$REPO_DIR" ]; then
  echo "Repository directory $REPO_DIR does not exist. Please clone it before running this script."
  exit 1
fi

mkdir -p "$RESULTS_DIR"

# Clear previous data
rm -rf "$DATA_DIR/*"

# Build the Docker image
docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH . > "$RESULTS_DIR/docker_build_$(date +%s).log" 2>&1

if [ $? -ne 0 ]; then
  echo "Docker image build failed. Check the log for details: $RESULTS_DIR/docker_build_$(date +%s).log"
  exit 1
fi

# Run the Docker container and mount the repository and results directory
docker run --gpus all --shm-size=4g --rm -v "$REPO_DIR:/app/ips_MaxRiffiAslett" -v "$RESULTS_DIR:/app/results" $IMAGE_NAME bash -c "
  cd /app/ips_MaxRiffiAslett
  
  # Ensure data directory exists
  mkdir -p $DATA_DIR
  
  # Generate the dataset and log the output
  echo 'Generating dataset...'
  python3 $DATA_SCRIPT_PATH 28 28 --width 3000 --height 3000 $DATA_DIR > /app/results/data_generation.log 2>&1
  
  # Check if parameters.json is created
  if [ ! -f '$DATA_DIR/parameters.json' ]; then
    echo 'parameters.json not found. Data generation failed.' >> /app/results/data_generation.log
    exit 1
  fi

  # Run the main script and capture the output
  unbuffer python3 $MAIN_SCRIPT_PATH | tee $OUTPUT_FILE
"
