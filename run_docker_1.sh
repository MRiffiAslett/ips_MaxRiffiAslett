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
DATA_SCRIPT_PATH="$SCRIPT_DIR/data/megapixel_mnist/PineneedleMegaMNIST.py"
DATA_DIR="$SCRIPT_DIR/data/megapixel_mnist/dsets/megapixel_mnist_1500"
OUTPUT_FILE="/app/results/results_28_28_3000_3000_100n.txt"
DOCKERFILE_PATH="$REPO_DIR/Dockerfile.txt"

# Ensure the repository and results directories exist
if [ ! -d "$REPO_DIR" ]; then
  echo "Repository directory $REPO_DIR does not exist. Please clone it before running this script."
  exit 1
fi

mkdir -p "$RESULTS_DIR"
mkdir -p "$DATA_DIR"

# Clear previous data
rm -rf "$DATA_DIR/*"

# Build the Docker image
DOCKER_BUILD_LOG="$RESULTS_DIR/docker_build_$(date +%s).log"
docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH . > "$DOCKER_BUILD_LOG" 2>&1

if [ $? -ne 0 ]; then
  echo "Docker image build failed. Check the log for details: $DOCKER_BUILD_LOG"
  exit 1
fi

# Run the Docker container and mount the repository and results directory
docker run --gpus all --shm-size=16g --rm -v "$REPO_DIR:/app/ips_MaxRiffiAslett" -v "$RESULTS_DIR:/app/results" $IMAGE_NAME bash -c "
  cd /app/ips_MaxRiffiAslett
  
  # Ensure data directory exists
  mkdir -p $DATA_DIR
  
  # Generate the dataset and log the output
  echo 'Generating dataset...'
  DATA_GEN_LOG='/app/results/data_generation_$(date +%s).log'
  python3 $DATA_SCRIPT_PATH 28 28 --width 3000 --height 3000 --n_noise 100 $DATA_DIR > \$DATA_GEN_LOG 2>&1
  
  # Check if data generation succeeded
  if grep -q 'Error' \$DATA_GEN_LOG; then
    echo 'Data generation failed. Check the log for details: \$DATA_GEN_LOG'
    exit 1
  fi

  # Check if parameters.json is updated (modification time is recent)
  if [ ! -f '$DATA_DIR/parameters.json' ] || [ \$(( \$(date +%s) - \$(stat -c %Y '$DATA_DIR/parameters.json') )) -gt 60 ]; then
    echo 'parameters.json not found or not recently updated. Data generation failed.'
    exit 1
  fi

  echo 'Data generation successful. Proceeding with training.'

  # Run the main script and capture the output
  unbuffer python3 $MAIN_SCRIPT_PATH | tee $OUTPUT_FILE
"

if [ $? -ne 0 ]; then
  echo "Data generation or training failed. Exiting."
  exit 1
fi
