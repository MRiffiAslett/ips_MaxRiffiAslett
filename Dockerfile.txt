# Use the base image with CUDA support
FROM nvcr.io/nvidia/cuda:11.6.2-base-ubuntu20.04

# Set the working directory
WORKDIR /app

# Set environment variables to configure tzdata non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

# Install system packages and Python
RUN apt-get update && apt-get install -y \
    openslide-tools \
    python3 \
    python3-pip \
    python3-openslide \
    expect \
    tzdata \
    git \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Check Python and pip installation
RUN python3 --version && pip3 --version

# Upgrade cryptography and pyopenssl to avoid compatibility issues
RUN pip3 install --upgrade cryptography pyopenssl

# Copy the requirements.txt file
COPY requirements.txt /app/requirements.txt

# Install Python packages from requirements.txt
RUN pip3 install -r /app/requirements.txt
RUN pip3 install torch==1.13.0 torchvision==0.14.0 --extra-index-url https://download.pytorch.org/whl/cu117

# Copy the local repository into the Docker image
COPY . /app/ips_attention_masking

# Set the command to run when the container starts
CMD ["bash"]
