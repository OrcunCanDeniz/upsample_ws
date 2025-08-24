# Use NVIDIA CUDA base image with Python 3.8
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    git \
    wget \
    curl python3-tk tk-dev \ 
    build-essential \
    git build-essential libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python (python3 already exists)
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch 1.12.0 with CUDA 11.3
RUN pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install chamfer distance package (requires GPU for compilation)
RUN pip install git+https://github.com/otaheri/chamfer_distance

# Set working directory
WORKDIR /workspace

# Install Python dependencies individually
RUN pip install einops matplotlib numpy Pillow tensorboardX \
                trimesh opencv-python pandas tqdm scipy wandb \
                scikit-learn timm tensorboard nuscenes-devkit

# Set default command
CMD ["/bin/bash"]
