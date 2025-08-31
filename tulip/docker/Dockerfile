# Use NVIDIA CUDA base image with Python 3.8
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6"
ENV FORCE_CUDA=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/BEVDepth:/workspace/mmdetection3d
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


# Set working directory
WORKDIR /workspace

# Install Python dependencies individually
RUN pip install einops matplotlib Pillow tensorboardX \
                trimesh opencv-python pandas tqdm scipy wandb \
                scikit-learn tensorboard nuscenes-devkit \
                timm==0.6.12  \ 
                "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# Install chamfer distance package (requires GPU for compilation)
RUN pip install git+https://github.com/otaheri/chamfer_distance

RUN pip install --no-cache-dir openmim==0.3.9 && \
    python -m mim install "mmcv-full==1.7.0" && \
    python -m mim install "mmdet==2.28.2" && \
    python -m mim install "mmsegmentation==0.30.0"
RUN git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git && \
    cd mmdetection3d && pip install -v -e . 

RUN python -m pip install "pip<24.1"
RUN pip install numpy==1.23.5 llvmlite==0.39.1 numba==0.56.4 pytorch-lightning==1.6.5

# Set default command
CMD ["/bin/bash"]
