# Use a base image with a recent Ubuntu or Debian version
FROM ubuntu:22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages for GUI and development
# xauth and xorg are crucial for X11 forwarding
# Added libxcb-* packages to resolve Qt XCB plugin loading issues
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libgl1-mesa-glx \
    libxkbcommon-x11-0 \
    xauth \
    xorg \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-sync1 \
    libxcb-util1 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxcb-xfixes0 \
    libxcb-render0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Add conda to PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# Accept Conda Terms of Service for the default channels
RUN conda config --set auto_activate_base false && \
    conda config --set always_yes yes && \
    conda config --add channels https://repo.anaconda.com/pkgs/main && \
    conda config --add channels https://repo.anaconda.com/pkgs/r && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create the conda environment from environment.yml
COPY environment.yml .
RUN conda env create -f environment.yml
ENV CONDA_DEFAULT_ENV=deepfake_env
ENV PATH=$CONDA_DIR/envs/$CONDA_DEFAULT_ENV/bin:$PATH

# Copy your application files into the container
WORKDIR /app
COPY . /app

# Set display environment variable for X11 forwarding
# This will be overridden when running the container
ENV DISPLAY=:0

# Command to run your PyQt application
# Ensure your main.py is executable or use python -u main.py
CMD ["conda", "run", "--no-capture-output", "-n", "deepfake_env", "python", "main.py"]