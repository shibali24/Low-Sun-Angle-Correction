# Use Ubuntu 18.04 - from the same era as TensorFlow 1.5.0
# Force AMD64/x86_64 platform for TensorFlow 1.x compatibility (doesn't support ARM64)
FROM --platform=linux/amd64 ubuntu:18.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.6 and system dependencies for GDAL, basemap, OpenCV, etc.
RUN apt-get update && apt-get install -y \
    python3.6 \
    python3.6-dev \
    python3-pip \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    libhdf5-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN python3.6 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements files
COPY requirements.txt requirements-modern.txt ./

# Install from cleaned, modernized requirements
# This removes 9 unused packages including basemap and GDAL
RUN pip install -r requirements-modern.txt

# Copy the rest of the project
COPY . .

# Set Python 3.6 as default python
RUN ln -s /usr/bin/python3.6 /usr/bin/python

CMD ["/bin/bash"]
