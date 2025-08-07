# v4

# Base image: Ubuntu 20.04 for amd64
FROM --platform=linux/amd64 ubuntu:20.04

# Configure apt for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Add deadsnakes PPA for Python 3.7 and install system dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.7 \
        libpng-dev \
        libjpeg-turbo8 \
        libtiff5 \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3.7 to python
RUN ln -s /usr/bin/python3.7 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy app files
COPY carla-0.9.14-py3.7-linux-x86_64.egg .
COPY run_simulation.py .

# Container entrypoint
CMD ["python", "run_simulation.py"]