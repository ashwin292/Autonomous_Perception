# FINAL DOCKERFILE - v9
# Use Ubuntu 20.04 as the base, specifying the Intel/AMD64 architecture
FROM --platform=linux/amd64 ubuntu:20.04

# Set environment variable to allow apt-get to run without user prompts
ENV DEBIAN_FRONTEND=noninteractive

# --- SINGLE RUN COMMAND TO AVOID CACHING ISSUES ---
# This combines all setup steps into one layer to ensure everything is installed correctly.
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.7 \
        python3.7-distutils \
        python3-pip \
        libpng-dev \
        libjpeg-turbo8 \
        libtiff5 \
    && ln -sf /usr/bin/python3.7 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && pip install pyzmq \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy your files into the container
COPY carla-0.9.14-py3.7-linux-x86_64.egg .
COPY run_simulation.py .
COPY carla_actor_factory.py .

# This is the command that will run when the container starts
CMD ["python", "run_simulation.py"]
