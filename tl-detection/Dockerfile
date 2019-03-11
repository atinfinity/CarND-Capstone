ARG cuda_version=10.0
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      libgtk2.0-dev \
      libprotobuf-dev \
      libprotoc-dev \
      protobuf-compiler \
      libhdf5-dev \
      python-dev \
      python-pip \
      python-setuptools \
      python-wheel \
      python-yaml \
      python-lxml \
      python-tk \
      python-matplotlib \
      python-pil \
      python-tqdm \
      wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip install \
    opencv-python \
    tensorflow-gpu==1.13.1 \
    tensorboard

RUN mkdir /workspace
WORKDIR /workspace
RUN cd /workspace && \
    git clone -b v1.13.0 https://github.com/tensorflow/models.git && \
    cd models/research && \
    protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH "/workspace/models/research:/workspace/models/research/slim:/workspace/models/research/object_detection:$PYTHONPATH"
