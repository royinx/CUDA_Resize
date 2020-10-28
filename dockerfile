FROM nvcr.io/nvidia/tensorrt:19.12-py3

# FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

# Build tools
RUN apt update && \
    apt install -y \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1-mesa-glx

# Python:
RUN python3 -m pip install opencv-python \
                            line_profiler

