FROM nvcr.io/nvidia/tensorrt:22.07-py3

# FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

# Build tools
RUN apt update && apt install -y libgl1-mesa-glx

# Python:
RUN python3 -m pip install opencv-python \
                            line_profiler \ 
                            cupy-cuda11x