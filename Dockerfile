FROM nvcr.io/nvidia/tensorrt:22.07-py3
ENV DEBIAN_FRONTEND noninteractive

# Build tools
RUN apt update && apt install -y libgl1-mesa-glx
RUN python3 -m pip install opencv-python \
                            line_profiler \
                            cupy-cuda11x \
                            pandas
WORKDIR /workspace
COPY . .