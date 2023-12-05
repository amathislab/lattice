FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    patchelf \
    python3 \
    python3-pip


RUN apt-get install -y --no-install-recommends -o APT::Immediate-Configure=false gcc g++ git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


ENV PYTHONUNBUFFERED 1

ADD requirements.txt /

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install myosuite==1.2.*
RUN pip install --upgrade pickle5 cloudpickle pybullet
RUN python3 -c "import mujoco_py"