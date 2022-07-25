FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    wget \
    git \
    ninja-build \
    ffmpeg \
    libsm6 \
    libxext6 \
    vim

RUN rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN pip3 install torch==1.10.0
RUN pip3 install torchvision==0.11.1
RUN pip3 install opencv-python==4.6.0.66
RUN pip3 install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
RUN pip3 install three60cube==0.0.9

RUN mkdir -p /code

WORKDIR /code
RUN git clone https://github.com/facebookresearch/Mask2Former.git

WORKDIR /code/Mask2Former

RUN pip3 install git+https://github.com/cocodataset/panopticapi.git
RUN pip3 install -r requirements.txt

WORKDIR /code/Mask2Former/mask2former/modeling/pixel_decoder/ops

RUN TORCH_CUDA_ARCH_LIST='7.0' FORCE_CUDA=1 python setup.py build install

RUN mkdir -p /code/segmentator
COPY ./code/* /code/segmentator/

ENTRYPOINT ["python", "/code/segmentator/run.py"]
