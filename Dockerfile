FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]
# Add UTF-8 support
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales && \
    locale-gen en_US.UTF-8
ENV PYTHONIOENCODING='UTF-8' LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8' PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        bash \
        build-essential \
        cmake \
        git \
        ssh \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopenblas-dev \
        libprotobuf-dev \
        libsnappy-dev \
        make \
        protobuf-compiler \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-flask \
        python3-tk \
        python3-opencv \
        wget \
        tmux \
        htop \
        vim  \
        unzip


RUN mkdir workspace
WORKDIR /workspace

RUN python3 -m pip install --upgrade pip

# COPY requirements.txt requirements.txt
RUN pip3 install --verbose ultralytics
RUN pip3 install --verbose comet_ml
RUN pip3 install --verbose opencv-python
RUN pip3 install --verbose PyYAML
RUN pip3 install --verbose pandas
RUN pip3 install --verbose numpy
RUN pip3 install --verbose tqdm
RUN pip3 install --verbose matplotlib
RUN pip3 install --verbose super-gradients
RUN pip3 install --verbose torchvision
RUN pip3 install --verbose kaggle

RUN pip3 install --upgrade setuptools
RUN pip3 install lxml==4.8.0
RUN apt-get update && apt-get install -y --no-install-recommends python-lxml
RUN pip3 install --verbose kaggle-cli==0.11.5

RUN mkdir /root/.kaggle
COPY kaggle.json /root/.kaggle/kaggle.json

RUN pip3 install --verbose pydicom

# RUN rm -rf /tmp/autogen && mkdir -p /tmp/autogen && unzip -qo /tmp/autogen.zip -d /tmp/autogen && \
#     mv /tmp/autogen/autogen-*/* /tmp/autogen && rm -rf /tmp/autogen/autogen-* && \
#     sudo chmod a+rx /tmp/autogen/autogen.sh

# RUN pip3 install hatch
# RUN hatch run install-deps
# RUN hatch run compile

# RUN pip3 install --verbose -f https://download.pytorch.org/whl/cu113/torch_stable.html torch==1.10.2+cu113
# RUN pip3 install --verbose --user --no-warn-script-location -r requirements.txt

# RUN pip3 install --user --no-warn-script-location -r requirements.txt
# RUN python3 -m pip install --user git+https://github.com/lessw2020/Ranger21.git/
# RUN pip3 install ffmpeg
# RUN pip3 install onnxruntime
# RUN pip3 install onnxruntime-gpu
# RUN pip3 install torchmetrics
