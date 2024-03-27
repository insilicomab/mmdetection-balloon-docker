FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV PYTHON_VERSION 3.11.6

# Install necessary packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    ca-certificates \
    git \
    libopencv-dev \
    tzdata \
    && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && rm -rf /var/lib/apt/lists/*

# Install Pyenv
ENV HOME=/root
ENV PYTHON_ROOT=$HOME/local/python-$PYTHON_VERSION
ENV PATH=$PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT=$HOME/.pyenv
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
    git \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
    && $PYENV_ROOT/plugins/python-build/install.sh \
    && /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT \
    && rm -rf $PYENV_ROOT

# Build MMDetection environments
WORKDIR /workspace
RUN pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -U openmim \
    && mim install mmengine \
    && mim install "mmcv>=2.0.0" \
    && rm -rf mmdetection \
    && git clone https://github.com/open-mmlab/mmdetection.git \
    && cd mmdetection \
    && pip install wheel \
    && pip install -v -e .

# Install dependencies
RUN cd ..
COPY env/requirements.txt .
RUN pip install -r requirements.txt

CMD ["tail", "-f", "/dev/null"]