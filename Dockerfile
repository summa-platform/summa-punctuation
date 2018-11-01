FROM ubuntu:16.04
MAINTAINER Ondrej Klejch <ondrej.klejch@gmail.com>

RUN apt-get update && apt-get install -y \
    cmake \
    git \
    libeigen3-dev \
    libopenblas-base \
    libopenblas-dev \
    python3-pip \
    gfortran \
    zlib1g-dev \
    g++ \
    automake \
    autoconf \
    libtool \
    libboost-all-dev \
    libgoogle-perftools-dev \
    unzip \
    wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

ARG ARCH=native

# Python 2.7
RUN wget https://bootstrap.pypa.io/ez_setup.py && python2.7 ez_setup.py && easy_install-2.7 pip

# Install amunmt (with Python 3 patch)
RUN git clone https://github.com/marian-nmt/marian -b 1.2.0 --depth 1
RUN mkdir -p marian/build && cd marian/build && cmake -DCMAKE_BUILD_TYPE=release -DCUDA=OFF -DMARIAN=OFF -DPYTHON_VERSION=3.5 .. && make -j 4 && make python -j 4
RUN apt-get update && apt-get install -y vim && apt-get install -y tree

# Install BPE
RUN git clone https://github.com/rsennrich/subword-nmt.git --depth 1

# Install RabbitMQ client
RUN pip3 install --upgrade pip && cp /usr/local/bin/pip3 /usr/bin
RUN pip3 install aio-pika==0.21.0 scipy statsmodels

# NOTE: first download models, then add other files - this optimizes docker build cache usage during development phase
# Download models
ADD download_models.sh /opt/app/
ARG MODEL_URL
RUN bash download_models.sh

ADD . /opt/app

ENV LANG C.UTF-8
ENV PYTHONUNBUFFERED y

ENTRYPOINT ["/opt/app/rabbitmq.py"]
