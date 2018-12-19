# Will be named as deepgaze/eap-models-dev
FROM pkdogcom/pytorch:py3
LABEL maintainer zijufeng@deepgaze.ai

# Install protobuf 3 & some other dependencies
RUN apt-get install -y software-properties-common python3-software-properties && \
    add-apt-repository -y ppa:maarten-fonville/protobuf && apt-get update && \
    apt-get install -y protobuf-compiler python3-setuptools && pip3 install Cython

# Install the tensorflow-gpu and other mode
RUN apt-get update && apt-get install -y --no-install-recommends libcupti-dev python3-pil python3-lxml && \
    pip3 install -U jupyter matplotlib tensorflow-gpu==1.9.0 html5lib bleach numpy

# Environment for Tensorflow Models
ENV MODELS_HOME=/root/tensorflow/models
ENV PYTHONPATH=$PYTHONPATH:$MODELS_HOME:$MODELS_HOME/research/slim:$MODELS_HOME/research

# Download and setup Tensorflow Models
RUN git clone https://github.com/itmessager/models.git $MODELS_HOME && \
    cd $MODELS_HOME/research && protoc object_detection/protos/*.proto --python_out=. 

# Install COCO API
RUN cd /tmp && git clone https://github.com/pkdogcom/coco.git && \
    cd coco/PythonAPI && make -j$(nproc) && \ 
cp -r pycocotools $MODELS_HOME/research/

# Install python libraries
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt -U --upgrade-strategy=only-if-needed &&\
    rm /tmp/requirements.txt

# Install visualization tools and the libraries for pycharm
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y gedit libgtk2.0-0 libcanberra-gtk-module libxext-dev libxrender-dev libxtst-dev mercurial && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* && \
    # Use Pillow-SIMD for faster preprocessing using PIL API
    pip3 uninstall -y pillow pillow-simd && pip3 install pillow-simd

# Install font for demo
RUN apt-get update && apt-get install -y ttf-wqy-zenhei

WORKDIR /root

