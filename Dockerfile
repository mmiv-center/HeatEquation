FROM ubuntu:18.04

RUN apt-get update -qq && apt-get install -yq build-essential \
    cmake git wget libboost-filesystem1.65-dev libboost-timer1.65-dev \
    libboost-system1.65-dev libboost-chrono1.65-dev

# upgrade the gcc
RUN apt-get install -yq software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update -qq && apt install -yq gcc-8 g++-8

# install itk
RUN cd /tmp/ \
    && wget https://github.com/InsightSoftwareConsortium/ITK/releases/download/v5.0.1/InsightToolkit-5.0.1.tar.gz \
    && cd /opt/ && tar xzvf /tmp/InsightToolkit-5.0.1.tar.gz && cd /opt/InsightToolkit-5.0.1 \
    && mkdir bin && cd bin && \
    export CC=/usr/bin/gcc-8 && \
    export CXX=/usr/bin/g++-8 && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && make

RUN mkdir /HeatEquation && cd /HeatEquation/ \
    && git clone https://github.com/mmiv-center/HeatEquation.git . \
    && export CC=/usr/bin/gcc-8 \
    && export CXX=/usr/bin/g++-8 \
    && cmake -DCMAKE_BUILD_TYPE=Release . && make

ENTRYPOINT [ "/HeatEquation/HeatEquation" ]
