FROM ubuntu:18.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -y update && \
    apt-get install -y libpq-dev build-essential python3-all-dev python3 python3-pip \
    libxml2-dev wget git cmake curl \
    libglib2.0-0 libsm6 libxrender1 libfontconfig1 \
    gnupg xserver-xorg x11-apps xorg-dev
RUN pip3 install -U pip
WORKDIR /app
COPY . /app

RUN pip3 install -r /app/requirements.txt

CMD tail -f /dev/null 
