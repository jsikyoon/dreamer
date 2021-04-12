from tensorflow/tensorflow:2.2.0-gpu

RUN apt-get update && apt-get install -y \
    curl \
    zip \
    unzip \
    git

#######################Jsik option##########################
RUN apt-get install -y \
    vim \
    net-tools \
    openssh-server \
    screen

# Setting
RUN git clone https://github.com/jsikyoon/my_ubuntu_settings --branch indent2 && \
    cd my_ubuntu_settings && \
    ./setup.sh && \
    rm -rf /my_ubuntu_settings
############################################################

RUN apt install -y libgl1-mesa-glx ffmpeg

RUN pip install tensorflow_probability==0.9.0 dm_control pandas matplotlib && \
    pip install gym && \
    pip install gym[atari]
