# Run-able on dsmlp
# 1) choose base container
# generally use the most recent tag

# data science notebook
# https://hub.docker.com/repository/docker/ucsdets/datascience-notebook/tags
ARG BASE_CONTAINER=ucsdets/datascience-notebook:2020.2-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root

# RUN	echo "===> install aria2, nmap, traceroute" &&\
# 	apt-get install -y aria2 &&\
# 	apt-get install -y nmap &&\
# 	apt-get install -y traceroute

RUN echo "===> install make, curl, perl, tmux" && \
    apt-get update && apt-get install -y --force-yes make curl perl tmux

RUN	echo "===> install tqdm" &&\
	pip install install

RUN echo "===> install pytorch" &&\
    conda install pytorch torchvision -c pytorch

RUN echo "===> install cv2" &&\
#    apt update &&\
#    sudo apt install python3-opencv -y --force-yes
    pip install opencv-python
    
# 4) change back to notebook user
COPY /run_jupyter.sh /
RUN chmod 755 /run_jupyter.sh
USER $NB_UID

# Override command to disable running jupyter notebook at launch
CMD ["/bin/bash"]
