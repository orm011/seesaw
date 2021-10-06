FROM continuumio/anaconda3
RUN mkdir workdir
WORKDIR workdir
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update
RUN apt-get install git
RUN conda env create -n seesaw -f ./conda_requirements.yml
RUN conda activate seesaw

RUN mkdir data
RUN mkdir repo