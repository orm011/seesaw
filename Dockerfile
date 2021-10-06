FROM continuumio/anaconda3
RUN mkdir workdir
WORKDIR workdir
SHELL ["/bin/bash", "--login", "-c"]

RUN conda env create -n seesaw -f ./conda_requirements.yml
RUN apt-get update
RUN apt-get install git
RUN conda activate seesaw