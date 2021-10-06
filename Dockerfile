FROM continuumio/anaconda3
RUN mkdir workdir
WORKDIR workdir
SHELL ["/bin/bash", "--login", "-c"]
#RUN conda init bash
RUN conda create -n seesawenv python=3.8 
RUN conda activate seesawenv
RUN conda install numpy scipy pandas pyarrow pillow scikit-learn scikit-image pytorch torchvision tensorboard jupyter cudatoolkit>=11.1 -c pytorch -c nvidia 
RUN pip install ray flask ipyvue pyroaring plotnine annoy notebook ipykernel ipywidgets

# RUN apt-get update
# RUN apt-get -y install nginx git
# RUN conda activate seesawenv