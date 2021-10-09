FROM continuumio/anaconda3
RUN mkdir workdir
WORKDIR workdir
SHELL ["/bin/bash", "--login", "-c"]

# system deps
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y gettext-base 
RUN apt-get install -y build-essential
RUN apt-get install -y nginx
RUN apt-get install -y lsb-release 
RUN apt-get install -y iputils-ping net-tools netcat less

# conda deps
# run with `export DOCKER_BUILDKIT=1 docker build .` to cache pkgs and avoid super time-consuming rebuilds
## only copy reqs file at this stage
## to avoid rebuilding this every time we update the repo. 
## only rerun when we add more conda/pip deps
COPY conda_requirements.yml .
RUN --mount=type=cache,target=/opt/conda/pkgs conda env create -n seesaw -f conda_requirements.yml
RUN echo 'conda activate seesaw' >> ~/.bashrc

# pre-download clip model
# RUN --mount=type=cache,target=/root/.cache/clip  # this version did not work, stuff was gone at runtime 
RUN python -c "import clip; _ = clip.load('ViT-B/32', device='cpu', jit=False)" && ls /root/.cache/clip

# install ipython kernel
RUN python -m ipykernel install --user --name seesaw --display-name "Python (seesaw)"

# install nginx server config for app
COPY ./seesaw.conf /etc/nginx/sites-enabled/
RUN rm /etc/nginx/sites-enabled/default
EXPOSE 9000

RUN nginx -T
RUN service nginx reload
# server logs available at /var/log/nginx/error.log 

## data folder
RUN mkdir data
VOLUME data

## notebook folder
RUN mkdir notebooks
VOLUME notebooks

# expose also API server, ray dashboard and jupyter notebook
EXPOSE 5000 8265 8888

## do this as late as possible within this file, everything after will be redone every time
COPY . repo
RUN rm -rf repo/.git
VOLUME repo

## install seesaw and check import works
RUN conda activate seesaw && pip install -e repo && python -c 'import seesaw'