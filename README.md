## running from repo:
- git checkout this repo and cd into it. 
- from within repo build dockerfile (only needed once)  and note image id :
`DOCKER_BUILDKIT=1 docker build --progress=plain  --network host .`
DOCKER_BUILDKIT is optional, mostly useful if you will be rebuilding docker file
- download some data to use (sample data coming soon):
- run container from image and  run run.bash script: 
The main service port is 9000 if want to remap it. Other useful ports are 8265 for the ray dashboard and 5000 for the internal api server.
Scripts assume there is a repo folder and a data folder you should provide:

- `docker run --network host -v <absolute repo path>:/workdir/repo/ -v <abs dataset path>:/workdir/data/<dataset name>  -it < image id>  /bin/bash --login -c '. repo/run.bash`
- the following links should work from your browser:
`http://localhost:9000/ui/` 
`http://localhost:9000/data/`


### pre-built image
I have saved a working image in dockerhub: orm011/seesaw, this may be a good starting point if deps have broken.
