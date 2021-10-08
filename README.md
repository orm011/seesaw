- git checkout this repo and cd into it. 
- from within repo build dockerfile (only needed once)  and note image id :
`DOCKER_BUILDKIT=1 docker build --progress=plain  --network host .`
DOCKER_BUILDKIT is optional, mostly useful if you will be rebuilding docker file
- download some data to use (sample data coming soon):
- run container from image and  run run.bash script: (main port is 9000 if want to remap)
- `docker run --network host -v <absolute repo path>:/workdir/repo/ -v <abs dataset path>:/workdir/data/<dataset name>  -it < image id>  /bin/bash --login -c '. repo/run.bash`
- the following links should work from your browser:
`http://localhost:9000/ui/` 
`http://localhost:9000/data/`
