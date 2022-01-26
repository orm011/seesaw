## running from repo:
- git checkout this repo and cd into it. 
- from within repo build dockerfile (only needed once)  and note image id :
`docker build --progress=plain -t seesaw_image  .`
DOCKER_BUILDKIT=1 is optional, mostly useful if you will be rebuilding docker file

- download some data to use (sample data coming soon):
  A sample already preprocessed data (a sample of 100 images from coco) is available at 
  https://drive.google.com/file/d/1SSkZeKI3CXIwkIY6g_JXCcLTan4cj5LC/view?usp=sharing

- run container from image and  run run.bash script: 
The main service port is 9000 if want to remap it. Other useful ports are 8265 for the ray dashboard and 8888 for a notebook server.
Scripts assume there is a repo folder and a data folder you should provide.

You can map port 9000 to an available host port eg here we map 9000 in the container (the main application port) to 9001 in the host.


  ```
  docker run \
    --network bridge \
    -p 127.0.0.1:9001:9000 \
    -p 127.0.0.1:8889:8888 \
    -p 127.0.0.1:8266:8265 \
    -v /nvme_drive/orm/vlsworkspace/repo/:/workdir/repo/ \
    -v /nvme_drive/orm/vlsworkspace/data/mini_coco:/workdir/data/mini_coco \
    -v /nvme_drive/orm/nbs/vloop_notebooks/:/workdir/notebooks/ \
    -v /nvme_drive/orm/vlsworkspace/data/:/workdir/notebooks/data  \
    -it seesaw_image:latest  \
    /bin/bash --login -c '. repo/run.bash'
   ```


- the application entry point is and show now work:
`http://localhost:9001/ui/` 
 
- if you map things as above, a running notebook is available at 
`http://localhost:8889`  


# ui
To build/develop this one:
* install node (from website)
* npm install yarn


## Project setup
```
yarn install
```

### Compiles and hot-reloads for development
```
yarn serve
```


### Compiles and minifies for production
```
yarn build
```

```
 yarn build --mode development 
```

### Lints and fixes files
```
yarn lint
```