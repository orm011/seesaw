#! /bin/bash

set -x
source ~/.bashrc # import sync_conda_to_local
sync_conda_to_local
source ~/.bashrc

mamba activate seesaw

SEESAW_DIR=$(dirname `python -c 'import seesaw; print(seesaw.__path__[0])'`)
DIR="$SEESAW_DIR/scripts/"

which python
which ray

export RAY_DISABLE_PYARROW_VERSION_CHECK=1

SIGFILE="$HOME/ray2.head"
# change current ray head
echo '' > $SIGFILE

bash +x $DIR/start_worker.bash --head 

python -c 'import ray; ray.init("auto", namespace="seesaw"); print(ray.available_resources());'
python $DIR/cache_server.py 

sleep infinity