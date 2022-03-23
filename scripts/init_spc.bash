#! /bin/bash

# . $HOME/miniconda3/etc/profile.d/conda.sh
# conda activate $HOME/seesaw_deploy/.seesaw_env
set -euxo pipefail
which nginx
which python
which ray

## needed for nginx/ray/python
DIR=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
nginx -s stop || echo 'starting nginx....'
nginx -c ./conf/seesaw.spc.conf


# echo 'starting head node'
ray stop || echo 'starting ray head node...'
bash +x $DIR/start_worker.bash --head

python -m seesaw.memory_cache

## start serve process
python -c 'import ray; from ray import serve; ray.init("auto", namespace="seesaw"); serve.start(detached=True)' &

wait # for model actor and ray serve
sleep infinity