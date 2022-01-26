#! /bin/bash
set -euxo pipefail 

nginx -s stop || echo 'starting nginx....'
nginx -c ./conf/seesaw.spc.conf

DIR=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd $DIR

# echo 'starting head node'
ray stop || echo 'starting ray head node...'
. start_worker.bash --head

python -m seesaw.memory_cache

## start serve process
python -c 'import ray; from ray import serve; ray.init("auto", namespace="seesaw"); serve.start(detached=True)' &

wait # for model actor and ray serve
sleep infinity