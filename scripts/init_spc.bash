#! /bin/bash
set -euxo pipefail 

nginx -s stop || echo 'starting nginx....'
nginx -c ./conf/seesaw.spc.conf

DIR=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd $DIR

# echo 'starting head node'
ray stop || echo 'starting ray head node...'
. start_worker.bash --head "--num-cpus=40"

python -m seesaw.memory_cache

## start serve process
python -c 'import ray; from ray import serve; ray.init("auto", namespace="seesaw"); serve.start(detached=True)' &

HEAD="`hostname`:6379"

# NUM_NODES=4
# echo 'starting secondary nodes'
# for i in `seq $NUM_NODES`; do
#     LLsub start_worker.bash  -s 48 -- $HEAD 
# done
# sleep 5 # give time for secondary nodes to start
wait # for model actor and ray serve
sleep infinity