#! /bin/bash
set -euxo pipefail 

(wget --quiet localhost:9000 && echo 'nginx already running') || (echo 'starting nginx...' && nginx -c ./conf/seesaw.spc.conf)

DIR=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd $DIR

echo 'starting head node'
# export TMPDIR="/state/partition1/slurm_tmp/`LLstat | grep Jupyter | awk '{ print $1 }'`.0.0"
. start_worker.bash --head "--num-cpus=40 --num-gpus=2" 

python init_model_actor.py --namespace seesaw &

## start serve process
python -c 'import ray; from ray import serve; ray.init("auto", namespace="seesaw"); serve.start(detached=True)' &


HEAD="`hostname`:6379"
NUM_NODES=4

echo 'starting secondary nodes'
for i in `seq $NUM_NODES`; do
    LLsub start_worker.bash  -s 48 -- $HEAD 
done

sleep 5 # give time for secondary nodes to start
wait # for model actor
python init_data_actors.py --load_ground_truth --load_coarse_embedding --namespace seesaw --datasets objectnet bdd coco dota lvis

echo 'DONE. inited all data actors'
sleep infinity