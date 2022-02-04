#! /bin/bash
set -x

cd /home/gridsan/$USER/seesaw/scripts/

TMPNAME=/state/partition1/user/$USER/raytmp/

OBJ_MEM_GB=64 # cpu work nodes have around 94, gpu nodes have 79  
OBJ_MEM_BYTES=$(( $OBJ_MEM_GB*(2**30) ))

COMMON_ARGS="--temp-dir=$TMPNAME  --object-store-memory=$OBJ_MEM_BYTES --num-cpus=$((2*SLURM_CPUS_ON_NODE))"

if [[ $1 == "--head" ]];
then
    echo $HOSTNAME:6379 > /home/gridsan/$USER/init_spc.head_node
    ray start $COMMON_ARGS --head
else 
    HEAD_ADDRESS=`cat /home/gridsan/$USER/init_spc.head_node`
    ray start $COMMON_ARGS --address=$HEAD_ADDRESS --redis-password=5241590000000000 
    sleep infinity # allows me to restart this ray node from elsewhre
fi
