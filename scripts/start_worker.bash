#! /bin/bash

set -x # print to output 
TMPNAME=/state/partition1/user/$USER/raytmp/
OBJ_MEM_GB=64 # cpu work nodes have around 94, gpu nodes have 79  
OBJ_MEM_BYTES=$(( $OBJ_MEM_GB*(2**30) ))

COMMON_ARGS="--temp-dir=$TMPNAME  --object-store-memory=$OBJ_MEM_BYTES --num-cpus=$((2*SLURM_CPUS_ON_NODE))"


if [[ $1 == "--head" ]];
then
    echo $HOSTNAME > $HOME/init_spc.head_node
    ray start $COMMON_ARGS --head
else 
    HEAD_NODE=`cat $HOME/init_spc.head_node`

    if [[ $USER == omoll ]]; # im using virtual env rather than default evnv.
    then 
        # copy environment first
        mkdir -p /state/partition1/user/omoll/venvs
        rsync -rlug $HEAD_NODE:/state/partition1/user/omoll/venvs/ /state/partition1/user/omoll/venvs/
    
        set +x
        source /state/partition1/user/omoll/venvs/seesaw/bin/activate
        set -x
    else
        echo 'using default env'
    fi

    ## just a test that these things work in the current environment
    python -c 'import torch; import ray; import transformers; import seesaw'
    which python
    which ray

    ray start $COMMON_ARGS --address=$HEAD_NODE:6379 --redis-password=5241590000000000 
    sleep infinity # script needs to keep running
fi
