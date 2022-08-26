#! /bin/bash

set -x # print to output 
TMPNAME=/state/partition1/user/$USER/raytmp/

## different nodes have differnt amounts available.
## leave some
SHM_AVAILABLE=`df /dev/shm | grep -v Available | awk '{print $4}'`
OBJ_MEM_BYTES=$(( $SHM_AVAILABLE - 10*1024 ))

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
        rsync -rlugvR  $HEAD_NODE:/state/partition1/user/omoll/venvs/seesaw/ /
    
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
