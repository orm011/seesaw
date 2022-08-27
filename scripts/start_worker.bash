#! /bin/bash

set -x # print to output 
TMPNAME=/state/partition1/user/$USER/raytmp/

## different nodes have differnt amounts available.
## leave some
SHM_AVAILABLE_KB=`df /dev/shm | grep -v Available | awk '{print $4}'`
OBJ_MEM_BYTES=$(( SHM_AVAILABLE_KB*1024 - 1024*1024  ))

COMMON_ARGS="--temp-dir=$TMPNAME  --object-store-memory=$OBJ_MEM_BYTES --num-cpus=$((2*SLURM_CPUS_ON_NODE))"

SIGFILE="$HOME/ray.head"

if [[ $1 == "--head" ]]
then
    ray start $COMMON_ARGS --head
    echo $HOSTNAME > $SIGFILE # signal change after done starting
else 
    PREV=

    while true
    do
        CURRENT=`stat -c '%y' $SIGFILE`
        HEAD_NODE=`cat $SIGFILE`

        if [[ $PREV != $CURRENT ]]
        then
            echo 'file changed... restarting node'
            PREV=$CURRENT
            if [[ $USER == omoll ]]; # im using virtual env rather than default evnv.
            then 
                # copy environment first
                rsync -rlugvR --delete $HEAD_NODE:/state/partition1/user/omoll/venvs/seesaw/ /
        
                set +x
                source /state/partition1/user/omoll/venvs/seesaw/bin/activate
                set -x
            fi
            ## just a test that these things work in the current environment
            # python -c 'import torch; import ray; import transformers; import seesaw'
            ray stop
            ray start $COMMON_ARGS --address=$HEAD_NODE:6379 --redis-password=5241590000000000
        else
            echo 'no change'
        fi
        sleep 5
    done    
fi