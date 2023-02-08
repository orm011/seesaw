#! /bin/bash

set -x # print to output 
TMPNAME=/state/partition1/user/$USER/raytmp/

## different nodes have differnt amounts available.
## leave some
SHM_AVAILABLE_KB=`df /dev/shm | grep -v Available | awk '{print $4}'`
OBJ_MEM_BYTES=$(( SHM_AVAILABLE_KB*1024 - 1024*1024*1024  )) # leave 1GB off
export RAY_DISABLE_PYARROW_VERSION_CHECK=1


if [ -z "$SLURM_CPUS_ON_NODE" ]; 
then
    echo 'SLURM_CPUS_ON_NODE not set or zero, ensure running within slurm' 
    exit 1
fi

COMMON_ARGS="--temp-dir=$TMPNAME  --object-store-memory=$OBJ_MEM_BYTES --num-cpus=$((2*SLURM_CPUS_ON_NODE))"

SIGFILE=$HOME/ray2.head

if [[ $1 == "--head" ]]
then
    echo 'starting head node'
    ray start $COMMON_ARGS --head
    echo $HOSTNAME > $SIGFILE # signal change after done starting
else 
    echo 'starting worker node'
    PREV=

    if [[ $USER == omoll ]];
    then
        set +x
        source ~/.bashrc
        set -x

        which python
    fi

    while true
    do
        CURRENT=`stat -c '%y' $SIGFILE`
        HEAD_NODE=`cat $SIGFILE`

        if [[ $PREV != $CURRENT ]]
        then
            echo 'file changed... restarting node'
            PREV=$CURRENT
            
            if [[ $USER == omoll && $HEAD_NODE != '' ]]; # im using virtual env rather than default evnv.
            then 
                # copy environment first
                rsync -rlugv --delete --exclude-from /home/gridsan/omoll/rsync_conda_exclude  $HEAD_NODE:/state/partition1/user/omoll/miniconda3/ /state/partition1/user/omoll/miniconda3/
                # rsync -rlugvR --exclude="pkgs/*" --delete $HEAD_NODE:/state/partition1/user/omoll/miniconda3/
                # set +x
                # source /state/partition1/user/omoll/venvs/seesaw/bin/activate
                # set -x
            fi

            ray stop

            if [[$HEAD_NODE == '-1']] # message to exit
            then 
                break
            fi

            if [[ $HEAD_NODE != '' ]]
            then
                ray start $COMMON_ARGS --address=$HEAD_NODE:6379
            fi
        else
            echo 'no change'
        fi
        sleep 5
    done    
fi