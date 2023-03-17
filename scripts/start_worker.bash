#! /bin/bash

set -x # print to output 

echo $-
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
    source ~/.bashrc

    PREV=

    if [[ $USER == omoll ]];
    then
        # sync conda env now, and rerun bashrc to activate the env
        sync_conda_to_local

        if [ -n `which mamba` ]; then
            ## need to rerun bashrc to setup mamba now it exists
            set +x
            source ~/.bashrc
            set -x
        fi

        [ `which mamba` ] && echo 'conda was setup correctly' || echo 'problem setting up conda'
        echo `which python`
        echo `which ray`
        echo `ray --version`
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
                sync_conda_to_local
            fi

            echo `which python`
            echo `which ray`
            echo `ray --version`
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