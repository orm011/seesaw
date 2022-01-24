#! /bin/bash
set -x

cd /home/gridsan/omoll/seesaw/scripts/

TMPNAME=/state/partition1/user/omoll/raytmp/
HEAD_ADDRESS=$1
OTHER_FLAGS=$2 # pass --block if needed

OBJ_MEM_GB=70 # cpu work nodes have around 94, gpu nodes have 79  
OBJ_MEM_BYTES=$(( $OBJ_MEM_GB*(2**30) ))


if [[ $HEAD_ADDRESS == "--head" ]];
then
    ray start --head --temp-dir=$TMPNAME $OTHER_FLAGS
else
    ray start --block --address=$HEAD_ADDRESS --redis-password='5241590000000000' --temp-dir=$TMPNAME
fi
