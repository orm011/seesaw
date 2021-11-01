#! /bin/bash
set -x

. ~/setup_tmp_link.bash

HEAD_ADDRESS=$1
OTHER_FLAGS=$2 # pass --block if needed

if [[ $HEAD_ADDRESS == "--head" ]];
then
    ray start --head --temp-dir=$TMPNAME $OTHER_FLAGS
else
    ray start --block --address=$HEAD_ADDRESS --redis-password='5241590000000000' --temp-dir=$TMPNAME
fi
