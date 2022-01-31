#! /bin/bash
set -x

cd /home/gridsan/$USER/seesaw/scripts/

TMPNAME=/state/partition1/user/$USER/raytmp/

OBJ_MEM_GB=64 # cpu work nodes have around 94, gpu nodes have 79  
OBJ_MEM_BYTES=$(( $OBJ_MEM_GB*(2**30) ))

#export PM2_HOME=/state/partition1/$USER/omoll/pm2_home

if [[ $1 == "--head" ]];
then
    echo $HOSTNAME:6379 > /home/gridsan/$USER/init_spc.head_node
    ray start --head --temp-dir=$TMPNAME --object-store-memory=$OBJ_MEM_BYTES
else
    # assumes this is the only app we run with pm2
    # install pm2 with npm: npm install pm2 -g
    HEAD_ADDRESS=`cat /home/gridsan/$USER/init_spc.head_node`
    RAY_CMD="ray start --block --address=$HEAD_ADDRESS --redis-password=5241590000000000 --temp-dir=$TMPNAME  --object-store-memory=$OBJ_MEM_BYTES"
    pkill -9 PM2
    pkill -9 ray
    pm2 start "$RAY_CMD" --no-daemon
fi
