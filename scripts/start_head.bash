#! /bin/bash

set -x

ENVSETUP=~/seesaw/scripts/env.bash
NAME=seesaw

bash +x ~/supercloud_util/start_worker.bash $ENVSETUP $NAME --head 
# should stay in loop.
