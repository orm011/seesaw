#! /bin/bash
### scenarios:
## 1. run indexing batch job (want independent from other stuff)
## 2. run benchmark. need ray cluster and cache.
## 3. query service:
        # want to use vite dev server

### tools:
### job/conda env agnostic ray cluster setup script is useful for other tasks and envs.
### being able to add nodes separately is useful.
### being able to have different ray clusters at the same time is useful as well
### want there to be a cluster name that other nodes can use.

## seesaw init script will start a ray head node
## well as a cache

SEESAW_ROOT=$1
SAVE_PATH=$2


set -euxo pipefail
which nginx; which python; which ray
nginx -s stop || echo 'starting nginx....'
nginx -c conf/seesaw.spc.conf # refers to conf/ folder relative to nginx root



DIR=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
#python $DIR/cache_server.py 
python $DIR/session_server.py --seesaw_root=${SEESAW_ROOT} --save_path=${SAVE_PATH}