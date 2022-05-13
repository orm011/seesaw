#! /bin/bash

## needed for nginx/ray/python
# . $HOME/miniconda3/etc/profile.d/conda.sh
# conda activate $HOME/seesaw_deploy/.seesaw_env
set -euxo pipefail
which nginx; which python; which ray

DIR=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
nginx -s stop || echo 'starting nginx....'
nginx -c conf/seesaw.spc.conf # refers to conf/ folder relative to nginx root

# echo 'starting head node'
ray stop || echo 'starting ray head node...'
bash +x $DIR/start_worker.bash --head 

# start frame server
uvicorn frameserver.server:app --host localhost.localdomain --port 8600  --workers=5  >> frameserver.log  2>&1 &

python $DIR/cache_server.py 

# wait # for cache and serve

if [ $# -gt 0 ]; then
    ## used by orm for user study.
    # if some arguments are passed in then use them to start a server and then warm up the server
    SERVER_FLAGS=$@
    python ./seesaw_deploy/scripts/session_server.py --no_block $SERVER_FLAGS

    ## for user study also warm cache with the datasets so we don't wait
    bash $DIR/warmup.bash

    ## NB assumes we are building in the same machine. disable if not
    #(cd $DIR/../frontend && yarn run vite build) & # ensure we use latest frontend code
    # (cd $DIR/../frontend && yarn run vite  --port 8500) # start dev server
fi

sleep infinity

