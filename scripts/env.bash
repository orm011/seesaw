. ~/supercloud_util/sync_tools.bash

function init_env(){
    export PYTHONNOUSERSITE=1 # dont add .local to python path, use conda pip
    setup_worker_mamba seesaw

    export RAY_DISABLE_PYARROW_VERSION_CHECK=1
    export MODIN_ENGINE=ray
    export __MODIN_AUTOIMPORT_PANDAS__=1
    mkdir -p $TMPDIR
    export TMPDIR=/state/partition1/user/$USER/tmpdir/
}

function after_start_head(){
    python -c 'import seesaw' # should succeed now
    python ~/seesaw/scripts/cache_server.py 
}