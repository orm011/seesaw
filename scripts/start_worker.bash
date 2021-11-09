#! /bin/bash
set -x

. ./setup_tmp_link.bash

HEAD_ADDRESS=$1
OTHER_FLAGS=$2 # pass --block if needed

export VECTORDIR=$TMPDIR/vector

## make if not exists
mkdir -p $VECTORDIR

# bc lustre file system is high latency, 
# copy to shm.
for ds in objectnet coco bdd dota; 
do
    echo ds
    stat /home/gridsan/omoll/seesaw_root/data/${ds}/meta/vectors.annoy
    stat $VECTORDIR/${ds}.annoy || (bash ./parallel_copy.bash /home/gridsan/omoll/seesaw_root/data/${ds}/meta/vectors.annoy $VECTORDIR/${ds}.annoy && echo copied $ds)
done

if [[ $HEAD_ADDRESS == "--head" ]];
then
    ray start --head --temp-dir=$TMPNAME $OTHER_FLAGS
else
    ray start --block --address=$HEAD_ADDRESS --redis-password='5241590000000000' --temp-dir=$TMPNAME
fi
