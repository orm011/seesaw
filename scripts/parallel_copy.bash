#!/bin/bash
infile=$1
outfile=$2
filesize=`stat --format %s $infile`
blocksize=1000000 # 1 MB blocks
numblocks=1000 # x partisze = 1GB

num_tasks=`seq 0 $numblocks $((filesize/blocksize)) | wc -l`
num_workers=$(( num_tasks > 20 ? 20 : num_tasks))

seq 0 $numblocks $((filesize/blocksize)) | xargs -I{} -P $num_workers dd if=$infile bs=$blocksize skip={} conv=sparse seek={} count=$numblocks of=$outfile > /dev/null 2>/dev/null

## adjust last chunk to have same size as input
truncate -s $filesize $outfile