#! /bin/bash

basedir=$1
destdir=$2
P=100

## the dot below is important to preserve relative dir structure
# f,l match only files and soft links (dirs are implicitly created if there are files)
## the print0 xargs -0 is important for handling file names with spaces or quotes etc (like within some datasets)

## --progress --human-readable --verbose
find $basedir/./ -type f,l -print0 | \
    xargs -0 -L 1 -P $P -I{} -t \
    rsync --chmod=ugo=rwX --relative --update --links  "{}" $destdir