#! /bin/bash

basedir=$1
P=100


find $basedir -type d -print0 | xargs -0 -L 1 -P $P -I{} -t bash -c 'chgrp fastai "{}"; chmod 751 "{}"'

if [[ $2 == 'full' ]]
then
    find $basedir -type f -print0 | xargs -0 -L 1 -P $P -I{} -t bash -c 'chgrp fastai "{}"; chmod 640 "{}"'
fi


