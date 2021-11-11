#! /bin/bash
set -x

TMP=$TMPDIR/raytmp
mkdir -p $TMP
rm -rf $TMP/*
mkdir -p /tmp/omoll
rm -rf /tmp/omoll/*
TMPNAME=/tmp/omoll/raytmp
ln -sfn $TMP $TMPNAME
file $TMPNAME
