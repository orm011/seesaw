N=$1
for i in `seq $N`; do
    #LLsub /home/gridsan/omoll/seesaw/scripts/start_worker.bash -g volta:1 -s 20 &
    LLsub /home/gridsan/omoll/seesaw/scripts/start_worker.bash  -s 48  &
done

wait
