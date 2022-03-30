set -x

for mode in default pytorch
do 
    for dataset in bdd coco objectnet
    do
        curl -X POST "http://localhost:9000/api/user_session?mode=${mode}&dataset=${dataset}&session_id=warmup_${mode}_${dataset}"
    done
done