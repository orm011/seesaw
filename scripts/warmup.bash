set -x

for mode in default pytorch; do 
    for dataset in bdd coco objectnet; do
        curl -X POST "http://localhost:9000/api/user_session?mode=${mode}&dataset=${dataset}&user=test&qkey=testkey" &
    done
done

wait 