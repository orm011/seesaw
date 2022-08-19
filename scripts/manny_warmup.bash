set -x


for dataset in bdd coco objectnet; do
    for index in multiscale multibeit roibased detr beit coarse; do 
        curl -X POST "http://localhost:9000/api/user_session?mode=pytorch&dataset=${dataset}&index=${index}" &
    done
done


wait 