#!/bin/bash
set -euxo pipefail # stop on error etc.

## assumes already in workdir with python env
## assumes repo and data both have been copied/mounted externally (use -v option in docker)

service nginx start # will listen on :9000
### see seesaw.conf for details

ray start --head 2>&1 > ./ray.log # dashboard on :8265
python ./repo/init_model_actor.py &
sleep 1
python ./repo/init_data_actors.py & #2>&1 > ./data_server.log &
sleep 1
FLASK_ENV=development FLASK_APP=./repo/session_server.py python -m flask run -p 5000



