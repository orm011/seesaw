#!/bin/bash
set -euxo pipefail # stop on error etc.

## assumes already in workdir with python env
## assumes repo and data both have been copied/mounted externally (use -v option in docker)

## use 0.0.0.0 for all interfaces (needed to access them from host)

service nginx start # will listen on all interfaces :9000
### see seesaw.conf for details
ray start --dashboard-host 0.0.0.0 --dashboard-port 8265 --head
python ./repo/init_model_actor.py & # check ray dashboard for logs
sleep 1
python ./repo/init_data_actors.py &  # check ray dashboard for logs
sleep 1
#FLASK_ENV=development FLASK_APP=./repo/session_server.py python -m flask run -p 5000
FLASK_APP=./repo/session_server.py python -m flask run -p 5000




