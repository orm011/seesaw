#!/bin/bash
set -euxo pipefail # stop on error etc.

## assumes already in workdir with python env
## assumes repo and data both have been copied/mounted externally (use -v option in docker)

service nginx start # will listen on :9000
### see seesaw.conf for details

(cd notebooks &&  jupyter notebook --allow-root --no-browser --ip=localhost --port 8888 --NotebookApp.token='' --NotebookApp.password='') &
# 2>&1 > ./notebook.logs 

ray start --head
python ./repo/init_model_actor.py & # check ray dashboard for logs
sleep 1
python ./repo/init_data_actors.py &  # check ray dashboard for logs
sleep 1
FLASK_ENV=development FLASK_APP=./repo/session_server.py python -m flask run -p 5000




