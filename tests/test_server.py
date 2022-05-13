from seesaw.web.common import SessionReq, ResetReq
from seesaw.web.seesaw_app import app
from fastapi.testclient import TestClient
from fastapi import Response
import random, string, os
import ray
import pytest

ray.init("auto", namespace="seesaw")


@pytest.mark.parametrize("mode", ["default", "pytorch"])
def test_server(mode):
    # TEST_ROOT = "/home/gridsan/omoll/fastai_shared/omoll/seesaw_root2/"
    # tmp_name = "".join([random.choice(string.ascii_letters) for _ in range(10)])
    # TEST_SAVE = f'{os.environ["TMPDIR"]}/test_save/{tmp_name}'

    client = TestClient(app)

    # check basic calls work
    resp = client.post(f"/session?mode={mode}")
    assert resp.status_code == 200
    session_id = resp.headers["set-cookie"].split(";")[0].split("=")[1]
    client.cookies.set(f"session_id", session_id)

    resp2 = client.post(
        "/next_task", json={"client_data": {"indices": [], "session": None}}
    )
    assert resp2.status_code == 200

    resp3 = client.post("/session_end")
    assert resp3.status_code == 200
