from fastapi import FastAPI
from seesaw.seesaw_web import add_routes, SessionReq, ResetReq
import random, string, os
import ray
from tqdm.auto import tqdm
from seesaw.configs import _session_modes, _dataset_map

TEST_ROOT = "/home/gridsan/omoll/fastai_shared/omoll/seesaw_root2/"
tmp_name = "".join([random.choice(string.ascii_letters) for _ in range(10)])
TEST_SAVE = f'{os.environ["TMPDIR"]}/test_save/{tmp_name}'

ray.init("auto", namespace="seesaw", ignore_reinit_error=True)

app = FastAPI()
WebSeesaw = add_routes(app)
webseesaw = WebSeesaw(TEST_ROOT, TEST_SAVE)

search_str = "a dog"
## check they all work (no crashing)
for mode in tqdm(_session_modes):
    for dataset in tqdm(_dataset_map):
        resp = webseesaw.user_session(mode, dataset)
        assert len(resp.session.gdata) == 0

        search_one = webseesaw.text(search_str)
        assert len(search_one.session.gdata) == 1

        next_req = SessionReq(client_data=search_one)
        next_state = webseesaw.next(next_req)
        assert len(next_state.session.gdata) == 2
