from fastapi import FastAPI
from seesaw.seesaw_web import add_routes, SessionReq, ResetReq
import random, string, os
import ray


TEST_ROOT = '/home/gridsan/omoll/fastai_shared/omoll/seesaw_root/'
tmp_name = ''.join([random.choice(string.ascii_letters) for _ in range(10)])
TEST_SAVE = f'{os.environ["TMPDIR"]}/test_save/{tmp_name}'

ray.init('auto', namespace='seesaw')

app = FastAPI()
WebSeesaw = add_routes(app)
webseesaw = WebSeesaw(TEST_ROOT, TEST_SAVE)

# check basic calls work
state = webseesaw.getstate()
assert len(state.session.gdata) == 0

state = webseesaw.text('bird')
assert len(state.session.gdata) == 1

for i in range(2,4):
    next_req = SessionReq(client_data=state)
    state = webseesaw.next(next_req)
    assert len(state.session.gdata) == i

## check reset call
state = webseesaw.reset(ResetReq(index=state.session.params.index_spec))
assert len(state.session.gdata) == 0