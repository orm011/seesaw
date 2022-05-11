import ray
from seesaw.dataset_manager import (
    GlobalDataManager,
)
from seesaw.basic_types import (
    SessionParams,
    BenchParams,
    IndexSpec,
)

from seesaw.seesaw_web import (
    SessionReq,
    ResetReq,
    SessionInfoReq,
    SessionManagerActor,
    app,
    WebSeesaw,
)

from seesaw.seesaw_bench import (
    BenchRunner,
)


import random, string, os
from fastapi import FastAPI
from seesaw.configs import std_linear_config

import math
import pytest

import requests


def test_session():
    ray.init("auto", namespace="seesaw", ignore_reinit_error=True)

    # TEST_ROOT = '/home/gridsan/omoll/fastai_shared/omoll/seesaw_root/'
    TEST_ROOT = "/home/gridsan/omoll/seesaw_root/"
    tmp_name = "".join([random.choice(string.ascii_letters) for _ in range(10)])
    TEST_SAVE = f'{os.environ["TMPDIR"]}/test_save/{tmp_name}'

    os.makedirs(TEST_SAVE, exist_ok=True)

    gdm = GlobalDataManager(TEST_ROOT)
    os.chdir(gdm.root)
    br = BenchRunner(gdm.root, results_dir=TEST_SAVE)

    b = BenchParams(
        name="seesaw_test",
        ground_truth_category="aerosol can",
        qstr="aerosol can",
        n_batches=4,
        max_feedback=None,
        box_drop_prob=0.0,
        max_results=100000,
    )

    p = SessionParams(
        index_spec=IndexSpec(
            d_name="data/lvis/", i_name="multiscale", c_name="aerosol can"
        ),
        interactive="pytorch",
        batch_size=3,
        method_config=std_linear_config,
    )

    bench_path = br.run_loop(b, p)

    session_manager = SessionManagerActor.options(name="test-session-manager").remote(
        root_dir=TEST_ROOT, save_path=TEST_SAVE, num_cpus_per_session=2
    )
    WebSeesaw.deploy(session_manager)

    webseesaw = WebSeesaw.get_handle()
    print(webseesaw)
    print(type(webseesaw))

    bench_state = webseesaw.session_info(SessionInfoReq(path=bench_path))
    assert len(bench_state.session.gdata) == b.n_batches

    ## use the provided config to run the same session
    state = await webseesaw.reset(ResetReq(config=bench_state.default_params))
    assert len(state.session.gdata) == 0

    state = webseesaw.text("bird")
    assert len(state.session.gdata) == 1

    for i in range(2, b.n_batches + 1):
        next_req = SessionReq(client_data=state)
        state = webseesaw.next(next_req)
        assert len(state.session.gdata) == i, f"{i}"

    assert len(state.session.gdata) == b.n_batches

    saved_state = state
    r = await webseesaw.save(SessionReq(client_data=saved_state))
    assert os.path.exists(r.path)

    restored_state = webseesaw.session_info(SessionInfoReq(path=r.path))
    assert len(restored_state.session.gdata) == len(saved_state.session.gdata)

    print("test success")
