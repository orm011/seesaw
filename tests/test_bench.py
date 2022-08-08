from seesaw.seesaw_bench import (
    get_all_session_summaries,
    BenchRunner,
    BenchSummary,
    get_metric_summary,
)

import ray
from seesaw.basic_types import (
    SessionParams,
    BenchParams,
    IndexSpec,
)
from seesaw.dataset_manager import GlobalDataManager
import random, string, os
from seesaw.configs import std_linear_config, std_textual_config


TEST_ROOT = "/home/gridsan/groups/fastai/omoll/seesaw_root2/"
tmp_name = "".join([random.choice(string.ascii_letters) for _ in range(10)])
TEST_SAVE = f"~/tmp/lvis_tests/test_save_{tmp_name}/"
TEST_SAVE = os.path.expanduser(TEST_SAVE)


cat = "bike"
qstr = "a bike"

cat_objectnet = "air freshener"
qstr_objectnet = "an air freshener"
## chosen so there are some positives withi this range

configs = [
    # (BenchParams(name='seesaw_test', ground_truth_category=cat, qstr=qstr,
    #   n_batches=4, max_feedback=None, box_drop_prob=0.0, max_results=10000),
    # SessionParams(index_spec=IndexSpec(d_name='data/lvis/', i_name='multiscale', c_name=cat),
    #   interactive='pytorch', batch_size=3, agg_method='avg_score', method_config=std_linear_config)
    # ),
    # (BenchParams(name='baseline', ground_truth_category=cat, qstr=qstr,
    #   n_batches=4, max_results=10, max_feedback=None, box_drop_prob=0.0),
    # SessionParams(index_spec=IndexSpec(d_name='data/lvis/', i_name='coarse', c_name=cat),
    #   interactive='plain', batch_size=3, agg_method='avg_score', method_config=std_linear_config)
    # ),
    # (BenchParams(name='seesaw_test', ground_truth_category=cat, qstr=qstr,
    #   n_batches=4, max_feedback=None, box_drop_prob=0.0, max_results=10000),
    # SessionParams(index_spec=IndexSpec(d_name='data/lvis/', i_name='multiscale', c_name=cat),
    #   interactive='pytorch', batch_size=3, agg_method='avg_score', method_config=std_linear_config)
    # ),
    (
        BenchParams(
            name="seesaw_test_textual",
            ground_truth_category=cat,
            qstr=qstr,
            provide_textual_feedback=True,
            n_batches=4,
            max_feedback=None,
            box_drop_prob=0.0,
            max_results=10000,
        ),
        SessionParams(
            index_spec=IndexSpec(d_name="lvis", i_name="multiscale", c_name=cat),
            interactive="textual",
            agg_method="avg_score",
            method_config={**std_textual_config, "mode": "linear"},
            batch_size=3,
        ),
    ),
    # (BenchParams(name='seesaw_test_textual', ground_truth_category=cat, qstr=qstr,
    #               provide_textual_feedback=True,
    #   n_batches=4, max_feedback=None, box_drop_prob=0.0, max_results=10000),
    # SessionParams(index_spec=IndexSpec(d_name='data/lvis/', i_name='multiscale', c_name=cat),
    #   interactive='textual', agg_method='avg_vector', method_config={**std_textual_config, 'mode':'finetune'}, batch_size=3)
    # ),
    # (BenchParams(name='seesaw_test_textual', ground_truth_category=cat_objectnet, qstr=qstr_objectnet,
    #   provide_textual_feedback=True,
    #   n_batches=4, max_feedback=None, box_drop_prob=0.0, max_results=10000),
    # SessionParams(index_spec=IndexSpec(d_name='data/objectnet/', i_name='multiscale', c_name=cat_objectnet),
    #   interactive='textual', agg_method='avg_score', method_config={**std_textual_config, 'mode':'linear'},  batch_size=3),
    # ),
    (
        BenchParams(
            name="seesaw_test_textual",
            ground_truth_category=cat_objectnet,
            qstr=qstr_objectnet,
            provide_textual_feedback=True,
            n_batches=4,
            max_feedback=None,
            box_drop_prob=0.0,
            max_results=10000,
        ),
        SessionParams(
            index_spec=IndexSpec(
                d_name="objectnet", i_name="multiscale", c_name=cat_objectnet
            ),
            interactive="textual",
            agg_method="avg_vector",
            method_config={**std_textual_config, "mode": "finetune"},
            batch_size=3,
        ),
    ),
    (
        BenchParams(
            name="seesaw_test_binary",
            ground_truth_category=cat_objectnet,
            qstr=qstr_objectnet,
            provide_textual_feedback=False,
            n_batches=4,
            max_feedback=None,
            box_drop_prob=0.0,
            max_results=10000,
        ),
        SessionParams(
            index_spec=IndexSpec(
                d_name="objectnet", i_name="multiscale", c_name=cat_objectnet
            ),
            interactive="pytorch",
            agg_method="avg_vector",
            method_config=std_linear_config,
            batch_size=3,
        ),
    ),
]

test_config = [(
        BenchParams(
            name="beit",
            ground_truth_category="bike",
            qstr="a bike",
            provide_textual_feedback=True,
            n_batches=4,
            max_feedback=None,
            box_drop_prob=0.0,
            max_results=10000,
        ),
        SessionParams(
            index_spec=IndexSpec(
                d_name="bdd", i_name="beit", c_name="bike"
            ),
            interactive="pytorch",
            agg_method="avg_vector",
            method_config=std_linear_config,
            batch_size=3,
        ),
    ),]
def test_detr_threshold(): 
    config = []
    old_bdd_cat_list = [("bike", "a bike"), ("gas stations scene", "a gas station"), ("person", "a person"), ("bus", "a bus"), ("train", "a train"), ("tunnel scene", "a tunnel"), ("truck", "a truck"), ("green traffic light", "a green traffic light"), ("parking lot scene", "a parking lot")]
    new_bdd_cat_list = [("bicycle", "a bike"), ("pedestrian", "a pedestrian"), ("bus", "a bus"), ("train", "a train"), ("traffic light", "a traffic light"), ("traffic sign", "a traffic sign"), ("trailer", "a trailer"), ("rider", "a rider")]
    coco_cat_list = [("bicycle", "a bicycle"), ("boat", "a boat"), ("wine glass", "a glass of wine"), ("kite", "a kite"), ("laptop", "a laptop"), ("zebra", "a zebra"), ("sink", "a sink"),("toaster", "a toaster"),]
    dataset = "bdd"
    batches = 33
    agg_method = "avg_score"
    feedback = False
    max_feedback = None
    for pair in new_bdd_cat_list: 
        catt = pair[0]
        qstrr = pair[1]
        config.extend([(BenchParams(
            name="detr",
            ground_truth_category=catt,
            qstr=qstrr,
            provide_textual_feedback=feedback,
            n_batches=batches,
            max_feedback=max_feedback,
            box_drop_prob=0.0,
            max_results=10000,
        ),
        SessionParams(
            index_spec=IndexSpec(
                d_name=dataset, i_name="detr", c_name=catt
            ),
            interactive="pytorch",
            agg_method=agg_method,
            method_config=std_linear_config,
            batch_size=3,
        ),), (BenchParams(
            name="detr70",
            ground_truth_category=catt,
            qstr=qstrr,
            provide_textual_feedback=feedback,
            n_batches=batches,
            max_feedback=max_feedback,
            box_drop_prob=0.0,
            max_results=10000,
        ),
        SessionParams(
            index_spec=IndexSpec(
                d_name=dataset, i_name="detr70", c_name=catt
            ),
            interactive="pytorch",
            agg_method=agg_method,
            method_config=std_linear_config,
            batch_size=3,
        ),), (BenchParams(
            name="detr80",
            ground_truth_category=catt,
            qstr=qstrr,
            provide_textual_feedback=feedback,
            n_batches=batches,
            max_feedback=max_feedback,
            box_drop_prob=0.0,
            max_results=10000,
        ),
        SessionParams(
            index_spec=IndexSpec(
                d_name=dataset, i_name="detr80", c_name=catt
            ),
            interactive="pytorch",
            agg_method=agg_method,
            method_config=std_linear_config,
            batch_size=3,
        ),), (BenchParams(
            name="detr90",
            ground_truth_category=catt,
            qstr=qstrr,
            provide_textual_feedback=feedback,
            n_batches=batches,
            max_feedback=max_feedback,
            box_drop_prob=0.0,
            max_results=10000,
        ),
        SessionParams(
            index_spec=IndexSpec(
                d_name=dataset, i_name="detr90", c_name=catt
            ),
            interactive="pytorch",
            agg_method=agg_method,
            method_config=std_linear_config,
            batch_size=3,
        ),), (BenchParams(
            name="multiscale",
            ground_truth_category=catt,
            qstr=qstrr,
            provide_textual_feedback=feedback,
            n_batches=batches,
            max_feedback=max_feedback,
            box_drop_prob=0.0,
            max_results=10000,
        ),
        SessionParams(
            index_spec=IndexSpec(
                d_name=dataset, i_name="multiscale", c_name=catt
            ),
            interactive="pytorch",
            agg_method=agg_method,
            method_config=std_linear_config,
            batch_size=3,
        ),), (
        BenchParams(
            name="roibased",
            ground_truth_category=catt,
            qstr=qstrr,
            provide_textual_feedback=feedback,
            n_batches=batches,
            max_feedback=max_feedback,
            box_drop_prob=0.0,
            max_results=10000,
        ),
        SessionParams(
            index_spec=IndexSpec(
                d_name=dataset, i_name="roibased", c_name=catt
            ),
            interactive="pytorch",
            agg_method=agg_method,
            method_config=std_linear_config,
            batch_size=3,
        ),
    )
        
        
        ])

    return config

def make_config(): 
    config = []
    old_bdd_cat_list = [("bike", "a bike"), ("gas stations scene", "a gas station"), ("person", "a person"), ("bus", "a bus"), ("train", "a train"), ("tunnel scene", "a tunnel"), ("truck", "a truck"), ("green traffic light", "a green traffic light"), ("parking lot scene", "a parking lot")]
    new_bdd_cat_list = [("bicycle", "a bike"), ("pedestrian", "a pedestrian"), ("bus", "a bus"), ("train", "a train"), ("traffic light", "a traffic light"), ("traffic sign", "a traffic sign"), ("trailer", "a trailer"), ("rider", "a rider")]
    coco_cat_list = [("bicycle", "a bicycle"), ("boat", "a boat"), ("wine glass", "a glass of wine"), ("kite", "a kite"), ("laptop", "a laptop"), ("zebra", "a zebra"), ("sink", "a sink"),("toaster", "a toaster"),]
    lvis_cat_list = [("apple", "an apple"), ("suitcase", "a suitcase"), ("award", "an award"), ("baseball bat", "a baseball bat"), ("bat (animal)", "a bat animal"), ("blueberry", "a blueberry"), ("bow-tie", "a bow tie"), ("elevator car", "an elevator"), ("card", "a card"), ("computer keyboard", "a keyboard"),]
    dataset = "lvis"
    batches = 33
    agg_method = "avg_score"
    feedback = False
    max_feedback = None
    for pair in lvis_cat_list: 
        catt = pair[0]
        qstrr = pair[1]
        config.extend([(
        BenchParams(
            name="multiscale",
            ground_truth_category=catt,
            qstr=qstrr,
            provide_textual_feedback=feedback,
            n_batches=batches,
            max_feedback=max_feedback,
            box_drop_prob=0.0,
            max_results=10000,
        ),
        SessionParams(
            index_spec=IndexSpec(
                d_name=dataset, i_name="multiscale", c_name=catt
            ),
            interactive="pytorch",
            agg_method=agg_method,
            method_config=std_linear_config,
            batch_size=3,
        ),
    ),
    (
        BenchParams(
            name="coarse",
            ground_truth_category=catt,
            qstr=qstrr,
            provide_textual_feedback=feedback,
            n_batches=batches,
            max_feedback=max_feedback,
            box_drop_prob=0.0,
            max_results=10000,
        ),
        SessionParams(
            index_spec=IndexSpec(
                d_name=dataset, i_name="coarse", c_name=catt
            ),
            interactive="pytorch",
            agg_method=agg_method,
            method_config=std_linear_config,
            batch_size=3,
        ),
    ),
    (
        BenchParams(
            name="roibased",
            ground_truth_category=catt,
            qstr=qstrr,
            provide_textual_feedback=feedback,
            n_batches=batches,
            max_feedback=max_feedback,
            box_drop_prob=0.0,
            max_results=10000,
        ),
        SessionParams(
            index_spec=IndexSpec(
                d_name=dataset, i_name="roibased", c_name=catt
            ),
            interactive="pytorch",
            agg_method=agg_method,
            method_config=std_linear_config,
            batch_size=3,
        ),
    ),(
        BenchParams(
            name="detr",
            ground_truth_category=catt,
            qstr=qstrr,
            provide_textual_feedback=feedback,
            n_batches=batches,
            max_feedback=max_feedback,
            box_drop_prob=0.0,
            max_results=10000,
        ),
        SessionParams(
            index_spec=IndexSpec(
                d_name=dataset, i_name="detr", c_name=catt
            ),
            interactive="pytorch",
            agg_method=agg_method,
            method_config=std_linear_config,
            batch_size=3,
        ),
    ),(
        BenchParams(
            name="beit",
            ground_truth_category=catt,
            qstr=qstrr,
            provide_textual_feedback=feedback,
            n_batches=batches,
            max_feedback=max_feedback,
            box_drop_prob=0.0,
            max_results=10000,
        ),
        SessionParams(
            index_spec=IndexSpec(
                d_name=dataset, i_name="beit", c_name=catt
            ),
            interactive="pytorch",
            agg_method=agg_method,
            method_config=std_linear_config,
            batch_size=3,
        ),
    ),])
    return config

import json


def test_bench():
    ray.init("auto", namespace="seesaw", ignore_reinit_error=True)
    os.makedirs(TEST_SAVE, exist_ok=False)

    gdm = GlobalDataManager(TEST_ROOT)
    os.chdir(gdm.root)
    br = BenchRunner(gdm.root, results_dir=TEST_SAVE, redirect_output=False)

    for (i, (b, p)) in enumerate(make_config()):
        print("test case", i)
        path = br.run_loop(b, p)
        print("done with loop")
        bs = json.load(open(path + "/summary.json"))
        bs = BenchSummary(**bs)
        summ = get_metric_summary(bs.result)
        # check termination makes sense

        reached_batch_max = len(bs.result.session.gdata) == bs.bench_params.n_batches
        reached_max_results = bs.bench_params.max_results <= len(
            summ["hit_indices"]
        )  # could excced due to batching
        reached_all_results = bs.result.ntotal == len(summ["hit_indices"])
        reached_all_images = summ["nseen"] == bs.result.nimages

        satisfied_batch_max = len(bs.result.session.gdata) <= bs.bench_params.n_batches
        assert satisfied_batch_max
        assert (
            reached_batch_max
            or reached_max_results
            or reached_all_results
            or reached_all_images
        )

    print("testing the rest")
    a = get_all_session_summaries(TEST_SAVE)
    assert a.shape[0] == len(configs)
    assert os.path.isdir(a["session_path"].values[0])  # session path is correct

test_bench()
