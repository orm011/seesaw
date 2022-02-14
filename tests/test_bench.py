from seesaw.seesaw_bench import *
import ray
from seesaw import GlobalDataManager, SessionParams, BenchParams, BenchRunner, IndexSpec, get_all_session_summaries
import random, string, os

ray.init('auto', namespace='seesaw', ignore_reinit_error=True)

#TEST_ROOT = '/home/gridsan/omoll/fastai_shared/omoll/seesaw_root/'
TEST_ROOT = '/home/gridsan/omoll/seesaw_root/'
tmp_name = ''.join([random.choice(string.ascii_letters) for _ in range(10)])
TEST_SAVE = f'~/tmp/seesaw_tests/test_save_{tmp_name}/'
TEST_SAVE = os.path.expanduser(TEST_SAVE)
os.makedirs(TEST_SAVE, exist_ok=False)

gdm = GlobalDataManager(TEST_ROOT)
os.chdir(gdm.root)
br = BenchRunner(gdm.root, results_dir=TEST_SAVE, redirect_output=False)

from seesaw.textual_feedback_box import std_textual_config

cat = 'soya milk'
qstr = 'a soya milk'

cat_objectnet = 'air freshener'
qstr_objectnet = 'an air freshener'
## chosen so there are some positives withi this range

configs = [ 
            (BenchParams(name='seesaw_test', ground_truth_category=cat, qstr=qstr, 
              n_batches=4, max_feedback=None, box_drop_prob=0.0, max_results=10000), 
            SessionParams(index_spec=IndexSpec(d_name='data/lvis/', i_name='multiscale', c_name=cat),
              interactive='pytorch', warm_start='warm', batch_size=3, 
              minibatch_size=10, learning_rate=0.005, max_examples=500, 
              loss_margin=0.1, num_epochs=2, model_type='cosine')
            ),
  
            (BenchParams(name='baseline', ground_truth_category=cat, qstr=qstr, 
              n_batches=4, max_results=10, max_feedback=None, box_drop_prob=0.0),      
            SessionParams(index_spec=IndexSpec(d_name='data/lvis/', i_name='coarse', m_name=None, c_name=cat), 
              interactive='plain', warm_start='warm', batch_size=3, 
              minibatch_size=10, learning_rate=0.005, max_examples=500, 
              loss_margin=0.1, num_epochs=2, model_type='cosine')
            ),

            (BenchParams(name='seesaw_test', ground_truth_category=cat, qstr=qstr, 
              n_batches=4, max_feedback=None, box_drop_prob=0.0, max_results=10000), 
            SessionParams(index_spec=IndexSpec(d_name='data/lvis/', i_name='multiscale', c_name=cat),
              interactive='pytorch', warm_start='warm', batch_size=3, 
              minibatch_size=10, learning_rate=0.005, max_examples=3, 
              loss_margin=0.1, num_epochs=2, model_type='cosine')
            ),
            (BenchParams(name='seesaw_test_textual', ground_truth_category=cat, qstr=qstr, 
              n_batches=4, max_feedback=None, box_drop_prob=0.0, max_results=10000), 
            SessionParams(index_spec=IndexSpec(d_name='data/lvis/', i_name='multiscale', c_name=cat),
              interactive='textual', method_config=std_textual_config, warm_start='warm', batch_size=3, 
              minibatch_size=10, learning_rate=0.005, max_examples=3, 
              loss_margin=0.1, num_epochs=2, model_type='cosine')
            ),

            (BenchParams(name='seesaw_test_textual', ground_truth_category=cat_objectnet, qstr=qstr_objectnet, 
              n_batches=4, max_feedback=None, box_drop_prob=0.0, max_results=10000), 
            SessionParams(index_spec=IndexSpec(d_name='data/objectnet/', i_name='multiscale', c_name=cat_objectnet),
              interactive='textual', method_config=std_textual_config, warm_start='warm', batch_size=3, 
              minibatch_size=10, learning_rate=0.005, max_examples=3, 
              loss_margin=0.1, num_epochs=2, model_type='cosine')
            ),


]
import json

for (i,(b,p)) in enumerate(configs):
  print('test case', i)
  path = br.run_loop(b,p)
  print ('done with loop')
  bs = json.load(open(path + '/summary.json'))
  bs = BenchSummary(**bs)
  summ = get_metric_summary(bs.result)
  # check termination makes sense

  reached_batch_max = len(bs.result.session.gdata) == bs.bench_params.n_batches
  reached_max_results = bs.bench_params.max_results <= len(summ['hit_indices']) # could excced due to batching
  reached_all_results = bs.result.ntotal == len(summ['hit_indices'])
  reached_all_images = summ['nseen'] == bs.result.nimages

  satisfied_batch_max = len(bs.result.session.gdata) <= bs.bench_params.n_batches
  assert satisfied_batch_max  
  assert reached_batch_max or reached_max_results or reached_all_results or reached_all_images

print('testing the rest')
a = get_all_session_summaries(TEST_SAVE)
assert a.shape[0] == len(configs)
assert os.path.isdir(a['session_path'].values[0]) # session path is correct