from seesaw.seesaw_bench import *
import ray
from seesaw import GlobalDataManager, SessionParams, BenchParams, BenchRunner, IndexSpec, get_metrics_table
import random, string, os

ray.init('auto', namespace='seesaw')

#TEST_ROOT = '/home/gridsan/omoll/fastai_shared/omoll/seesaw_root/'
TEST_ROOT = '/home/gridsan/omoll/seesaw_root/'
tmp_name = ''.join([random.choice(string.ascii_letters) for _ in range(10)])
TEST_SAVE = f'{os.environ["TMPDIR"]}/test_save_{tmp_name}/'
os.makedirs(TEST_SAVE, exist_ok=False)

gdm = GlobalDataManager(TEST_ROOT)
os.chdir(gdm.root)
br = BenchRunner(gdm.root, results_dir=TEST_SAVE)

configs = [ (BenchParams(name='seesaw_test', ground_truth_category='aerosol can', qstr='aerosol can', 
              n_batches=4, max_feedback=None, box_drop_prob=0.0, max_results=10000), 
            SessionParams(index_spec=IndexSpec(d_name='data/lvis/', i_name='multiscale', c_name='aerosol can'),
              interactive='pytorch', warm_start='warm', batch_size=3, 
              minibatch_size=10, learning_rate=0.005, max_examples=500, 
              loss_margin=0.1, num_epochs=2, model_type='cosine')
            ),
  
            (BenchParams(name='baseline', ground_truth_category='aerosol can', qstr='aerosol can', 
              n_batches=4, max_results=10, max_feedback=None, box_drop_prob=0.0),      
            SessionParams(index_spec=IndexSpec(d_name='data/lvis/', i_name='coarse', m_name=None, c_name='aerosol can'), 
              interactive='plain', warm_start='warm', batch_size=3, 
              minibatch_size=10, learning_rate=0.005, max_examples=500, 
              loss_margin=0.1, num_epochs=2, model_type='cosine')
            ),

            (BenchParams(name='seesaw_test', ground_truth_category='aerosol can', qstr='aerosol can', 
              n_batches=4, max_feedback=None, box_drop_prob=0.0, max_results=10000), 
            SessionParams(index_spec=IndexSpec(d_name='data/lvis/', i_name='multiscale', c_name='aerosol can'),
              interactive='pytorch', warm_start='warm', batch_size=3, 
              minibatch_size=10, learning_rate=0.005, max_examples=3, 
              loss_margin=0.1, num_epochs=2, model_type='cosine')
            ),
]
import json

for (b,p) in configs:
  path = br.run_loop(b,p)
  bs = json.load(open(path + '/summary.json'))
  bs = BenchSummary(**bs)
  summ = get_metric_summary(bs.result.session)
  # check termination makes sense

  reached_batch_max = len(bs.result.session.gdata) == bs.bench_params.n_batches
  reached_max_results = bs.bench_params.max_results <= len(summ['hit_indices']) # could excced due to batching
  reached_all_results = bs.result.ntotal == len(summ['hit_indices'])
  reached_all_images = summ['total_seen'] == bs.result.nimages

  satisfied_batch_max = len(bs.result.session.gdata) <= bs.bench_params.n_batches
  assert satisfied_batch_max  
  assert reached_batch_max or reached_max_results or reached_all_results or reached_all_images



a = get_metrics_table(TEST_SAVE, at_N=3)
assert a.shape[0] == len(configs)
assert os.path.isdir(a['session_path'].values[0]) # session path is correct