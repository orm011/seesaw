import ray
from seesaw import GlobalDataManager, SessionParams, BenchParams, BenchRunner
import random, string, os

ray.init('auto', namespace='seesaw')

#TEST_ROOT = '/home/gridsan/omoll/fastai_shared/omoll/seesaw_root/'
TEST_ROOT = '/home/gridsan/omoll/seesaw_root/'
tmp_name = ''.join([random.choice(string.ascii_letters) for _ in range(10)])
TEST_SAVE = f'{os.environ["TMPDIR"]}/test_save/{tmp_name}'

os.makedirs(TEST_SAVE, exist_ok=True)

gdm = GlobalDataManager(TEST_ROOT)
os.chdir(gdm.root)
br = BenchRunner(gdm.root, results_dir=TEST_SAVE)


p = SessionParams(interactive='pytorch', 
                                  warm_start='warm', batch_size=3, 
                                  minibatch_size=10, learning_rate=0.01, max_examples=225, loss_margin=0.1,
                                  tqdm_disabled=True, granularity='multi', positive_vector_type='vec_only', 
                                  num_epochs=2, n_augment=None, min_box_size=10, model_type='multirank2', 
                                  solver_opts={'C': 0.1, 'max_examples': 225, 'loss_margin': 0.05})

b = BenchParams(ground_truth_category='car', dataset_name='data/bdd_100/', index_name='multiscale', qstr='car',
           n_batches=4, max_feedback=10, box_drop_prob=0)

b2 = BenchParams(ground_truth_category='car', dataset_name='data/bdd_100/', index_name='coarse', qstr='car',
           n_batches=4, max_feedback=10, box_drop_prob=0)

print('multiscale')
br.run_loop(b,p)

print('coarse')
br.run_loop(b2,p)