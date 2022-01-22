import ray
from seesaw import GlobalDataManager, SessionParams, BenchParams, BenchRunner, IndexSpec
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

b = BenchParams(name='seesaw_test', 
  ground_truth_category='aerosol can', qstr='aerosol can', 
  n_batches=3, max_feedback=None, box_drop_prob=0.0, max_results=10000)

p = SessionParams(index_spec=IndexSpec(d_name='data/lvis/', i_name='multiscale', c_name='aerosol can'), interactive='pytorch', warm_start='warm', batch_size=3, 
  minibatch_size=10, learning_rate=0.005, max_examples=500, 
  loss_margin=0.1, num_epochs=2, model_type='cosine')

print('lvis case')
br.run_loop(b,p)

p = SessionParams(index_spec=IndexSpec(d_name='data/bdd_100/', i_name='multiscale', c_name='car'),
                    interactive='pytorch', 
                                  warm_start='warm', batch_size=3, 
                                  minibatch_size=10, learning_rate=0.01, max_examples=225, loss_margin=0.1,
                                  num_epochs=2, model_type='multirank2')

p2 = p.copy(update=dict(index_spec=IndexSpec(d_name='data/bdd_100/', i_name='coarse', c_name='car')))

b = BenchParams(name='b', ground_truth_category='car',  qstr='car',
           n_batches=4, max_feedback=10, box_drop_prob=0, max_results=10000)

print('bdd multiscale')
br.run_loop(b,p)

print('bdd coarse')
br.run_loop(b,p2)