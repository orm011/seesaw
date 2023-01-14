import pandas as pd
from seesaw.user_data_analysis import load_session_data, compute_session_tables
import os

# not sure what this was for..
# from seesaw.user_data_analysis import load_mturk_batches
# mturk_df = load_mturk_batches('/home/gridsan/omoll/mturk_batches/')
# real_session_ids = mturk_df['Answer.surveycode'].values.tolist() + ['DNykKNNIHVtNJv1pKwiTE1C0oWTXp54R']

base_path = '/home/gridsan/groups/fastai/seesaw/user_study_data/'
new_session_path = f'{base_path}/sessions_mturk/'
old_path = '/home2/gridsan/omoll/fastai_shared/omoll/user_study_sessions_v4/'

sessions = load_session_data(mturk_path, use_ray=False)
sess_review = pd.read_csv(f'{base_path}/cleanup/session review - sessions (2).csv', index_col=0).iloc[:102]
sess_review = sess_review.assign(include=sess_review['include user'].astype('bool'))
good_sessions1 = sess_review.session_path.values[sess_review.include.values]
new_sess_review = pd.read_csv(f'{base_path}/cleanup/session review - new_sessions (2).csv', index_col=0).iloc[:165]
new_sess_review = new_sess_review.assign(include=new_sess_review['Include user'].astype('bool'))
good_sessions2 = new_sess_review.session_path.values[new_sess_review.include.astype('bool').values]
good_sessions = set(good_sessions1).union(set(good_sessions2))

good_sessions_short = [ os.path.normpath(sname.replace(old_path, new_session_path)) for sname in good_sessions ]
new_session_tables = compute_session_tables(sessions, filter_paths=good_sessions_short)
accept_df = new_session_tables['accept_df']
#accept_df.to_parquet('time_view_v4.parquet')
accept_df.to_parquet('data_mturk.parquet')