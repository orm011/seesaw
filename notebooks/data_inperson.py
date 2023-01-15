import pandas as pd
import os
import numpy as np
from seesaw.user_data_analysis import load_session_data

import pytz
import datetime

from seesaw.basic_types import is_image_accepted, Imdata
from seesaw.user_data_analysis import get_session_summary

import datetime
import hashlib

def get_user_hash(user):
    md = hashlib.md5(user.encode())
    user_hash = md.hexdigest()[:10]
    return user_hash


real_users = [
    'ferdy',
    'jialin',
    'favyen',
    'ziniu',
    'xinjing',
    'kapil',
    'eugenie',
    'faraz',
    'brit',
    'test',#'brit' (between times 1649076970  1649084170)
    'markos',
    'harshal',
    'geoffrey',
    'siva',
]

brit_start = datetime.datetime(2022, 4, day=4, hour=8, minute=30, tzinfo=pytz.timezone('US/Eastern'))
brit_end = datetime.datetime(2022, 4, day=4, hour=9, minute=15, tzinfo=pytz.timezone('US/Eastern'))
users = real_users
base_modes = ['pytorch', 'default']
hash2uname = {get_user_hash(u):u for u in users}

excluded_session = [
    'pzldslyrpx', # a test session under brit's name
]


def linear_gdata(sess, accepted_only=True):
    summary = get_session_summary(sess)
    session = sess['session']
    action_log = session['action_log']
    gdata = session['gdata']
        
    task_started = summary['task_started']
    ret = []
    ret.append([-1,-1, 0, 0, False])
    for i,l in enumerate(gdata):
        for j,r in enumerate(l):
            is_accepted = is_image_accepted(Imdata(**r))
            if 'timing' in r:
                for time_rec in r['timing']:
                    ret.append([i,j,time_rec['start_ms']/1000. - task_started, 
                                time_rec['end_ms']/1000. - task_started, is_accepted])
                    break
            
    df = pd.DataFrame(ret, columns=['i', 'j', 'start_s', 'end_s', 'accepted'])
    df = df.sort_values(['start_s']).reset_index(drop=True)
    assert df.start_s.is_monotonic_increasing 
    df = df.assign(total_accepted=df.accepted.cumsum(), total_seen=np.arange(df.shape[0]))
    if accepted_only:
        df = df.groupby('total_accepted').apply(lambda x : x.head(n=1)).reset_index(drop=True)
    df = df.assign(**summary)
    return df

                

def session_totals(sess, seen_limit = 75, accepted_limit=10, mode=None, clean_paths=None):
    
    if clean_paths:
        if os.path.normpath(sess['session_path']) not in clean_paths:
            return []
    
    mode in [None, 'image_timing', 'time_progress']
    sess_info = sess['session']
    params = sess_info['params']
    action_log = sess_info['action_log']
    
    sid = params.get('session_id')
    if sid is None or sid in excluded_session: # some sessions are excluded
        return [] 
    
    other_params = params['other_params']
    uname = hash2uname.get(params['other_params'].get('user'))
    
    if uname not in ['test', None]:
        pass
    elif (brit_start.timestamp() <= action_log[0]['time']  <= brit_end.timestamp()):
        uname = 'brit'
    else:
        return []
    
    if uname == 'siva':
        if other_params['qkey'] == 'cd':
            other_params['qkey'] = 'dg'
        elif other_params['qkey'] == 'dg':
            other_params['qkey'] = 'cd'
    
    df = linear_gdata(sess, accepted_only=False)
    df['qkey'] = other_params['qkey']
    df['uname'] = uname
    return [df]
        
base_path = '/home/gridsan/groups/fastai/seesaw/user_study_data/'
new_session_path = f'{base_path}/sessions_mit/'
old_path = '/home2/gridsan/omoll/fastai_shared/omoll/user_study_sessions2/'

sessions = load_session_data(new_session_path, use_ray=False)
target_accepted = 10

summaries = []
for x in sessions:
    ret = session_totals(x, accepted_limit=target_accepted)
    summaries.extend(ret)

#tot = pd.DataFrame.from_records(summaries)
tot = pd.concat(summaries, ignore_index=True)
#tot = tot.sort_values('start_time', ascending=True)
#tot = tot.reset_index(drop=True)
# user_df =pd.DataFrame()
# tot = tot.merge(user_df, left_on='user', right_on='user', how='left')
tot = tot[tot.uname.isin(real_users) & ~tot.qkey.isin(['amb', 'pc']) & ~tot.uname.isin(['jialin'])]
tot = tot.groupby(['uname', 'session_id', 'init_time']).apply(lambda df : df.sort_values('last_time', ascending=False).head(n=1)).reset_index(drop=True)
tot = tot.assign(completed=tot.accepted >= target_accepted)

clean_session_paths = [os.path.normpath(p) for p in tot['session_path'].values.tolist()]
gdatas = sum(map(lambda x : session_totals(x, accepted_limit=target_accepted, clean_paths=clean_session_paths), sessions), start=[])
all_gdata = pd.concat(gdatas, ignore_index=True)
timing_summaries = sum(map(lambda x : session_totals(x, accepted_limit=10, mode='image_timing'), sessions), start=[])
# image_timings = pd.DataFrame.from_records(timing_summaries)
image_timings = pd.concat(timing_summaries, ignore_index=True)
image_timings = image_timings[(~image_timings.qkey.isin(['pc'])) & (~image_timings.uname.isin(['jialin'])) ]
# time_prog = sum(map(lambda x : session_totals(x, accepted_limit=10, mode='time_progress'), sessions), start=[])
time_prog_df = all_gdata.assign(elapsed_time=all_gdata['end_s'], accepted=all_gdata['total_accepted'])
time_prog_df = time_prog_df[~time_prog_df.uname.isin(['jialin'])]
tpdf = time_prog_df[time_prog_df.elapsed_time <= 6*60]
tpdf.drop('last_time', axis=1).to_parquet('time_view_v3.parquet')


# plot_time_prog_data = time_prog_df.groupby(['qkey', 'mode','accepted']).elapsed_time.apply(bootstrap_stat).reset_index()
# plot_time_prog_data = plot_time_prog_data.assign(grp=plot_time_prog_data[['mode', 'accepted']].apply(tuple,axis=1))