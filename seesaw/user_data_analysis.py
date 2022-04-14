import glob
import pandas as pd
import os
import datetime, pytz
import math
from seesaw import is_image_accepted, Imdata
import hashlib
import random
import string
import datetime
import pytz
import numpy as np

def load_mturk_batches():
    mturk_batches = glob.glob('./Batch*csv')
    dfs = []
    for b in mturk_batches:
        dfs.append(pd.read_csv(b))
        
    mturk_df = pd.concat(dfs, ignore_index=True)
    return mturk_df

def process_ts(timestamp_s, timezone=pytz.timezone('US/Eastern')):
    """converts epoch (in seconds) to an easier to read datetime with timezone"""
    rounded_ts = round(timestamp_s, ndigits=3)
    ts = pd.to_datetime(rounded_ts, unit='s')
    tsloc = pytz.utc.localize(ts)
    ts_east = tsloc.astimezone(timezone)
    return ts_east

def process_action_log(log):
    start_entry = None
    end_entry = None
    
    START_MESSAGE = 'task.started'
    END_MESSAGE = 'task.end'
    
    IMAGE_START_MESSAGE = 'selection.start'
    IMAGE_LOAD_MESSAGE = 'image_loaded'
    IMAGE_END_MESSAGE = 'selection.end'
    
    entry_dict = {}
    
    im_start_entry = None
    im_load_entry = None
    im_end_entry = None
    
    accepted_so_far = -1
    seen_so_far = -1
    
    seen_timeline = []
    accepted_timeline = []
    
    for entry in log:
        if end_entry is not None:
            break
            
        if start_entry is None:
            if entry['message'] != START_MESSAGE:
                continue
            else:
                start_entry = entry
                start_time = entry['time']
            
        if entry['message'] in [IMAGE_START_MESSAGE, IMAGE_LOAD_MESSAGE, IMAGE_END_MESSAGE]:
            if not im_start_entry and entry['message'] == IMAGE_START_MESSAGE:
                im_start_entry = entry
            elif im_start_entry and entry['message'] == IMAGE_LOAD_MESSAGE:
                im_load_entry = entry
            elif im_start_entry and entry['message'] == IMAGE_END_MESSAGE:
                im_end_entry = entry
                key = tuple(im_start_entry['other_fields'].values())
                key2 = tuple(im_start_entry['other_fields'].values())
                assert key == key2
                delta = im_end_entry['time'] - im_start_entry['time']
                acc_t = entry_dict.get(key,0)
                entry_dict[key] = acc_t + delta

                im_start_entry = None
                im_load_entry = None
                im_end_entry = None
                
            
        if entry['message'] == END_MESSAGE:
            end_entry = entry
            
        if len(entry_dict) > seen_so_far:
            seen_so_far = len(entry_dict)
            seen_timeline.append({'seen':seen_so_far, 'elapsed_time': entry['time'] - start_time})
        
        if entry['accepted'] > accepted_so_far:
            accepted_so_far = entry['accepted']
            accepted_timeline.append({'accepted':accepted_so_far, 'elapsed_time':entry['time'] - start_time})

    return {'accepted_timeline':accepted_timeline, 
           'seen_timeline':seen_timeline, 'per_image_times':entry_dict, 
               'start_entry':start_entry, 'end_entry':end_entry}

def get_session_summary(sess):
    session = sess['session']
    action_log = session['action_log']
    params = session['params']
    other_params = params['other_params']
    session_path = sess['session_path']

    init_time = process_ts(action_log[0]['time'])
    last_time = process_ts(action_log[-1]['time'])
    
    ans = { 'session_path':session_path,
            'init_time':init_time,
            'last_time':last_time,
            **other_params,
           }

    
    if 'session_id' in session and 'session_id' not in ans:
        ans['session_id'] = session['session_id']
        
    return ans

def get_session_summaries(sessions, latest_only=True):
    all_df = pd.DataFrame(list(map(get_session_summary, sessions)))

    def get_latest(gp):
        gp = gp.sort_values('last_time', ascending=False)
        return gp.head(n=1)

    if latest_only:
        all_df = all_df.groupby(['session_id', 'qkey', 'init_time']).apply(get_latest).reset_index(drop=True)
    return all_df

def process_session(sess, filter_paths=None):
    summary = get_session_summary(sess)
    if filter_paths:
        if summary['session_path'] not in filter_paths:
            return []
            
    log_results = process_action_log(sess['session']['action_log'])
    if log_results['start_entry'] and log_results['end_entry']:
        summary['task_start_time'] = process_ts(log_results['start_entry']['time'])
        summary['task_end_time'] = process_ts(log_results['end_entry']['time'])
        summary['task_duration_s'] = log_results['end_entry']['time'] - log_results['start_entry']['time']
        summary['total_images_accepted'] = log_results['end_entry']['accepted']
        summary['total_images_seen'] = len(log_results['seen_timeline'])
        summary.update(**log_results)
    else:
        return []
    
    return [summary]

def compute_session_tables(sessions, filter_paths):
    all_summaries = sum(map(lambda x : process_session(x, filter_paths=filter_paths), sessions), start=[])
    adf = pd.DataFrame(all_summaries)
    accept_timelines = []
    seen_timelines = []
    image_times = []
    max_accepted = 10
    for s in all_summaries:
        if s['end_entry'] and s['start_entry']:
            duration = s['end_entry']['time'] - s['start_entry']['time']
        else:
            continue
        
        acc_timeline = s['accepted_timeline']
        for ent in acc_timeline:
            ent['session_id'] = s['session_id']
            ent['qkey'] = s['qkey']
            ent['mode'] = s['mode']
            ent['session_path'] = s['session_path']
            ent['duration'] = duration
            accept_timelines.append(ent)
                
        total_accepted = ent['accepted']
        for i in range(total_accepted+1, max_accepted+1):
            temp_ent = ent.copy()
            temp_ent['accepted'] = i
            temp_ent['elapsed_time'] = duration
            accept_timelines.append(temp_ent)
            
        for ent in seen_timelines:
            ent['session_id'] = s['session_id']
            ent['qkey'] = s['qkey']
            ent['mode'] = s['mode']
            ent['session_path'] = s['session_path']
            seen_timelines.append(ent)

    accept_df = pd.DataFrame(accept_timelines)
    seen_df = pd.DataFrame(seen_timelines)
    return {'adf':adf, 'accept_df':accept_df, 'seen_df':seen_df}


def bootstrap_stat(ser, confidence_level=.95, n_resamples=10000):
    samp = ser.sample(n=ser.shape[0]*n_resamples, replace=True)
    samp = samp.values.reshape((ser.shape[0],-1))
    means = np.mean(samp, axis=0)
    assert means.shape[0] == n_resamples
    q0 = (1 - confidence_level)/2.
    q1 = 1. - q0
    assert math.isclose(q1 - q0, confidence_level)    
    x = np.quantile(means, q=[q0, .5, q1])

    return pd.DataFrame([{'lower':x[0], 'med':ser.median(), 'mean':ser.mean(), 'high':x[2], 'confidence_level':confidence_level, 'n':ser.shape[0]}])
