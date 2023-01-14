#!/usr/bin/env python
# coding: utf-8

# In[11]:


import importlib
# import seesaw
#import ray
# from seesaw import load_session_data
import pandas as pd
import seesaw.user_data_analysis
importlib.reload(seesaw.user_data_analysis)


# In[16]:


from seesaw.user_data_analysis import load_mturk_batches, load_session_data


# In[7]:


#ls /home/gridsan/omoll/fastai_shared/omoll/user_study_sessions_v4/


# In[13]:


search_paths = ['/home/gridsan/omoll/fastai_shared/omoll/user_study_sessions_v4/']


# In[10]:


#ls /home/gridsan/omoll/fastai_shared/omoll/user_study_sessions_v4/


# In[14]:


mturk_df = load_mturk_batches('/home/gridsan/omoll/mturk_batches/')


# In[17]:


sessions = load_session_data(search_paths[0], use_ray=False)


# In[18]:


real_session_ids = mturk_df['Answer.surveycode'].values.tolist() + ['DNykKNNIHVtNJv1pKwiTE1C0oWTXp54R']


# In[19]:


session_summaries = get_session_summaries(sessions)


# In[20]:


# '/nvme_drive/orm/userstudy'


# In[73]:


sess_review = pd.read_csv('./session review - sessions (2).csv', index_col=0).iloc[:102]
sess_review = sess_review.assign(include=sess_review['include user'].astype('bool'))
good_sessions1 = sess_review.session_path.values[sess_review.include.values]
new_sess_review = pd.read_csv('./session review - new_sessions (2).csv', index_col=0).iloc[:165]
new_sess_review = new_sess_review.assign(include=new_sess_review['Include user'].astype('bool'))
good_sessions2 = new_sess_review.session_path.values[new_sess_review.include.astype('bool').values]
good_sessions = set(good_sessions1).union(set(good_sessions2))


# In[74]:


new_sess_review


# In[75]:


def make_clickable(val):
    return f'<a target="_blank" href="http://localhost:9001/session_info?path={val}">click here</a>'


# In[61]:


import pickle


# In[21]:


pickle.dump(sessions, open('all_sessions.pkl', 'wb'))


# In[35]:


len(sessions)


# In[38]:


sessions[0]['session_path']


# In[76]:


good_sessions_short = ['./' + sname[len('/home2/gridsan/omoll/fastai_shared/omoll/'):] for sname in good_sessions ]


# In[87]:


new_session_tables = compute_session_tables(sessions, filter_paths=good_sessions_short)


# In[98]:


new_session_tables['adf'].session_id.value_counts()


# In[48]:


#new_session_tables['adf']


# In[49]:


#new_session_tables['adf']


# In[29]:


#df = new_session_tables['adf'].assign(per_image_times=new_session_tables['adf'].per_image_times.map(lambda x : np.array(list(x.values()))))


# In[88]:


#df2 = df[[col for col in df.columns if col not in ['init_time', 'last_time', 'task_start_time', 'task_end_time']]]


# In[28]:


#df2.to_parquet('/home2/gridsan/omoll/fastai_shared/omoll/session_table.parquet')


# In[27]:


#new_session_tables


# In[88]:


accept_df = new_session_tables['accept_df']
#seen_df = pd.DataFrame(seen_timelines)


# In[89]:


accept_df


# In[80]:


accept_df.groupby(['qkey','mode','accepted']).size().reset_index()


# In[97]:


accept_df.to_parquet('time_view_v4.parquet')


# In[ ]:





# In[90]:


qaccept_df = accept_df.groupby(['qkey','mode','accepted']).elapsed_time.apply(bootstrap_stat).reset_index()
qaccept_df = qaccept_df.assign(grp=qaccept_df[['mode', 'accepted']].apply(tuple,axis=1))


# In[91]:


from plotnine import *


# In[96]:


( ggplot(qaccept_df) + 
     geom_errorbarh(aes(y='accepted', xmin='lower', xmax='high', 
                       group='grp', color='mode'), alpha=.5, position='identity') +
     geom_point(aes(y='accepted', x='med', group='grp', color='mode'), alpha=.5, position='identity') +
     geom_text(aes(y='accepted', x='high', label='n',
                    group='grp', color='mode'), va='bottom', ha='left', alpha=.5, position='identity') +
     facet_wrap(['qkey'], ncol=2) +
     annotate('vline', xintercept=6*60) +
     theme(subplots_adjust={'hspace':.2, 'wspace':.2},)
)


# In[95]:


( ggplot(qaccept_df) + 
     geom_errorbar(aes(x='accepted', ymin='lower', ymax='high', 
                       group='grp', color='mode'), alpha=.5, position='identity') +
     geom_point(aes(x='accepted', y='med',
                        group='grp', color='mode'), alpha=.5, position='identity') +
     geom_text(aes(x='accepted', y='high', label='n',
                        group='grp', color='mode'), va='bottom', ha='left', alpha=.5, position='identity') +

     facet_wrap(['qkey'], ncol=2) +
     coord_flip() +
     theme(subplots_adjust={'hspace':.2, 'wspace':.2},)
)


# In[55]:


( ggplot(qaccept_df) + 
     geom_errorbar(aes(x='accepted', ymin='lower', ymax='high', 
                       group='grp', color='mode'), alpha=.5, position='identity') +
     geom_point(aes(x='accepted', y='med',
                        group='grp', color='mode'), alpha=.5, position='identity') +
     geom_text(aes(x='accepted', y='high', label='n',
                        group='grp', color='mode'), va='bottom', ha='left', alpha=.5, position='identity') +

     facet_wrap(['qkey'], ncol=2) +
     coord_flip() +
     theme(subplots_adjust={'hspace':.2, 'wspace':.2},)
)


# In[63]:


( ggplot(qaccept_df) + 
     geom_errorbar(aes(x='accepted', ymin='lower', ymax='high', 
                       group='grp', color='mode'), alpha=.5, position='identity') +
     geom_point(aes(x='accepted', y='med',
                        group='grp', color='mode'), alpha=.5, position='identity') +
     geom_text(aes(x='accepted', y='high', label='n',
                        group='grp', color='mode'), va='bottom', ha='left', alpha=.5, position='identity') +

     facet_wrap(['qkey'], ncol=2) +
     coord_flip() +
     theme(subplots_adjust={'hspace':.2, 'wspace':.2},)
)


# In[29]:


( ggplot(qaccept_df) + 
     geom_errorbar(aes(x='accepted', ymin='lower', ymax='high', 
                       group='grp', color='mode'), alpha=.5, position='identity') +
     geom_point(aes(x='accepted', y='med',
                        group='grp', color='mode'), alpha=.5, position='identity') +
     geom_text(aes(x='accepted', y='high', label='n',
                        group='grp', color='mode'), va='bottom', ha='left', alpha=.5, position='identity') +

     facet_wrap(['qkey'], ncol=2) +
     coord_flip() +
     theme(subplots_adjust={'hspace':.2, 'wspace':.2},)
)


# In[32]:


dat = np.arange(15).reshape(3,5)


# In[34]:


np.median(dat, axis=0)


# In[85]:


( ggplot(qaccept_df) + 
     geom_errorbar(aes(x='accepted', ymin='lower', ymax='high', 
                       group='grp', color='mode'), alpha=.5, position='identity') +
     geom_point(aes(x='accepted', y='med',
                        group='grp', color='mode'), alpha=.5, position='identity') +
     geom_text(aes(x='accepted', y='high', label='n',
                        group='grp', color='mode'), va='bottom', ha='left', alpha=.5, position='identity') +

     facet_wrap(['qkey'], ncol=2) +
     coord_flip() +
     theme(subplots_adjust={'hspace':.2, 'wspace':.2},)
)


# In[ ]:


1. split into hard / easy cases
2. go over cases that overlap with benchmark. explain timing differences
3. break down into latency / image annotation cost

4. address errors to some extent.


# In[230]:


( ggplot(qaccept_df) + 
     geom_errorbar(aes(x='accepted', ymin='lower', ymax='high', 
                       group='grp', color='mode'), alpha=.5, position='identity') +
     geom_point(aes(x='accepted', y='med',
                        group='grp', color='mode'), alpha=.5, position='identity') +
     geom_text(aes(x='accepted', y='high', label='n',
                        group='grp', color='mode'), va='bottom', ha='left', alpha=.5, position='identity') +

     facet_wrap(['qkey'], ncol=2) +
     coord_flip() +
     theme(subplots_adjust={'hspace':.2, 'wspace':.2},)
)


# In[160]:


( ggplot(qaccept_df) + 
     geom_errorbar(aes(x='accepted', ymin='lower', ymax='high', 
                       group='grp', color='mode'), alpha=.5, position='identity') +
     geom_point(aes(x='accepted', y='med',
                        group='grp', color='mode'), alpha=.5, position='identity') +
     facet_wrap(['qkey'], ncol=2) +
     coord_flip() +
     theme(subplots_adjust={'hspace':.2, 'wspace':.2},)
)

