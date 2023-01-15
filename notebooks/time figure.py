
import pandas as pd
import seesaw.user_data_analysis
import importlib
importlib.reload(seesaw.user_data_analysis)
from seesaw.user_data_analysis import *

mturk_path = './data_mturk.parquet'
mit_path = './time_view_v3.parquet'
accept_df = pd.concat([pd.read_parquet(mit_path), pd.read_parquet(mturk_path)], ignore_index=True)

accept_df = accept_df[accept_df.accepted <= 10] 

accept_df[['session_id', 'uname']].apply(lambda x : x.session_id if not x.uname else x.uname, axis=1)

accept_df.groupby(['session_id']).size()

qaccept_df = accept_df.groupby(['qkey','mode','accepted']).elapsed_time.apply(bootstrap_stat).reset_index()
qaccept_df = qaccept_df.assign(grp=qaccept_df[['mode', 'accepted']].apply(tuple,axis=1))


from plotnine import *


codes = {
    'pc':dict(dataset='bdd', qstr='police cars', 
                description='''Police vehicles that have lights and some marking related to police. ''',
                negative_description='''Sometimes private security vehicles or ambulances look like police cars but should not be included'''),
    'dg':dict(dataset='bdd', qstr='A - dogs'),
    'cd':dict(dataset='bdd', qstr='C - car with open doors', 
                description='''Any vehicles with any open doors, including open trunks in cars, and rolled-up doors in trucks and trailers.''',
                negative_description='''We dont count rolled down windows as open doors'''),
    'wch':dict(dataset='bdd', qstr='B - wheelchairs',
                description='''We include wheelchair alternatives such as electric scooters for the mobility impaired. ''',
                negative_description='''We do not include wheelchair signs or baby strollers'''),
    'mln':dict(dataset='coco', qstr='D - melon', 
                description='''We inclulde both cantaloupe (orange melon) and honeydew (green melon), whole melons and melon pieces. ''',
                negative_description='''We dont include any other types of melon, including watermelons, papaya or pumpkins, which can look similar. 
                If you cannot tell whether a fruit piece is really from melon don't sweat it and leave it out.'''),
    'spn':dict(dataset='coco', qstr='E - spoons', 
                description='''We include spoons or teaspons of any material for eating. ''', 
                negative_description='''We dont include the large cooking or serving spoons, ladles for soup, or measuring spoons.'''),
    'dst':dict(dataset='objectnet', qstr='F - dustpans',
                description='''We include dustpans on their own or together with other tools, like brooms, from any angle.''',
                negative_description='''We dont include brooms alone'''),
    'gg':dict(dataset='objectnet', qstr='G - egg cartons',
                description='''These are often made of cardboard or styrofoam. We include them viewed from any angle.''', 
                negative_description='''We dont include the permanent egg containers that come in the fridge''')
}


qaccept_df = qaccept_df.assign(qstr=qaccept_df.qkey.map(lambda x : codes[x]['qstr']))


qaccept_df = qaccept_df[~qaccept_df.qkey.isin(['pc'])]


qaccept_df = qaccept_df.assign(method=qaccept_df['mode'].map(lambda m : {'pytorch': 'this work', 'default':'baseline'}[m]))


show_minutes = lambda x : f'{int(x/60):d}'



qaccept_df



qaccept_df.groupby(['session_id', 'user', 'qkey']).size()


plot = ( ggplot(qaccept_df) + 
     geom_errorbarh(aes(y='accepted', xmin='lower', xmax='high', 
                       group='grp', color='method'), height=1., alpha=.5, position='identity', show_legend=False) +
     geom_point(aes(y='accepted', x='med', group='grp', color='method'), alpha=.5, position='identity') +
#      geom_text(aes(y='accepted', x='high', label='n',
#                     group='grp', color='mode'), va='bottom', ha='left', alpha=.5, position='identity') +
     facet_wrap(['qstr'], ncol=2, ) +
     scale_x_continuous(breaks=[0, 60, 120, 180, 240, 300, 360], labels=lambda a : list(map(show_minutes,a)) )+
     scale_y_continuous(breaks=[0, 3, 6, 10]) +
     xlab('elapsed time (min)') +
     ylab('results marked relevant') + 
     annotate('vline', xintercept=6*60, linetype='dashed') +
#      annotate('text', label=360, x=360,y=0, va='top')
     theme(legend_position='top', legend_direction='horizontal', legend_title=element_blank(), legend_box_margin=0,
          legend_margin=0, plot_margin=0, panel_grid_minor=element_blank(), figure_size=(3,5), )
)
plot

