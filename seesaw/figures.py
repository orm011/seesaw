import pandas as pd
import numpy as np
import sklearn.metrics
import pyroaring as pr
import math

from collections import defaultdict

from IPython.display import display
from plotnine import *
from plotnine import ggplot


def brief_formatter(num_sd):
    assert num_sd > 0
    def formatter(ftpt):
        if math.isclose(ftpt, 0.):
            return '0'
        
        if math.isclose(ftpt,1.):
            return '1'

        if ftpt < 1.:
            exp = -math.floor(math.log10(abs(ftpt)))
            fmt_string = '{:.0%df}' % (exp + num_sd - 1)
            dec = fmt_string.format(ftpt)    
        else:
            fmt_string = '{:.02f}'
            dec = fmt_string.format(ftpt)
            
        zstripped = dec.lstrip('0').rstrip('0')
        return zstripped.rstrip('.')
    return formatter

brief_format = brief_formatter(1)

def times_format(ftpt):
    return brief_format(ftpt) + 'x'

def make_labeler(fmt_func):
    def fun(arrlike):
        return list(map(fmt_func, arrlike))
    return fun

def side_by_side_comparison(stats, *, baseline_variant, metric):
    stats = stats.assign(session_index=stats.index.values)
    v1 = stats
    metrics = list(set(['nfound', 'nfirst'] + [metric]))

    v1 = v1[['session_index', 'dataset', 'category', 'variant', 'ntotal'] + metrics]
    v2 = stats[stats.variant == baseline_variant]
    rename_dict = {}
    for m in metrics + ['variant', 'session_index']:
        rename_dict[m] = f'base_{m}'
            
    v2 = v2[['dataset', 'category', 'variant', 'session_index'] + metrics]
    v2['base'] = v2[metric]
    v2 = v2.rename(mapper=rename_dict, axis=1)
    assert 'variant' not in v2.columns.values
    assert 'base_variant' in v2.columns.values
    
    sbs = v1.merge(v2, right_on=['dataset', 'category'], left_on=['dataset', 'category'], how='outer')
    sbs = sbs.assign(ratio=sbs[metric]/sbs['base'])
    sbs = sbs.assign(delta=sbs[metric] - sbs['base'])
    return sbs

def bsw_table(sbs, *, variant, metric, reltol):
    invtol = 1/reltol
    sbs = sbs.assign(better=sbs[metric] > reltol*sbs.base, worse=sbs[metric] < invtol*sbs.base,
                    same=sbs[metric].between(invtol*sbs.base, reltol*sbs.base))
    bsw = sbs[sbs.variant == variant].groupby('dataset')[['better', 'same', 'worse']].sum()
    dstot = bsw.sum(axis=1)
    bsw = bsw.assign(total=dstot)
    tot = bsw.sum()
    bsw_table = bsw.transpose().assign(total=tot).transpose()
    display(bsw_table)
    print(bsw_table.to_latex())
    return bsw_table
    
def summary_breakdown(sbs, metric):
    base_metric = 'base'
    part = sbs[base_metric].map(lambda x : '1.' if x > .3 else '.3' if x > .1 else '.1')
    sbs = sbs.assign(part=part)
    totals = sbs.groupby(['part', 'dataset', 'variant'])[metric].mean().reset_index().groupby(['part', 'variant'])[metric].mean().unstack(level=0)
    counts = sbs.groupby(['part', 'dataset', 'variant']).size().rename('cats').reset_index().groupby(['part', 'variant']).cats.sum().unstack(level=0)
    
    tr = totals.transpose()
    count_col = counts.transpose()['plain']
    
    tr = tr.assign(counts=count_col)[['counts'] + tr.columns.values.tolist()].transpose()
    return tr

def remove_leading_zeros(fmtr):
    def fun(x):
        fx =  fmtr(x).lstrip('0')
        
        if x >= 1:
            fx = fx.rstrip('0')
            
        fx = fx.rstrip('.')
        return fx
    return fun

def comparison_table(tot_res, *, variant, baseline_variant):
    sys = variant
    baseline = baseline_variant
    tot_res = tot_res.transpose()
    tot_res = tot_res.assign(ratio=tot_res[sys]/tot_res[baseline])
    total_counts = tot_res['counts'].sum()
    tot_res = (tot_res[['counts', baseline,sys, 'ratio']]
            .rename(mapper={baseline:'baseline',sys:'this work'}, axis=1))
    tot_res = tot_res.transpose()    
    fmtter = remove_leading_zeros('{:.02f}'.format)
    print('total_counts: ', total_counts)
    with pd.option_context('display.float_format', fmtter):
        display(tot_res)
    print(tot_res.to_latex(float_format=fmtter))
    return tot_res

def ablation_table(tot_res, variants_list):
    baseline = variants_list[0]
    inter = variants_list[1]
    sys = variants_list[2]
    #tot_res = tot_res.loc[]
    tot_res = tot_res.transpose()[[baseline,inter,sys]].rename(mapper={baseline:'semantic embeddding',
                                                 inter:'+ multiscale search',
                                      sys:'+ feedback fusion'}, axis=1).transpose()

    # display()#.#style.highlight_max(axis=0))
    # print(res)
    # print(tot_res.diff().to_latex(float_format=brief_format))
    delta_ablation  = tot_res.diff().iloc[1:]
    delta_ablation = delta_ablation.rename(mapper=lambda x : x + ' delta', axis=1)
    ablation = tot_res#.rename(mapper=lambda x : str(x), axis=1)
    ablation = pd.concat([ablation, delta_ablation], axis=1)[['.1', '.1 delta', '.3', '.3 delta', '1.', '1. delta']]

    with pd.option_context('display.float_format', remove_leading_zeros('{:.02f}'.format)):
        display(ablation)
    print(ablation.to_latex(float_format=remove_leading_zeros('{:.02f}'.format)))
    
    return ablation


def print_tables(stats, *, variant,  baseline_variant, metric, reltol, show_latex=False):
    res = {}
    all_vars = stats.groupby(['dataset', 'category', 'variant',])[metric].mean().unstack(-1)
    means = all_vars.groupby('dataset').mean()
    counts = all_vars.groupby('dataset').size().rename('num_queries')
    per_dataset = pd.concat([means, counts],axis=1)
    print('by dataset')
    display(per_dataset)
    res['by_dataset'] = per_dataset

    sbs = side_by_side_comparison(stats, baseline_variant=baseline_variant, metric=metric)
    res['sbs'] = sbs
    assert 'variant' in sbs.columns.values

    print('bsw')
    bsw = bsw_table(sbs, variant=variant, metric=metric, reltol=reltol)
    res['bsw'] = bsw
    
    tot_res = summary_breakdown(sbs, metric=metric)
    cmp = comparison_table(tot_res, variant=variant, baseline_variant=baseline_variant)
    res['comparison'] = cmp

    print('ablation')
    if variant == 'seesaw':
      abtab = ablation_table(tot_res, variants_list=['plain', 'multiplain','seesaw',])
      display(abtab)
      res['ablation'] = abtab
    else:
      print('skipping ablation table bc. using other variant')

    x = np.geomspace(.02, 1, num=5)
    y = 1/x
    diag_df = pd.DataFrame({'x':x, 'y':y})

    ### plot
    plotdata = sbs[sbs.variant == variant]
    xcol = 'base'
    ycol = 'ratio'
    plotdata = plotdata.assign(x=plotdata[xcol], y=plotdata[ycol])
    plotdata = plotdata.assign(sbs_index=plotdata.index.values)
    session_text = plotdata[['session_index', 'base_session_index']].apply(tuple, axis=1).map(lambda tup : f'{tup[0]} vs. {tup[1]}')
    plotdata = plotdata.assign(session_text=session_text)
    res['plotdata'] = plotdata


    scatterplot = (ggplot(plotdata)
        + geom_point(aes(x='x', y='y', fill='dataset', color='dataset'), alpha=.6, size=1.) 
    #                 shape=plotdata.dataset.map(lambda x : '.' if x in ['lvis','objectnet'] else 'o'), 
    #                 size=plotdata.dataset.map(lambda x : 1. if x in ['lvis','objectnet'] else 2.))
    #  + geom_text(aes(x='base', y='delta', label='category', color='dataset'), va='bottom', 
    #              data=plotdata1[plotdata1.ratio < .6], 
    #              position=position_jitter(.05, .05), show_legend=False)
        + geom_line(aes(x='x', y='y'), data=diag_df)
        + geom_text(aes(x='x', y='y', label='session_text'), va='top', data=plotdata[(plotdata.y < .4) | (plotdata.y > 3)])
     + ylab(ycol)
    #               + geom_area(aes(y2=1.1, y=.9), linetype='dashed', alpha=.7)
                   + geom_hline(aes(yintercept=1.1), linetype='dashed', alpha=.7)
                   + geom_hline(aes(yintercept=.9), linetype='dashed', alpha=.7)


                    + geom_vline(aes(xintercept=.1,), linetype='dashed', alpha=.7)
                    + geom_vline(aes(xintercept=.3,), linetype='dashed', alpha=.7)
    #+ geom_abline()
    #    + geom_point(aes(x='recall', y='precision', color='variant'), size=1.)
    #     + facet_wrap(facets=['cat'], ncol=6, scales='free_x')
     + xlab(xcol)
    # +scale_color_discrete()
        + theme(figure_size=(8,5), legend_position='top',
               subplots_adjust={'hspace': 0.5}, legend_title=element_blank(),
                legend_box_margin=-1, legend_margin=0.,
                axis_text=element_text(size=12, margin={'t':.2, 'l':-.3}),
                legend_text=element_text(size=11),
                axis_title=element_text(size=12, margin={'r':-.2, 'b':0., 'l':0, 't':0.}),
               )
        + scale_x_log10(labels=make_labeler(brief_format), breaks=[.01, .1, .3, 1.])
        + scale_y_log10(labels=make_labeler(brief_format), breaks=[.5, 0.9, 1.1, 2., 3.,6, 12])
    )
    display(scatterplot)
    return res
