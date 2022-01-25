import pandas as pd
import numpy as np
import sklearn.metrics
import pyroaring as pr
import math

from IPython.display import display
from plotnine import *


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

def ndcg_score_fn(*, hit_indices, total_frames_seen, at_k_frames, total_positives):
    hit_indices = hit_indices.astype('int')
    assert (hit_indices < total_frames_seen).all()
    assert total_positives > 0, 'undefined if no positives'
    assert at_k_frames > 0, 'undefined if no frames'
    if at_k_frames > total_frames_seen:
      at_k_frames = total_frames_seen # decide what to do in the calling code (can do this check)
      #assert hit_indices.shape[0] == total_positives, 'no data available for k larger than total_frames_seen'
        
    assert at_k_frames <= total_frames_seen
    best_hits = np.zeros(at_k_frames)
    best_hits[:total_positives] = 1.

    hits = np.zeros(at_k_frames)
    rel_indices = hit_indices[hit_indices < at_k_frames]    
    hits[rel_indices] = 1.
    
    wi = np.arange(at_k_frames)
    ws = 1./np.log2(wi+2)
    # following https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/metrics/_ranking.py#L1239
    top = hits @ ws
    best = best_hits @ ws
    return top/best

def ndcg_rank_score(ytrue, ordered_idxs):
    '''
        wraps sklearn.metrics.ndcg_score to grade a ranking given as an ordered array of k indices,
        rather than as a score over the full dataset.
    '''
    ytrue = ytrue.astype('float')
    # create a score that is consistent with the ranking.
    ypred = np.zeros_like(ytrue)
    fake_scores = np.arange(ordered_idxs.shape[0],0,-1)/ordered_idxs.shape[0]
    assert (fake_scores > 0).all()
    ypred[ordered_idxs] = fake_scores
    return sklearn.metrics.ndcg_score(ytrue.reshape(1,-1), y_score=ypred.reshape(1,-1), k=ordered_idxs.shape[0])

def test_score_sanity():
    ytrue = np.zeros(10000)
    randorder = np.random.permutation(10000)
    
    numpos = 100
    posidxs = randorder[:numpos]
    negidxs = randorder[numpos:]
    numneg = negidxs.shape[0]

    ytrue[posidxs] = 1.
    perfect_rank = np.argsort(-ytrue)
    bad_rank = np.argsort(ytrue)

    ## check score for a perfect result set is  1
    ## regardless of whether the rank is computed when k < numpos or k >> numpos
    assert np.isclose(ndcg_rank_score(ytrue,perfect_rank[:numpos//2]),1.)
    assert np.isclose(ndcg_rank_score(ytrue,perfect_rank[:numpos]),1.)
    assert np.isclose(ndcg_rank_score(ytrue,perfect_rank[:numpos*2]),1.)    
    
        ## check score for no results is 0    
    assert np.isclose(ndcg_rank_score(ytrue,bad_rank[:-numpos]),0.)

  
    ## check score for same amount of results worsens if they are shifted    
    gr = perfect_rank[:numpos//2]
    br = bad_rank[:numpos//2]
    rank1 = np.concatenate([gr,br])
    rank2 = np.concatenate([br,gr])
    assert ndcg_rank_score(ytrue, rank1) > .5, 'half of entries being relevant, but first half'
    assert ndcg_rank_score(ytrue, rank2) < .5

def test_score_rare():
    n = 10000 # num items
    randorder = np.random.permutation(n)

    ## check in a case with only few positives
    for numpos in [0,1,2]:
        ytrue = np.zeros_like(randorder)
        posidxs = randorder[:numpos]
        negidxs = randorder[numpos:]
        numneg = negidxs.shape[0]
    
        ytrue[posidxs] = 1.
        perfect_rank = np.argsort(-ytrue)
        bad_rank = np.argsort(ytrue)
    
        scores = []
        k = 200
        for i in range(k):
            test_rank = bad_rank[:k].copy()
            if len(posidxs) > 0:
                test_rank[i] = posidxs[0]
            sc = ndcg_rank_score(ytrue,test_rank)
            scores.append(sc)
        scores = np.array(scores)
        if numpos == 0:
            assert np.isclose(scores ,0).all()
        else:
            assert (scores > 0 ).all()

# test_score_sanity()
# test_score_rare()
def compute_metrics(*, hit_indices, batch_size, total_seen, total_positives, ndatabase, at_N):
    ndcg_score = ndcg_score_fn(hit_indices=hit_indices, total_frames_seen=total_seen, 
                      total_positives=total_positives, at_k_frames=at_N)

    ntotal=total_positives
  
    hpos_full = np.ones(total_positives)*np.inf
    hpos_full[:hit_indices.shape[0]] = hit_indices
    nseen = hpos_full + 1

    nfound = np.arange(total_positives) + 1
    
    precisions = nfound/nseen
    AP = np.mean(precisions)

    best_possible_seen = (np.arange(total_positives) + 1).astype('float')
    best_possible_seen[total_seen:] = np.inf # cannot get instances beyond what is seen
    best_precisions = best_possible_seen/nfound
    bestAP = np.mean(best_precisions)
    relAP = AP/bestAP

    ## a key metric: how long did it take to find the second thing after feedback...
    batch_no = nseen//batch_size
    nfirst_batch = batch_no[0]
    nfirst = nseen[0]

    if (batch_no > batch_no[0]).any():
      gtpos = np.where(batch_no > batch_no[0])[0]
      assert gtpos.shape[0] > 0
      first_after_feedback = gtpos[0]
      nfirst2second_batch = batch_no[first_after_feedback] - nfirst_batch
      nfirst2second = nseen[first_after_feedback] - nfirst
    else:
            # only one batch contains all positives (or maybe no positives at all)
      # this metric is not well defined.
      nfirst2second = np.nan
      nfirst2second_batch = np.nan


    # fbatch = batchno[0]
    # laterbatch = batchno > fbatch
  
    return dict(ntotal=ntotal, nfound=nfound, 
                ndcg_score=ndcg_score,
                ndatabase=ndatabase, abundance=ntotal/ndatabase,
                nframes=total_seen,
                AP=AP,
                bestAP=bestAP,
                relAP=relAP,
                nfirst = nseen[0], 
                nfirst_batch = nfirst_batch,
                nfirst2second = nfirst2second,
                nfirst2second_batch = nfirst2second_batch,
                reciprocal_rank=precisions[0])


def side_by_side_comparison(stats, *, baseline_variant, metric):
    v1 = stats
    metrics = list(set(['nfound', 'nfirst'] + [metric]))

    v1 = v1[['dataset', 'category', 'variant', 'ntotal', 'abundance'] + metrics]
    v2 = stats[stats.variant == baseline_variant]
    rename_dict = {}
    for m in metrics:
        rename_dict[metric] = f'base_{metric}'
        
    rename_dict['base_variant'] = baseline_variant
    
    v2 = v2[['dataset', 'category'] + metrics]
    v2['base'] = v2[metric]
    v2 = v2.rename(mapper=rename_dict, axis=1)
    
    sbs = v1.merge(v2, right_on=['dataset', 'category'], left_on=['dataset', 'category'], how='outer')
    sbs = sbs.assign(ratio=sbs[metric]/sbs['base'])
    sbs = sbs.assign(delta=sbs[metric] - sbs['base'])
    return sbs

def better_same_worse(stats, *, variant, baseline_variant, metric, reltol,
                     summary=True):
    sbs = side_by_side_comparison(stats, baseline_variant=baseline_variant, metric=metric)
    invtol = 1/reltol
    sbs = sbs.assign(better=sbs[metric] > reltol*sbs.base, worse=sbs[metric] < invtol*sbs.base,
                    same=sbs[metric].between(invtol*sbs.base, reltol*sbs.base))
    bsw = sbs[sbs.variant == variant].groupby('dataset')[['better', 'same', 'worse']].sum()
    if summary:
        return bsw
    else:
        return sbs

def bsw_table(stats, *, variant, baseline_variant, metric, reltol):
    bsw = better_same_worse(stats, metric=metric, baseline_variant=baseline_variant, variant=variant, reltol=reltol)
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
#display(tot_res.style.highlight_max(axis=0))
# print(res)
# print(tot_res.to_latex())
# tot_res = pd.concat(acc, axis=1)

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

from collections import defaultdict

def old_benchresults(resultlist):
    ans = defaultdict(lambda : [])
    for tup in resultlist:
        ans[tup[0]['dataset']].append(tup)
    return ans

def print_tables(stats, *, variant,  baseline_variant, metric, reltol):
    all_vars = stats.groupby(['dataset', 'category', 'variant',])[metric].mean().unstack(-1)
    means = all_vars.groupby('dataset').mean()
    counts = all_vars.groupby('dataset').size().rename('num_queries')
    per_dataset = pd.concat([means, counts],axis=1)
    print('by dataset')
    display(per_dataset)
    
    print('by query')
    bsw_table(stats, variant=variant, baseline_variant=baseline_variant, metric=metric, reltol=reltol)
    
    
    print('breakdown by initial res')
    sbs = better_same_worse(stats, variant=variant, baseline_variant=baseline_variant, reltol=reltol, metric=metric, summary=False)
    tot_res = summary_breakdown(sbs, metric=metric)
    comparison_table(tot_res, variant=variant, baseline_variant=baseline_variant)


    print('ablation')
    if variant == 'seesaw':
      abtab = ablation_table(tot_res, variants_list=['plain', 'multiplain','seesaw',])
      display(abtab)
    else:
      print('skipping ablation table bc. using other variant')

    x = np.geomspace(.02, 1, num=5)
    y = 1/x
    diag_df = pd.DataFrame({'x':x, 'y':y})

    ### plot

    plotdata = sbs[sbs.variant == variant]
    scatterplot = (ggplot(plotdata)
        + geom_point(aes(x='base', y='ratio', fill='dataset', color='dataset'), alpha=.6, size=1.) 
    #                 shape=plotdata.dataset.map(lambda x : '.' if x in ['lvis','objectnet'] else 'o'), 
    #                 size=plotdata.dataset.map(lambda x : 1. if x in ['lvis','objectnet'] else 2.))
    #  + geom_text(aes(x='base', y='delta', label='category', color='dataset'), va='bottom', 
    #              data=plotdata1[plotdata1.ratio < .6], 
    #              position=position_jitter(.05, .05), show_legend=False)
        + geom_line(aes(x='x', y='y'), data=diag_df)
     + ylab('VSL/baseline')
    #               + geom_area(aes(y2=1.1, y=.9), linetype='dashed', alpha=.7)
                   + geom_hline(aes(yintercept=1.1), linetype='dashed', alpha=.7)
                   + geom_hline(aes(yintercept=.9), linetype='dashed', alpha=.7)


                    + geom_vline(aes(xintercept=.1,), linetype='dashed', alpha=.7)
                    + geom_vline(aes(xintercept=.3,), linetype='dashed', alpha=.7)
    #+ geom_abline()
    #    + geom_point(aes(x='recall', y='precision', color='variant'), size=1.)
    #     + facet_wrap(facets=['cat'], ncol=6, scales='free_x')
     + xlab(f'baseline {metric}')
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

    return scatterplot

from tqdm.auto import tqdm

def latency_table(ldf, variant, evs2):
    #v = 'multiplain_warm_vec_fast'
    #v = 'multiplain_warm_vec_only'
    v = variant
    gps = ldf.groupby(['dataset', 'category','variant'])[['lookup', 'label', 'refine']].median()
    meds = gps.reset_index().groupby(['dataset', 'variant']).median().reset_index()

    lat = meds[meds.variant.isin([v])][['dataset','lookup', 'refine']].set_index('dataset')
    lat = lat.rename(mapper={'lookup':'latency_pyr_lookup', 'refine':'latency_vec_refine'},axis=1)
    latplain = meds[meds.variant.isin(['plain'])][['dataset','lookup']].set_index('dataset')
    lat = lat.assign(latency_plain_lookup=latplain['lookup'])
    lat = lat[['latency_plain_lookup', 'latency_pyr_lookup', 'latency_vec_refine']]
    
    dataset_info = []
    for (k,ev) in evs2.items():
        c = ev.query_ground_truth.columns.shape[0]
        ans = {'dataset':k, 'queries':c, 'images':ev.query_ground_truth.shape[0],
              'vectors':ev.fine_grained_embedding.shape[0]}
        dataset_info.append(ans)
        
    lat2 = pd.concat([lat, pd.DataFrame(dataset_info)[['dataset', 'images', 'vectors']].set_index('dataset')], axis=1)
    lat2['vecs/im'] = np.round(lat2.vectors/lat2.images)
    lat2 = lat2.sort_values('vectors')
    lat2 = lat2[['latency_plain_lookup', 'images', 'latency_pyr_lookup', 'vectors', 'latency_vec_refine', 'vecs/im']]
    print(lat2.astype('float').to_latex(float_format=remove_leading_zeros('{:.02f}'.format)))
    return lat2
