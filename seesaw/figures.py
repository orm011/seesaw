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



def compute_metrics(results, indices, gt):
    hits = results
    ndcg_score = ndcg_rank_score(gt.values, indices)
    ndatabase=gt.shape[0]
    ntotal=gt.values.sum()
    
    hpos= np.where(hits > 0)[0] # > 0 bc. nan.
    if hpos.shape[0] > 0:
        nfirst = hpos.min() + 1
        rr = 1./nfirst
    else:
        nfirst = np.inf
        rr = 1/nfirst
        
    nfound = (hits > 0).cumsum()
    nframes = np.arange(hits.shape[0]) + 1
    precision = nfound/nframes
    recall = nfound/ntotal
    total_prec = (precision*hits).cumsum() # see Evaluation of ranked retrieval results page 158
    average_precision = total_prec/ntotal
    
    return dict(ndcg_score=ndcg_score, ntotal=ntotal, nfound=nfound[-1], 
                ndatabase=ndatabase, abundance=ntotal/ndatabase, nframes=hits.shape[0],
                nfirst = nfirst, reciprocal_rank=rr,
               precision=precision[-1], recall=recall[-1], average_precision=average_precision[-1])


def process_tups(evs, benchresults, keys, at_N):
    tups = []
    ## lvis: qgt has 0,1 and nan. 0 are confirmed negative. 1 are confirmed positive.
    for k in keys:#['lvis','dota','objectnet', 'coco', 'bdd' ]:#,'bdd','ava', 'coco']:
        val = benchresults[k]
        ev = evs[k]
    #     qgt = benchgt[k]
        for tup,exp in val:
            if exp is None:
                continue
            hits = np.concatenate(exp['results'])
            indices = np.concatenate(exp['indices'])
            index_set = pr.BitMap(indices)
            assert len(index_set) == indices.shape[0], 'check no repeated indices'
            #ranks = np.concatenate(exp['ranks'])
            #non_nan = ~np.isnan(ranks)
            gtfull = ev.query_ground_truth[tup['category']]

            isna = gtfull.isna()
            if isna.any():
                gt = gtfull[~isna]
                idx_d = dict(zip(gt.index,np.arange(gt.shape[0]).astype('int')))
                idx_map = lambda x : idx_d[x]
            else:
                gt = gtfull
                idx_map = lambda x : x

            local_idx = np.array([idx_map(idx) for idx in indices])
            metrics = compute_metrics(hits[:at_N], local_idx[:at_N], gt)
            output_tup = {**tup, **metrics}
            #output_tup['unique_ranks'] = np.unique(ranks[non_nan]).shape[0]
            output_tup['dataset'] = k
            tups.append(output_tup)
            
    return pd.DataFrame(tups)

def side_by_side_comparison(stats, baseline_variant, metric):
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

def better_same_worse(stats, variant, baseline_variant='plain', metric='ndcg_score', reltol=1.1,
                     summary=True):
    sbs = side_by_side_comparison(stats, baseline_variant='plain', metric='ndcg_score')
    invtol = 1/reltol
    sbs = sbs.assign(better=sbs.ndcg_score > reltol*sbs.base, worse=sbs.ndcg_score < invtol*sbs.base,
                    same=sbs.ndcg_score.between(invtol*sbs.base, reltol*sbs.base))
    bsw = sbs[sbs.variant == variant].groupby('dataset')[['better', 'same', 'worse']].sum()
    if summary:
        return bsw
    else:
        return sbs

def bsw_table(stats, variant, reltol=1.1):
    bsw = better_same_worse(stats, variant=variant, reltol=reltol)
    dstot = bsw.sum(axis=1)
    bsw = bsw.assign(total=dstot)
    tot = bsw.sum()
    bsw_table = bsw.transpose().assign(total=tot).transpose()
    display(bsw_table)
    print(bsw_table.to_latex())
    return bsw_table
    
def summary_breakdown(sbs):
    base_metric = 'base'
    part = sbs[base_metric].map(lambda x : '1.' if x > .3 else '.3' if x > .1 else '.1')
    sbs = sbs.assign(part=part)
    totals = sbs.groupby(['part', 'dataset', 'variant']).ndcg_score.mean().reset_index().groupby(['part', 'variant']).ndcg_score.mean().unstack(level=0)
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

def comparison_table(tot_res, variant):
    baseline = 'plain'
    sys = variant
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

def ablation_table(tot_res, variant):
    baseline = 'plain'
    inter = 'multiplain'
    sys = variant
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

def print_tables(evs2, variant, resultlist, at_N):
    benchresults = old_benchresults(resultlist)
    stats = process_tups(evs=evs2, keys=evs2.keys(), benchresults=benchresults, at_N=at_N)
    all_vars = stats.groupby(['dataset', 'category', 'variant',]).ndcg_score.mean().unstack(-1)
    means = all_vars.groupby('dataset').mean()
    counts = all_vars.groupby('dataset').size().rename('num_queries')
    per_dataset = pd.concat([means, counts],axis=1)
    print('by dataset')
    display(per_dataset)
    
    print('by query')
    bsw_table(stats, variant=variant, reltol=1.1)
    
    
    print('breakdown by initial res')
    sbs = better_same_worse(stats, variant=variant, summary=False)
    tot_res = summary_breakdown(sbs)
    comparison_table(tot_res, variant=variant)


    print('ablation')
    display(ablation_table(tot_res, variant=variant))

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
     + xlab('baseline NDCG')
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

def latency_profile(evs2, results, variant):
    #v = 'multiplain_warm_vec_fast'
    #v = 'multiplain_warm_vec_only'
    v = variant
    dfs = []
    for r in tqdm(results):
        tup = r[0]
        res = r[1]
        df = pd.DataFrame.from_records(res['latency_profile'])
        df = df.assign(dataset=tup['dataset'], variant=tup['variant'], category=tup['category'])
        dfs.append(df)
    ldf = pd.concat(dfs, ignore_index=True)
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
    print(lat2.astype('float').to_latex(float_format=remove_leading_zeros('{:.02f}'.format)))
    return lat2
