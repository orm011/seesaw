import pandas as pd
import numpy as np
import pyroaring as pr
import math

from collections import defaultdict

from IPython.display import display
from plotnine import (
    ggplot,
    geom_point,
    geom_jitter,
    geom_abline,
    scale_y_log10,
    scale_x_log10,
    geom_hline,
    aes,
    theme,
    geom_line,
    geom_vline,
    element_text,
    element_blank,
    xlab,
    ylab,
    geom_text,
)


import bokeh


def brief_formatter(num_sd):
    assert num_sd > 0

    def formatter(ftpt):
        if math.isclose(ftpt, 0.0):
            return "0"

        if math.isclose(ftpt, 1.0):
            return "1"

        if ftpt < 1.0:
            exp = -math.floor(math.log10(abs(ftpt)))
            fmt_string = "{:.0%df}" % (exp + num_sd - 1)
            dec = fmt_string.format(ftpt)
        else:
            fmt_string = "{:.02f}"
            dec = fmt_string.format(ftpt)

        zstripped = dec.lstrip("0").rstrip("0")
        return zstripped.rstrip(".")

    return formatter


brief_format = brief_formatter(1)


_higher_is_better = [
    "ndcg_score",
    "AP",
    "nAP",
    "reciprocal_rank_first",
    "reciprocal_rank_last",
]
_lower_is_better = ["rank_first", "rank_last"]


def bsw_table2(compare, *, metric, reltol):
    invtol = 1 / reltol

    if metric in _higher_is_better:
        higher_is_better = True
    elif metric in _lower_is_better:
        higher_is_better = False
    else:
        assert False, "need to specify if higher is better for metric"

    metric_col = compare[metric]
    baseline_col = compare[f"{metric}_baseline"]

    higher = (baseline_col * reltol) < metric_col
    lower = metric_col < (baseline_col * invtol)
    similar = ((baseline_col * invtol) <= metric_col) & (
        metric_col <= (baseline_col * reltol)
    )

    neither = ~(higher | lower | similar)

    if higher_is_better:
        better = higher
        worse = lower
    else:
        better = lower
        worse = higher

    compare = compare.assign(better=better, same=similar, worse=worse, neither=neither)
    bydataset = compare.groupby("dataset")[["better", "same", "worse", "neither"]].sum()
    dstot = bydataset.sum(axis=1)
    bsw = bydataset.assign(total=dstot)
    tot = bsw.sum()
    bsw_table = bsw.transpose().assign(total=tot).transpose()
    return bsw_table


def times_format(ftpt):
    return brief_format(ftpt) + "x"


def make_labeler(fmt_func):
    def fun(arrlike):
        return list(map(fmt_func, arrlike))

    return fun


def side_by_side_comparison(stats, *, baseline_variant, metric):
    stats = stats.assign(session_index=stats.index.values)
    v1 = stats
    metrics = list(set(["nfound", "reciprocal_rank_first"] + [metric]))

    v1 = v1[["session_index", "dataset", "category", "variant", "ntotal"] + metrics]
    v2 = stats[stats.variant == baseline_variant]
    rename_dict = {}
    for m in metrics + ["variant", "session_index"]:
        rename_dict[m] = f"base_{m}"

    v2 = v2[["dataset", "category", "variant", "session_index"] + metrics]
    v2["base"] = v2[metric]
    v2 = v2.rename(mapper=rename_dict, axis=1)
    assert "variant" not in v2.columns.values
    assert "base_variant" in v2.columns.values

    sbs = v1.merge(
        v2,
        right_on=["dataset", "category"],
        left_on=["dataset", "category"],
        how="outer",
    )
    sbs = sbs.assign(ratio=sbs[metric] / sbs["base"])
    sbs = sbs.assign(delta=sbs[metric] - sbs["base"])
    return sbs


def bsw_table(sbs, *, variant, metric, reltol):
    invtol = 1 / reltol
    sbs = sbs.assign(
        better=sbs[metric] > reltol * sbs.base,
        worse=sbs[metric] < invtol * sbs.base,
        same=sbs[metric].between(invtol * sbs.base, reltol * sbs.base),
    )
    bsw = (
        sbs[sbs.variant == variant]
        .groupby("dataset")[["better", "same", "worse"]]
        .sum()
    )
    dstot = bsw.sum(axis=1)
    bsw = bsw.assign(total=dstot)
    tot = bsw.sum()
    bsw_table = bsw.transpose().assign(total=tot).transpose()
    return bsw_table


def summary_breakdown(sbs, metric):
    base_metric = "base"

    # any variant name actually in the data should work for counts later on
    example_variant = sbs.variant.iloc[0]
    part = sbs[base_metric].map(
        lambda x: "1." if x > 0.3 else ".3" if x > 0.1 else ".1"
    )
    sbs = sbs.assign(part=part)
    totals = (
        sbs.groupby(["part", "dataset", "variant"])[metric]
        .mean()
        .reset_index()
        .groupby(["part", "variant"])[metric]
        .mean()
        .unstack(level=0)
    )
    counts = (
        sbs.groupby(["part", "dataset", "variant"])
        .size()
        .rename("cats")
        .reset_index()
        .groupby(["part", "variant"])
        .cats.sum()
        .unstack(level=0)
    )

    tr = totals.transpose()

    count_col = counts.transpose()[example_variant]

    tr = tr.assign(counts=count_col)[
        ["counts"] + tr.columns.values.tolist()
    ].transpose()
    return tr


def remove_leading_zeros(fmtr):
    def fun(x):
        fx = fmtr(x).lstrip("0")

        if x >= 1:
            fx = fx.rstrip("0")

        fx = fx.rstrip(".")
        return fx

    return fun


def comparison_table(tot_res, *, variant, baseline_variant):
    sys = variant
    baseline = baseline_variant
    tot_res = tot_res.transpose()
    tot_res = tot_res.assign(ratio=tot_res[sys] / tot_res[baseline])
    total_counts = tot_res["counts"].sum()
    tot_res = tot_res[["counts", baseline, sys, "ratio"]].rename(
        mapper={baseline: "baseline", sys: "this work"}, axis=1
    )
    tot_res = tot_res.transpose()
    return tot_res


def ablation_table(tot_res, variants_list):
    baseline = variants_list[0]
    inter = variants_list[1]
    sys = variants_list[2]
    # tot_res = tot_res.loc[]
    tot_res = (
        tot_res.transpose()[[baseline, inter, sys]]
        .rename(
            mapper={
                baseline: "semantic embeddding",
                inter: "+ multiscale search",
                sys: "+ feedback fusion",
            },
            axis=1,
        )
        .transpose()
    )

    # display()#.#style.highlight_max(axis=0))
    # print(res)
    # print(tot_res.diff().to_latex(float_format=brief_format))
    delta_ablation = tot_res.diff().iloc[1:]
    delta_ablation = delta_ablation.rename(mapper=lambda x: x + " delta", axis=1)
    ablation = tot_res  # .rename(mapper=lambda x : str(x), axis=1)
    ablation = pd.concat([ablation, delta_ablation], axis=1)[
        [".1", ".1 delta", ".3", ".3 delta", "1.", "1. delta"]
    ]

    with pd.option_context(
        "display.float_format", remove_leading_zeros("{:.02f}".format)
    ):
        display(ablation)
    print(ablation.to_latex(float_format=remove_leading_zeros("{:.02f}".format)))

    return ablation


def print_tables(
    stats,
    *,
    variant,
    baseline_variant,
    metric,
    reltol,
    intermediate_variant=None,
    brief=True,
    jitter=0.01,
):
    if intermediate_variant is None:
        intermediate_variant = baseline_variant

    res = {}
    all_vars = (
        stats.groupby(
            [
                "dataset",
                "category",
                "variant",
            ]
        )[metric]
        .mean()
        .unstack(-1)
    )
    means = all_vars.groupby("dataset").mean()
    counts = all_vars.groupby("dataset").size().rename("num_queries")
    per_dataset = pd.concat([means, counts], axis=1)
    if not brief:
        print("by dataset")
        display(per_dataset)

    res["by_dataset"] = per_dataset
    sbs = side_by_side_comparison(
        stats, baseline_variant=baseline_variant, metric=metric
    )
    res["sbs"] = sbs
    assert "variant" in sbs.columns.values

    print("bsw")
    bsw = bsw_table(sbs, variant=variant, metric=metric, reltol=reltol)
    display(bsw)

    if not brief:
        print(bsw.to_latex())

    res["bsw"] = bsw

    tot_res = summary_breakdown(sbs, metric=metric)
    cmp = comparison_table(tot_res, variant=variant, baseline_variant=baseline_variant)

    if not brief:
        fmtter = remove_leading_zeros("{:.02f}".format)
        with pd.option_context("display.float_format", fmtter):
            display(tot_res)
        print(tot_res.to_latex(float_format=fmtter))
    res["comparison"] = cmp

    if not brief:
        print("ablation")
        abtab = ablation_table(
            tot_res, variants_list=[baseline_variant, intermediate_variant, variant]
        )
        display(abtab)
        res["ablation"] = abtab

    plot = rel_plot(sbs, variant, jitter=jitter)
    display(plot)

    res["plotdata"] = plot.data

    return res


def compare_stats(stats, variant, baseline_variant):
    valid_variants = stats.variant.unique()
    assert variant in valid_variants
    assert baseline_variant in valid_variants
    all_pairs = pd.merge(
        stats[stats.variant == variant],
        stats[stats.variant == baseline_variant],
        left_on=["dataset", "ground_truth_category"],
        right_on=["dataset", "ground_truth_category"],
        suffixes=["", "_baseline"],
    )
    return all_pairs


def rel_plot(sbs, variant, jitter=0.01):
    plotdata = sbs[sbs.variant == variant]
    xcol = "base"
    ycol = "ratio"
    plotdata = plotdata.assign(x=plotdata[xcol], y=plotdata[ycol])
    plotdata = plotdata.assign(sbs_index=plotdata.index.values)
    session_text = (
        plotdata[["session_index", "base_session_index"]]
        .apply(tuple, axis=1)
        .map(lambda tup: f"{tup[0]} vs. {tup[1]}")
    )
    plotdata = plotdata.assign(session_text=session_text)

    x = np.geomspace(0.02, 1, num=5)
    y = 1 / x
    diag_df = pd.DataFrame({"x": x, "y": y})

    scatterplot = (
        ggplot(plotdata)
        + geom_jitter(
            aes(x="x", y="y", fill="dataset", color="dataset"),
            width=jitter,
            height=jitter,
            alpha=0.6,
            size=1.0,
        )
        #                 shape=plotdata.dataset.map(lambda x : '.' if x in ['lvis','objectnet'] else 'o'),
        #                 size=plotdata.dataset.map(lambda x : 1. if x in ['lvis','objectnet'] else 2.))
        #  + geom_text(aes(x='base', y='delta', label='category', color='dataset'), va='bottom',
        #              data=plotdata1[plotdata1.ratio < .6],
        #              position=position_jitter(.05, .05), show_legend=False)
        + geom_line(aes(x="x", y="y"), data=diag_df)
        # + geom_text(aes(x='x', y='y', label='session_text'), va='top', data=plotdata[(plotdata.y < .4) | (plotdata.y > 3)])
        + ylab(ycol)
        #               + geom_area(aes(y2=1.1, y=.9), linetype='dashed', alpha=.7)
        + geom_hline(aes(yintercept=1.1), linetype="dashed", alpha=0.7)
        + geom_hline(aes(yintercept=0.9), linetype="dashed", alpha=0.7)
        + geom_vline(
            aes(
                xintercept=0.1,
            ),
            linetype="dashed",
            alpha=0.7,
        )
        + geom_vline(
            aes(
                xintercept=0.3,
            ),
            linetype="dashed",
            alpha=0.7,
        )
        # + geom_abline()
        #    + geom_point(aes(x='recall', y='precision', color='variant'), size=1.)
        #     + facet_wrap(facets=['cat'], ncol=6, scales='free_x')
        + xlab(xcol)
        # +scale_color_discrete()
        + theme(
            figure_size=(8, 5),
            legend_position="top",
            subplots_adjust={"hspace": 0.5},
            legend_title=element_blank(),
            legend_box_margin=-1,
            legend_margin=0.0,
            axis_text=element_text(size=12, margin={"t": 0.2, "l": -0.3}),
            legend_text=element_text(size=11),
            axis_title=element_text(
                size=12, margin={"r": -0.2, "b": 0.0, "l": 0, "t": 0.0}
            ),
        )
        + scale_x_log10(labels=make_labeler(brief_format), breaks=[0.01, 0.1, 0.3, 1.0])
        + scale_y_log10(
            labels=make_labeler(brief_format), breaks=[0.5, 0.9, 1.1, 2.0, 3.0, 6, 12]
        )
    )

    return scatterplot


def plot_compare(
    stats, variant, variant_baseline, metric, mode="identity", jitter=0.01
):
    assert mode in ["identity", "ratio", "difference"]
    plotdata = compare_stats(stats, variant, variant_baseline)
    bsw = bsw_table2(plotdata, metric=metric, reltol=1.0)
    display(bsw)
    baseline_name = f"{metric}_baseline"
    plotdata = plotdata[[metric, baseline_name, "dataset"]].assign(
        ratio=plotdata[metric] / plotdata[baseline_name],
        difference=plotdata[metric] - plotdata[baseline_name],
    )

    if mode == "identity":
        return (
            ggplot(data=plotdata)
            + geom_jitter(
                aes(x=f"{metric}_baseline", y=metric, fill="dataset"),
                width=jitter,
                height=jitter,
            )
            + scale_x_log10()
            + scale_y_log10()
            + geom_abline(aes(slope=1, intercept=0))
        )
    elif mode == "ratio":
        return (
            ggplot(data=plotdata)
            + geom_jitter(
                aes(x=f"{metric}_baseline", y="ratio", fill="dataset"),
                width=jitter,
                height=jitter,
            )
            + scale_x_log10()
            + scale_y_log10()
            ## ablines are drawn wrt the already log-transformed axes. hence 0 = log(1) in scale
            + geom_abline(aes(slope=0, intercept=0.0))
            + geom_abline(aes(slope=-1, intercept=0.0))  # max
        )
    elif mode == "difference":
        return (
            ggplot(data=plotdata)
            + geom_jitter(
                aes(x=f"{metric}_baseline", y="difference", fill="dataset"),
                width=jitter,
                height=jitter,
            )
            + scale_x_log10()
            + scale_y_log10()
            + geom_abline(aes(slope=0, intercept=0))
        )
    else:
        assert False, "unknown mode"


from bokeh.models import (
    HoverTool,
    TapTool,
    OpenURL,
    BoxZoomTool,
    ResetTool,
    PanTool,
    WheelZoomTool,
)
from bokeh.palettes import d3
from bokeh.transform import factor_cmap, jitter
from bokeh.plotting import figure, show, ColumnDataSource, output_notebook


def make_color_map(df, column_name):
    factors = df[column_name].unique()
    total = max(len(factors), 3)
    palette = d3["Category10"][total]
    return factor_cmap(column_name, palette=palette, factors=factors)


def interactive_compare(
    stats,
    variant,
    variant_baseline,
    metric,
    tooltip_cols=["dataset", "category", "ntotal", "nimages"],
    metric_cols=["rank_first", "rank_last", "nfound"],
    mode="identity",
    jitter_size=0.001,
):
    plotdata = compare_stats(stats, variant, variant_baseline)

    baseline_name = f"{metric}_baseline"
    plotdata = plotdata.assign(
        ratio=plotdata[metric] / plotdata[baseline_name],
        difference=plotdata[metric] - plotdata[baseline_name],
    )

    assert mode in ["identity", "ratio"]

    output_notebook()

    base_metric = f"{metric}_baseline"

    if metric not in metric_cols:
        metric_cols = [metric] + metric_cols

    tooltips = []
    tooltips.append(("(x,y)", "($x{.02f},$y{.02f})"))
    tooltips.extend([(m, f"@{m}") for m in tooltip_cols])
    tooltips.extend(
        [(metric, f"(@{metric}_baseline, @{metric})") for metric in metric_cols]
    )

    p = figure(
        title="comparison",
        y_axis_type="log",
        x_axis_type="log",
        plot_width=500,
        plot_height=500,
        tools=[HoverTool(), PanTool(), WheelZoomTool(), ResetTool()],
        tooltips=tooltips,
        background_fill_color="#fafafa",
    )

    source = ColumnDataSource(plotdata)

    if mode == "identity":
        p.circle(
            x=jitter(base_metric, width=jitter_size),
            y=jitter(metric, width=jitter_size),
            size=10,
            fill_color=make_color_map(plotdata, "dataset"),
            line_color="black",
            source=source,
        )

        p.match_aspect = True
        minval = plotdata[base_metric].min()
        maxval = (plotdata[base_metric][(plotdata[base_metric] < np.inf)]).max()
        p.segment(x0=minval, x1=maxval, y0=0, y1=maxval)
    elif mode == "ratio":
        p.circle(
            x=jitter(base_metric, width=jitter_size),
            y=jitter("ratio", width=jitter_size),
            size=10,
            fill_color=make_color_map(plotdata, "dataset"),
            line_color="black",
            source=source,
        )

        minval = 0.001  # plotdata[base_metric].min()
        maxval = 1.0  # (plotdata[base_metric][(plotdata[base_metric] < np.inf)]).max()
        p.segment(x0=minval, x1=1.0, y0=1.0, y1=1.0)
        p.segment(x0=minval, x1=1.0, y0=1.0 / minval, y1=1.0)
    else:
        assert False

    def url_tool(url_column1, url_column2):
        url = f"http://localhost:9000/compare?path=@{url_column1}&other=@{url_column2}"
        taptool = TapTool()
        taptool.callback = OpenURL(url=url)
        return taptool

    p.add_tools(url_tool("session_path", "session_path_baseline"))
    show(p)


def interactive_scatterplot(
    pdata, tooltip_cols=["x", "y", "dataset", "category", "frequency"]
):
    output_notebook()

    p = figure(
        title="comparison",
        y_axis_type="log",
        x_axis_type="log",
        plot_width=800,
        plot_height=500,
        tools=[HoverTool(), PanTool(), BoxZoomTool(), ResetTool()],
        tooltips=", ".join(["@{}".format(col) for col in tooltip_cols]),
        #                "@x, @y, @dataset, @category, @query_string, @frequency",
        background_fill_color="#fafafa",
    )

    source = ColumnDataSource(pdata)

    p.circle(
        x=jitter("x", width=0.1),
        y=jitter("y", width=0.1),
        size=10,
        fill_color=make_color_map(pdata, "dataset"),
        line_color="black",
        source=source,
    )

    def url_tool(url_column1, url_column2):
        url = f"http://localhost:9000/compare?path=@{url_column1}&other=@{url_column2}"
        taptool = TapTool()
        taptool.callback = OpenURL(url=url)
        return taptool

    p.add_tools(url_tool("session_path", "base_session_path"))
    show(p)
