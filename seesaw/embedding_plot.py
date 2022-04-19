import bokeh
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import (
    HoverTool,
    ColumnDataSource,
    CategoricalColorMapper,
    CategoricalScale,
    CustomJS,
    Button,
)
from bokeh.palettes import Spectral10, Magma256, Category20
from bokeh.transform import jitter
from bokeh.layouts import column
import numpy as np
import pkgutil
import umap
import hdbscan
import sklearn.preprocessing
import pandas as pd
from tqdm.auto import tqdm


class Visualizer(object):
    def __init__(self, ev):
        self.ev = ev
        self.reducer = None
        self.projected_images = None
        self.projected_text = None
        self.df_projected_images = pd.DataFrame({"cluster_id": np.ones(len(ev))})


def fit_projection(self):
    reducer = umap.UMAP(metric="cosine")
    ev = self.ev
    db = ev.db
    qgt = ev.query_ground_truth
    self.reducer = reducer
    queries = pd.DataFrame({"query_str": qgt.columns, "concept": qgt.columns})
    self.queries = queries

    qvecs = np.stack([ev.embedding.from_raw(c) for c in queries.query_str]).squeeze()
    train_vecs = sklearn.preprocessing.normalize(np.concatenate([db.embedded, qvecs]))
    self.query_vectors = sklearn.preprocessing.normalize(qvecs)
    self.reducer.fit(train_vecs)

    self.projected_images = reducer.transform(
        sklearn.preprocessing.normalize(db.embedded)
    )
    self.projected_text = reducer.transform(self.query_vectors)

    db_df = pd.DataFrame(self.projected_images, columns=("x", "y"))
    self.df_projected_images = db_df.assign(
        point_id=np.arange(len(self.ev)).astype("int"), image_url=ev.db.urls
    )
    self.df_projected_text = queries.assign(
        x=self.projected_text[:, 0], y=self.projected_text[:, 1]
    )


def assign_clusters(self):
    hdb = hdbscan.HDBSCAN(min_cluster_size=40)
    hdb.fit(self.projected_images)
    self.hdb = hdb
    self.df_projected_images = self.df_projected_images.assign(
        cluster_id=[i for i in hdb.labels_]
    )


def assign_top_scores(self, topk):
    assert isinstance(topk, int)
    topvecs = {}
    topscores = {}
    for (qstr, vc) in tqdm(list(zip(self.queries.query_str, self.query_vectors))):
        a, b = self.ev.db.query(
            topk=topk, mode="nearest", vector=vc.reshape(1, -1), return_scores=True
        )
        topvecs[qstr] = a
        topscores[qstr] = b

    self.query_topk_index = pd.DataFrame(topvecs)
    self.query_topk_score = pd.DataFrame(topscores)


def make_db_plot(vz, show_text=False):
    ev = vz.ev
    db = ev.db
    db_df = vz.df_projected_images
    qdf = vz.df_projected_text
    qvecs = vz.query_vectors
    qgt = ev.query_ground_truth

    lang_cds = ColumnDataSource(qdf)  # pd.concat(catdf, embed_df], ignore_index=True)
    db_df = db_df.assign(cluster_id=db_df.cluster_id.map(lambda x: str(x)))
    db_cds = ColumnDataSource(db_df)
    gt_cds = ColumnDataSource(qgt)

    plot_figure = figure(
        title="embedding plot",
        plot_width=900,
        plot_height=900,
        tools=("pan, wheel_zoom, reset, tap"),
        output_backend="webgl",
    )

    cids = db_df.cluster_id.unique()
    cids = (["-1"] if "-1" in cids else []) + [
        x for x in np.random.permutation(cids) if x != "-1"
    ]
    colors = Magma256[:: len(Magma256) // len(cids)][: len(cids)]
    color_mapping = CategoricalColorMapper(factors=cids, palette=colors)

    db_scatter = plot_figure.circle(
        x="x",
        y="y",
        radius=0.01,
        radius_units="data",
        fill_alpha=0.5,
        line_alpha=0.5,
        color=dict(field="cluster_id", transform=color_mapping),
        source=db_cds,
    )

    plot_figure.add_tools(
        HoverTool(
            tooltips="""
    <div style="border: 1px solid;border-radius: 5px;padding:2px;margin:2px">
        <div>
            <img src="@image_url" style='float: center; margin: 5px 5px 5px 5px; padding:1px; width:100%; height:100%; object-fit: contain'/>
        </div>
        <div>
            <span style='font-size: 16px; color: #224499'>Id: @point_id</span>
            <span style='font-size: 16px; color: #224499'>Cluster: @cluster_id</span>
        </div>
    </div>
    """,
            renderers=[db_scatter],
        )
    )

    if show_text:
        text_scatter = plot_figure.text(
            x=jitter("x", 0.1),
            y=jitter("y", 0.1),
            text={"field": "query_str"},
            selection_text_alpha=1.0,
            nonselection_text_alpha=0.5,
            source=lang_cds,
            text_font_size={"value": "14px"},
        )

        plot_figure.add_tools(
            HoverTool(
                tooltips="""
        <div style="border: 1px solid;border-radius: 5px;padding:1px;margin:2px">
            <div>
                <span style='font-size: 16px; color: #224499'>Query: "@query_str"</span>
            </div>
            <div>
                <span style='font-size: 16px; color: #224499'>Label: @concept</span>
            </div>
        </div>
        """,
                renderers=[text_scatter],
            )
        )

    ## placeholder data sources for selections: ground truth vs. nearest
    segment_source = ColumnDataSource(
        {
            "x0": [],
            "y0": [],
            "x1": [],
            "y1": [],
            "color": [],
            "width": [],
            "rank": [],
            "image_url": [],
            "score": [],
        }
    )
    neighbor_segment = plot_figure.segment(
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        color={"field": "color"},
        alpha=0.5,
        line_width={"field": "width"},
        source=segment_source,
    )

    plot_figure.add_tools(
        HoverTool(
            tooltips="""
    <div>
        <div>
            <img src="@image_url" style='float: left; margin: 5px 5px 5px 5px'/>
        </div><div>
            <span style='font-size: 16px; color: #224499'>rank: @rank</span>
            <span style='font-size: 16px; color: #224499'>score: @score</span>
        </div>
    </div>
    """,
            renderers=[neighbor_segment],
        )
    )

    gt_source = ColumnDataSource({"x": [], "y": []})
    plot_figure.scatter(
        x="x",
        y="y",
        color="black",
        size=10,
        alpha=0.9,
        fill_alpha=0.0,
        marker=dict(value="cross"),
        source=gt_source,
    )

    embedding_plot_js = pkgutil.get_data(__name__, "embedding_plot.js").decode("utf-8")
    if show_text:
        db_top = ColumnDataSource(vz.query_topk_index)
        db_score = ColumnDataSource(vz.query_topk_score)
        topk = vz.query_topk_index.shape[0]

        select_cb = CustomJS(
            args=dict(
                lang_cds=lang_cds,
                db_cds=db_cds,
                gt_cds=gt_cds,
                db_top=db_top,
                db_score=db_score,
                gt_source=gt_source,
                segment_source=segment_source,
                mark_edges=(topk is not None),
            ),
            code=embedding_plot_js,
        )

        lang_cds.selected.js_on_change("indices", select_cb)

    return plot_figure


def animate_trace(vz, fig_fun, sample_trace, ground_truth):
    plot_figure = fig_fun()
    button = Button(label="next iter", button_type="success")
    search_cds = ColumnDataSource(
        {"x": [], "y": [], "color": [], "alpha": [], "width": [], "batch": []}
    )
    db_scatter = plot_figure.scatter(
        x="x",
        y="y",
        color=dict(field="color"),
        size=10,
        line_width=dict(field="width"),
        alpha=dict(field="alpha"),
        fill_alpha=0.0,
        marker=dict(value="circle"),
        source=search_cds,
    )
    plot_figure.add_tools(
        HoverTool(
            tooltips="""
    <div>
        <span style='font-size: 16px; color: #224499'>batch: @batch</span>
    </div>
    """,
            renderers=[db_scatter],
        )
    )

    gts = {"x": [], "y": []}
    gt_source = ColumnDataSource(gts)
    positives = np.where(ground_truth)[0]
    for idx in positives:
        gts["x"].append(vz.df_projected_images["x"][idx])
        gts["y"].append(vz.df_projected_images["y"][idx])

    plot_figure.scatter(
        x="x",
        y="y",
        color="black",
        size=10,
        alpha=0.9,
        fill_alpha=0.0,
        marker=dict(value="cross"),
        source=gt_source,
    )

    button.js_on_click(
        CustomJS(
            args=dict(
                search_cds=search_cds,
                db_df=ColumnDataSource(vz.df_projected_images),
                gt_source=gt_source,
                iter=[0],
                ground_truth=ground_truth,
                selection=np.where(ground_truth > 0)[0],
                sample_trace=sample_trace,
                interval_ms=1000,
            ),
            code="""
                                    // clear previous data
                                    //for (const k in search_cds.data){
                                    //    search_cds.data[k] = [];
                                    //}

                                    let mark_idxs = (db, view, dbidxs) => {
                                        let view_data = {'x':[], 'y':[]};
                                        for (var i = 0; i < dbidxs.length; i++){
                                            view_data['x'].push(db.data.x[dbidxs[i]]);
                                            view_data['y'].push(db.data.y[dbidxs[i]]);
                                        }
                                        view.data = view_data     
                                    };

                                    let paint = (iter) => {
                                        if (iter >= sample_trace.length){
                                            return;
                                        }

                                        let arr = sample_trace[iter];
                                        let new_search_cds = _.cloneDeep(search_cds.data);

                                        // first update alpha for previous stuff
                                        for (var i = 0; i < new_search_cds.x.length; i++){
                                            new_search_cds.alpha[i] = .7
                                            new_search_cds.width[i] = 1.
                                        }

                                        for (var i = 0; i < arr.length; i++){
                                            new_search_cds.x.push(db_df.data.x[arr[i]]);
                                            new_search_cds.y.push(db_df.data.y[arr[i]]);
                                            new_search_cds.alpha.push(1.);
                                            new_search_cds.batch.push(iter);
                                            new_search_cds.width.push(2.);


                                            let color = ground_truth[arr[i]] > 0 ? 'green' : 'red';
                                            new_search_cds.color.push(color);
                                        }

                                        search_cds.data = new_search_cds;
                                        //setTimeout(() => paint(iter+1), interval_ms);
                                    }

                                    paint(iter[0]);
                                    iter[0] += 1;
                                """,
        )
    )

    return column(button, plot_figure)
