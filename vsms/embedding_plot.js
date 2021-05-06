
// function anonymous(db_cds,db_score,db_top,gt_cds,gt_source,lang_cds,mark_edges,segment_source,cb_obj,cb_data)
let clear_data = (ds) => {
    let cln = {};
    for (const k in ds){
        cln[k] = []
    }
    return cln;
}

if (lang_cds.selected.indices.length == 0){ // clear selection
    gt_source.data = clear_data(gt_source.data);
    segment_source.data = clear_data(segment_source.data);
    return;
}

let row_idx = lang_cds.selected.indices[0]
let query_str = lang_cds.data.query_str[row_idx]
let concept = lang_cds.data.concept[row_idx]

let find_true = (vals, fn) => {
    let paired = _.zipWith(vals,_.range(vals.length));
    return paired.filter(arr => fn(arr[0])).map(arr => arr[1])
}

let col = gt_cds.data[concept]
let selection = find_true(col,  x => x !== 0)

let mark_idxs = (db, view, dbidxs) => {
    let view_data = {'x':[], 'y':[]};
    for (var i = 0; i < dbidxs.length; i++){
        view_data['x'].push(db.data.x[dbidxs[i]]);
        view_data['y'].push(db.data.y[dbidxs[i]]);
    }
    view.data = view_data     
};
mark_idxs(db_cds, gt_source, selection);

let indices = db_top.data[query_str];

let mark_edgesf = (db_cds, lang_cds, segment_source,row_idx,indices) => {
    let new_neighbor_data = {'x0':[], 'y0':[], 'x1':[], 'y1':[], 'color':[], 'width':[], 'rank':[], 'score':[], 'image_url':[]};
    let x0 = lang_cds.data.x[row_idx];
    let y0 = lang_cds.data.y[row_idx];

    for (var i = 0; i < indices.length; i++){
        new_neighbor_data['x0'].push(x0);
        new_neighbor_data['y0'].push(y0);

        let idx = indices[i];
        new_neighbor_data['x1'].push(db_cds.data.x[idx]);
        new_neighbor_data['y1'].push(db_cds.data.y[idx]);

        let color = selection.includes(idx) ? 'green' : 'red'
        let width = 4. - i*(4./indices.length) + .5
        new_neighbor_data['color'].push(color);
        new_neighbor_data['width'].push(width);

        new_neighbor_data['rank'].push(i);
        new_neighbor_data['score'].push(db_score.data[query_str][i])
        new_neighbor_data['image_url'].push(db_cds.data.image_url[idx])
    }
    segment_source.data = new_neighbor_data;
}

if (mark_edges){
    mark_edgesf(db_cds, lang_cds, segment_source, row_idx, indices);
}
