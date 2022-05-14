from genericpath import exists
import numpy as np
from ray.data.extensions import TensorArray
import os
from ...definitions import resolve_path
import shutil
import ray.data
from ...services import get_parquet


def infer_coarse_embedding(pdtab):
    # max_zoom_out = pdtab.groupby('file_path').zoom_level.max().rename('max_zoom_level')
    # wmax = pd.merge(pdtab, max_zoom_out, left_on='file_path', right_index=True)
    wmax = pdtab
    lev1 = wmax[wmax.zoom_level == wmax.max_zoom_level]
    ser = lev1.groupby("dbidx").vectors.mean().reset_index()
    res = ser["vectors"].values.to_numpy()
    normres = res / np.maximum(np.linalg.norm(res, axis=1, keepdims=True), 1e-6)
    return ser.assign(vectors=TensorArray(normres))


def from_fine_grained(fine_grained_path, output_path):
    fine_grained_path = resolve_path(fine_grained_path)
    output_path = resolve_path(output_path)
    assert os.path.isdir(fine_grained_path)
    assert not os.path.exists(output_path)

    outdir, outname = os.path.dirname(output_path), os.path.basename(output_path)
    final_output_path = f"{outdir}/{outname}"
    output_path = f"{outdir}/.tmp.{outname}"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    os.makedirs(output_path)
    ## copy images and model links
    shutil.copy2(
        fine_grained_path + "/model", output_path + "/model", follow_symlinks=False
    )

    shutil.copy2(
        fine_grained_path + "/dataset", output_path + "/dataset", follow_symlinks=False
    )

    vector_path = f"{output_path}/vectors"
    os.makedirs(vector_path)

    df = ray.data.read_parquet(f"{fine_grained_path}/vectors.sorted.cached").to_pandas()
    coarse_emb = infer_coarse_embedding(df)
    assert coarse_emb.dbidx.is_monotonic_increasing
    coarse_emb.to_parquet(vector_path + "/part0.parquet")

    os.rename(output_path, final_output_path)
