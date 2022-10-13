import argparse
from seesaw.research.knn_methods import KNNGraph
import os

parser = argparse.ArgumentParser(
    description="create and preprocess dataset for use by Seesaw"
)

parser.add_argument(
    "--column",
    type=str,
    default='vectors',
    help="name of column within dataset",
)

parser.add_argument(
    "--k",
    type=int,
    # default=50,
    help="the k in k-nn: how many neighbors",
)

parser.add_argument(
    "--inputpath",
    type=str,
    help="Parquet dataset with the vectors readable by ray.data",
)

parser.add_argument(
    "--outputpath", type=str, help="where to save this"
)

args = parser.parse_args()

import ray
ray.init('auto', namespace='seesaw')
#from seesaw.services import get_parquet
from seesaw.util import parallel_read_parquet

inpath = os.path.expandvars(args.inputpath)
assert os.path.exists(inpath)

outpath = os.path.expandvars(args.outputpath)
assert not os.path.exists(outpath), 'output path already exists.'
# os.makedirs(outpath, exist_ok=False)

df = parallel_read_parquet(inpath)
vectors = df[args.column].to_numpy()
knng, other = KNNGraph.from_vectors(vectors, n_neighbors=args.k)
knng.save(outpath, overwrite=True) #bc I just made the directory



