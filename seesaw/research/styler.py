import numpy as np
from ray.data.extensions import TensorDtype
from IPython.display import HTML
from ray.data.extensions import TensorArray, TensorDtype


def to_ndarray(tensor_array):
    arr = tensor_array.to_numpy()
    out_arr = np.array([None for _ in arr], dtype="object")
    for i, l in enumerate(arr):
        out_arr[i] = l
    return out_arr


TensorArray.__array__ = to_ndarray


class ImageDisplayer:
    def __init__(self, dataset, host="localhost.localdomain:10000"):
        self.host = host
        self.ds = dataset

    def __call__(self, idx_or_string):  # use as formatter
        if isinstance(idx_or_string, (int, float, np.int64, np.int32)):
            path = self.ds.paths[int(idx_or_string)]
        elif isinstance(idx_or_string, str):
            path = idx_or_string
        else:
            assert False, type(idx_or_string)

        url = f"http://{self.host}/{self.ds.image_root}/{path}"
        return f'<img src="{url}" />'

    def show(self, idx_or_string):
        return HTML(self.__call__(idx_or_string))

    def display_df(self, df, im_col="dbidx", max_rows=10):
        def vec_formatter(vec):
            return np.array2string(
                np.array(vec),
                precision=2,
                suppress_small=True,
                threshold=8,
                separator=",",
            )

        vector_cols = [
            k for k, v in df.dtypes.to_dict().items() if isinstance(v, TensorDtype)
        ]
        df = df.assign(image_col=df[im_col])

        formatters = {"image_col": self, **{k: vec_formatter for k in vector_cols}}
        return HTML(df.to_html(max_rows=max_rows, formatters=formatters, escape=False))
