import inspect
import os
import torch
import logging

import pandas as pd
import pyarrow as pa

import copy
import json
import numpy as np

import os
import shutil
from contextlib import contextmanager
import warnings

@contextmanager
def transactional_folder(folder_path, force=False):
    '''
        Example usage
        folder_path = "/path/to/folder"
        with transactional_folder(folder_path) as temp_folder:
            Perform file modifications inside the 'with' block
            You can use temp_folder to create, modify, or delete files

        If everything was successful, the changes are now committed as a single folder
        If an exception occurred, no changes have been applied
        force means we overwrite an existing folder if it exists, but only if successful
    '''
    if os.path.exists(folder_path):
        if force:
            warnings.warn(
                f'folder {folder_path} already exists. overwriting due to force=True'
            )
        else:
            raise Exception(f'folder {folder_path} already exists. use "force=True" to allow overwrite')

    
    folder_path = folder_path.rstrip('/')
    prepath = os.path.dirname(folder_path)
    name = os.path.basename(folder_path)
    temp_folder = os.path.join(prepath, '.temp_' + name)
    
    try:
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        
        # Create a temporary folder to store the changes
        os.makedirs(temp_folder)

        # Yield the temporary folder to the code inside the 'with' block
        yield temp_folder

        # If everything succeeds, commit the changes by moving the temporary folder
        shutil.rmtree(folder_path, ignore_errors=True)
        shutil.move(temp_folder, folder_path)
    except Exception as e:
        # If an exception occurs, clean up the temporary folder
        shutil.rmtree(temp_folder, ignore_errors=True)
        raise e


def is_valid_filename(filename):
    # Check if the filename is empty or None
    if not filename:
        return False

    # Check if the filename contains any invalid characters
    invalid_chars = {'/', '\0', '\\'}
    if any(char in filename for char in invalid_chars):
        return False

    # Check if the filename is '.' or '..'
    if filename == '.' or filename == '..':
        return False

    return True

def _get_fixed_metadata(schema):
    import ray.data
    import ray.data.extensions

    new_meta = copy.deepcopy(schema.metadata)
    new_pandas = json.loads(new_meta[b'pandas'].decode())
    column_info = new_pandas['columns']

    for i,field in enumerate(schema):
        typ = field.type

        if (isinstance(typ,ray.data.extensions.ArrowTensorType) and 
                column_info[i]['numpy_type'] == 'TensorDtype'): # old style metadata format

                pd_dtype = typ.to_pandas_dtype()
                typestr = str(np.dtype(pd_dtype._dtype))

                # new style format
                column_info[i]['numpy_type'] = f'TensorDtype(shape={pd_dtype._shape}, dtype={typestr})'

    new_meta[b'pandas'] = json.dumps(new_pandas).encode()
    return new_meta

import pyarrow as pa
import pyarrow.parquet as pq

def arrow_to_df(tab):
    fixed_meta = _get_fixed_metadata(tab.schema)
    return tab.replace_schema_metadata(fixed_meta).to_pandas()

def parallel_read_parquet(path, columns=None, parallelism=-1) -> pd.DataFrame:
    """uncached version"""
    if parallelism != 0:
        import ray.data

        ds = ray.data.read_parquet(path, columns=columns, parallelism=parallelism)
        tabs = ray.get(ds.to_arrow_refs())

        if len(tabs) > 0:
            fixed_meta = _get_fixed_metadata(tabs[0].schema) # 
            tabs = [tab.replace_schema_metadata(fixed_meta) for tab in tabs]

        dfs = [tab.to_pandas() for tab in tabs]
        df = pd.concat(dfs)
    else:
        tab = pq.read_table(path, columns=columns)
        df = tab.to_pandas()

    return df


def as_batch_function(fun):
    def bfun(batch):
        res = []
        if isinstance(batch, pd.DataFrame):
            for b in batch.itertuples(index=False):
                res.append(fun(b._asdict()))

            return pd.DataFrame.from_records(res)

        elif isinstance(batch, pa.Table):
            assert False, 'not sure how to iterate here'
        elif isinstance(batch, dict):
            assert False, 'unsupported format'
        else: # try just iterating
            assert False, 'python no longer supported by ray data.'

    return bfun

def vls_init_logger():
    if False: # no longer used
        import pytorch_lightning as pl
        pl.utilities.distributed.log.setLevel(logging.ERROR)
        logging.getLogger("lightning").setLevel(logging.ERROR)
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)  
    logging.captureWarnings(True)


def reset_num_cpus(num_cpus: int):
    print(f"resetting num cpus for process {num_cpus}")
    os.environ["OMP_NUM_THREADS"] = str(num_cpus)
    torch.set_num_threads(num_cpus)
    assert os.environ["OMP_NUM_THREADS"] == str(num_cpus)
    assert torch.get_num_threads() == num_cpus


def copy_locals(dest_name: str = None):
    """
    copies all local variables from this context into the jupyter top level, eg, for easier
    debugging of data and for prototyping new code that is eventually meant to run within this context.
    """
    stack = inspect.stack()

    caller = stack[1]
    local_dict = {
        k: v for k, v in caller.frame.f_locals.items() if not k.startswith("_")
    }

    notebook_caller = None
    for st in stack:
        if st.function == "<module>":
            notebook_caller = st
            break

    if notebook_caller is None:
        print("is this being called from within a jupyter notebook?")
        return

    if dest_name is None:
        print("copying variables to <module> globals...", list(local_dict.keys()))
        notebook_caller.frame.f_globals.update(local_dict)
    else:
        print(
            "copying variables to <module> variable {}".format(dest_name)
        )  # list(local_dict.keys()))
        notebook_caller.frame.f_globals[dest_name] = local_dict
