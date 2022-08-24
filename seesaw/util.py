import inspect
import os
import torch
import logging


def vls_init_logger():
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
