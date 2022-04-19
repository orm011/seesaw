"""allows me to do from seesaw.all_imports import * behave as if everything is a single file"""
import importlib
import pkgutil
import os

package_path = os.path.dirname(__file__)
this_module_name = os.path.basename(__file__).split(".")[0]


def _import_all_names(module):
    if hasattr(module, "__all__"):
        all_names = module.__all__
    else:
        all_names = [name for name in dir(module) if not name.startswith("_")]

    globals().update({name: getattr(module, name) for name in all_names})


_exclude = [
    this_module_name,
    "embedding_plot",
    "vloop_dataset_loaders",
    "clip_module",
    "textual_feedback_box",
    "figures",
]
for mod_info in pkgutil.iter_modules([package_path]):
    if mod_info.name in _exclude:
        continue

    mod = importlib.import_module(f"{__package__}.{mod_info.name}")
    _import_all_names(mod)
