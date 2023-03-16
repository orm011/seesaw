#import pytorch_lightning as pl
import torch
import torch.nn as nn
import math
import transformers
from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPConfig,
    CLIPVisionConfig,
    CLIPTextConfig,
)
import os
#from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import PIL
import numpy as np
import random
import io
from ray import tune


class MappedDataset(torch.utils.data.Dataset):
    def __init__(self, ds, tx):
        self.ds = ds
        self.tx = tx

    def map(self, tx):
        return MappedDataset(self, tx=tx)

    def __getitem__(self, idx):
        return self.tx(self.ds[idx])

    def __len__(self):
        return len(self.ds)


def split_into_sentences(desc):
    lst = desc.split(".")
    stripped = [l.strip(". ") for l in lst]
    whole = [s for s in stripped if s != ""]
    return whole


class CLIPTx:
    def __init__(self, processor):
        self.processor = processor

    def preprocess_tx(self, dataset_tup):
        im = PIL.Image.open(io.BytesIO(dataset_tup["image_bytes"]))
        sentences = split_into_sentences(dataset_tup["text"])
        sentence = random.choice(sentences)
        inputs = self.processor(text=[sentence], images=[im], return_tensors="pt")
        return inputs

    def pad_collate(self, input_list):
        ## list of dicts with elements 'input ids, attention mask, pixel_values'
        token_dict = {
            "input_ids": [d["input_ids"][0] for d in input_list],
            "attention_mask": [d["attention_mask"][0] for d in input_list],
        }

        ans = self.processor.tokenizer.pad(
            token_dict, padding="longest", return_tensors="pt"
        )
        ans["pixel_values"] = torch.cat([d["pixel_values"] for d in input_list])
        return ans


class MultiModalDataModule(pl.LightningDataModule):
    def __init__(
        self, dataset, processor, *, test_size, batch_size, val_batch_size, num_workers
    ):
        super().__init__()
        self.processor = processor
        self.dataset = dataset.shuffle().train_test_split(test_size=test_size)
        self.preproc = CLIPTx(processor)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            MappedDataset(self.dataset["train"], self.preproc.preprocess_tx),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.preproc.pad_collate,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            MappedDataset(self.dataset["test"], self.preproc.preprocess_tx),
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.preproc.pad_collate,
        )


def warmup(warmup_epochs, k=1.0):
    def fun(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs  # 0

        curr_epoch = max(epoch - warmup_epochs, 1)  # >= 1
        return 1.0 / math.sqrt(max(curr_epoch / k, 1.0))

    return fun


def add_to_group(opt_config, groups, name, param):
    """places param within the most specific group possible based on the config"""

    longest = ""
    for gp in opt_config.keys():
        if name.startswith(gp) and len(gp) > len(longest):
            longest = gp

    if longest == "":
        print("warning: no rule for parameter", name)
        assert False

    opts = opt_config[longest]
    if opts is None:
        return  # ignoring this param bc config tells us to

    if longest not in groups:
        bias_opts = {**opts, "weight_decay": 0}  # don't decay bias
        groups[longest] = (
            {"name": name, "params": [], **opts},
            {"name": name + "_biases", "params": [], **bias_opts},
        )

    gp = groups[longest]
    if name.endswith("bias"):
        gp[1]["params"].append(param)
    else:
        gp[0]["params"].append(param)


def configure_optimizer(m, h):
    prelim_groups = {}
    opt_config = h["opt_config"]
    for (name, param) in m.named_parameters():
        add_to_group(opt_config, prelim_groups, name, param)

    groups = []
    for _, gps in prelim_groups.items():
        for gp in gps:
            if len(gp["params"]) > 0:
                groups.append(gp)

    optimizer = transformers.AdamW(params=groups)

    lr_scheduler = transformers.get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=h["num_warmup_steps"]
    )
    return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class CLIPFineTunedModel(pl.LightningModule):
    def __init__(self, path_or_model, **kwargs):
        super().__init__()
        # todo: its easier to let pl save everything so it can also restore with .load_from_checkpoint
        # then, instead of passing a model or path, we should pass a config to construct a blank clip model,
        # then use load weights to get the weights
        self.save_hyperparameters(ignore="path_or_model")

        if isinstance(path_or_model, str):
            path = path_or_model
            model = CLIPModel.from_pretrained(path)
        elif isinstance(path_or_model, CLIPModel):
            model = path_or_model

        model.logit_scale = nn.Parameter(
            torch.tensor(
                self.hparams.logit_scale_init,
                device=model.logit_scale.device,
                dtype=model.logit_scale.dtype,
            )
        )

        self.model = model

    def encode_string(self, arg):
        with torch.no_grad():
            self.model.eval()
            strs = self.proc.tokenizer(arg, return_tensors="pt", padding=True)
            return self.model.get_text_features(**strs)

    def forward(self, inputs):
        return self.model(**inputs, return_loss=True)

    def basic_step(self, batch, train_or_val):
        batch_out = self.forward(batch)
        loss = batch_out.loss
        self.log(f"{train_or_val}_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.basic_step(batch, "val")

    def training_step(self, batch, batch_idx):
        self.log("logit_scale", self.model.logit_scale)
        ans = self.basic_step(batch, "train")
        return ans

    def configure_optimizers(self):
        return configure_optimizer(self.model, self.hparams.opt_config)


import pyarrow as pa
from datasets import Dataset


# from finetuning previous round on bird dataset, not necessarily good for other stuff
_best_config = {
    "batch_size": 128,
    "visual_projection_lr": 0.005393393095168505,
    "text_projection_lr": 0.00014453864858663312,
    "vision_model_lr": 5.0388709172642817e-05,
    "text_model_lr": 3.6322830104757226e-05,
    "logit_scale_lr": 0.00022781599914833148,
    "lr": 2.5723555839546368e-05,
    "visual_projection_decay": 0.08149420777631101,
    "text_projection_decay": 0.022803100867131183,
    "vision_model_decay": 5.6197902543582653e-05,
    "text_model_decay": 0.01651777070548456,
    "logit_scale_decay": 0.00018720038359285344,
    "decay": 0.008557100269919979,
    "init_scale": 17.08693539204199,
    "num_warmup_steps": 8,
    "num_workers": 8,
    "test_size": 1000,
    "val_batch_size": 500,
}

import ray.tune as tune


def generate_sample(v):
    if isinstance(v, dict):
        return {k: generate_sample(w) for (k, w) in v.items()}
    elif isinstance(v, list):
        return [generate_sample(w) for w in v]
    elif getattr(v, "sample", None) != None:
        return v.sample()
    else:
        return v


from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)

# def make_model(config)


def clip_fine_tune(
    config,
    num_epochs,
    num_gpus,
    dataset: pa.Table,
    init_config: CLIPConfig,
    init_state_dict: dict,
    processor: CLIPProcessor,
):
    if "SLURM_NTASKS" in os.environ:
        del os.environ["SLURM_NTASKS"]

    if "SLURM_JOB_NAME" in os.environ:
        del os.environ["SLURM_JOB_NAME"]

    bird_dataset = dataset
    data_mod = MultiModalDataModule(
        dataset=bird_dataset,
        processor=processor,
        test_size=config["test_size"],
        batch_size=config["batch_size"],
        val_batch_size=config["val_batch_size"],
        num_workers=config["num_workers"],
    )

    clip_model = CLIPModel(init_config)
    clip_model.load_state_dict(init_state_dict)
    model = CLIPFineTunedModel(clip_model, **config)

    tune_cbs = [TuneReportCheckpointCallback(["val_loss"], on="validation_end")]
    logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".")

    trainer = pl.Trainer(
        logger=logger,
        num_sanity_val_steps=0,
        max_epochs=num_epochs,
        gpus=math.ceil(num_gpus),
        progress_bar_refresh_rate=0,
        log_every_n_steps=1,
        callbacks=[LearningRateMonitor(logging_interval="step")] + tune_cbs,
    )

    trainer.validate(model, data_mod)
    trainer.fit(model, data_mod)
    return trainer


from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


def make_trainable(
    *, num_epochs, gpus_per_trial, dataset, init_config, init_state_dict, processor
):
    return tune.with_parameters(
        clip_fine_tune,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial,
        dataset=dataset,
        init_config=init_config,
        init_state_dict=init_state_dict,
        processor=processor,
    )


import ray
import argparse

from ray.tune import register_trainable


def load_experiment(tune_exp_dir):  # can be run on ray 1.9.2
    # eg. if exp name is try5 /home/gridsan/omoll/ray_results/try5
    register_trainable(
        "clip_fine_tune",
        make_trainable(
            num_epochs=10,
            gpus_per_trial=0.5,
            dataset=None,
            init_config=None,
            init_state_dict=None,
            processor=None,
        ),
    )
    exp = tune.ExperimentAnalysis(tune_exp_dir)
    return exp


def load_best_model(exp: tune.ExperimentAnalysis, metric="val_loss", mode="min"):
    """Loads best checkpoint overall"""
    tr = exp.get_best_trial(metric=metric, mode=mode, scope="all")
    chkpoint = exp.get_best_checkpoint(tr, metric=metric, mode=mode)
    model = CLIPFineTunedModel.load_from_checkpoint(chkpoint + "/checkpoint")
    return model


# can return None (which means don't train this param at all)
def random_rate(
    prob_none,
    prob_zero_decay=0.2,
    lr_scale=tune.loguniform(1e-7, 1e-2),
    decay_scale=tune.loguniform(1e-5, 1e-1),
):
    def random_fun(spec):
        if tune.uniform(0, 1).sample() < prob_none:
            return None
        else:
            if tune.uniform(0, 1).sample() < prob_zero_decay:
                decay = 0.0
            else:
                decay = decay_scale.sample()

            return {"lr": lr_scale.sample(), "weight_decay": decay}

    return tune.sample_from(random_fun)


def make_config(num_cpus):
    config = {
        "batch_size": tune.choice([32, 64]),
        "logit_scale_init": tune.quniform(
            -1.0, 5.0, q=0.1
        ),  # value from checkpoint is 4.6
        "opt_config": {  # for each parameter, will find the longest matching prefix and apply that rule.
            "logit_scale": random_rate(0.0),
            "text_model": random_rate(0.7),
            "text_model.embeddings": random_rate(0.7),
            "text_model.encoder.layers.0.layer_norm": random_rate(0.5),
            "text_model.encoder.layers.11": random_rate(0.5),
            "text_model.encoder.layers.11.mlp": random_rate(0.3),
            "text_model.encoder.layers.11.layer_norm2": random_rate(0.3),
            "text_model.final_layer_norm": random_rate(0.1),
            "text_projection": random_rate(0.1),
            "vision_model": None,
            "visual_projection": None,
        },
        "num_warmup_steps": tune.choice([20, 40]),
        "num_workers": num_cpus,
        "test_size": 1000,
        "val_batch_size": 500,
    }
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run a fine-tuning search")
    parser.add_argument(
        "--num_samples", type=int, default=1, help="how many different runs to try"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=3, help="max epochs to run each"
    )
    parser.add_argument("--local_dir", type=str, required=True, help="tune local dir")
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="name for experiment, also for subdir within local dir where to save",
    )
    # parser.add_argument('--text_model_only', type=bool, default=False, required=True, help='fine tune only text model side of model')

    args = parser.parse_args()

    ray.init("auto", namespace="seesaw")
    tmpdir = os.environ["TMPDIR"] + "/base"
    os.system(
        f"rsync -Rrv /home/gridsan/omoll/./data/bird_guide_single_parquet/ {tmpdir}"
    )
    bird_tab = pa.parquet.read_table(
        f"{tmpdir}/data/bird_guide_single_parquet/",
        columns=["description", "image_bytes"],
    )
    bird_dataset = Dataset(bird_tab).filter(
        lambda tup: tup["description"] is not None and tup["image_bytes"] is not None
    )
    bird_dataset = bird_dataset.rename_column("description", "text")

    cpus_per_trial = 20
    gpus_per_trial = 1
    config = make_config(cpus_per_trial)
    grace_period = 5

    metric = "val_loss"

    scheduler = ASHAScheduler(
        max_t=max(grace_period, args.max_epochs),
        grace_period=grace_period,
        reduction_factor=2,
    )

    reporter = CLIReporter(
        parameter_columns=[],
        metric_columns=[
            metric,
            "training_iteration",
        ],
        max_report_frequency=60,
    )

    base_model = CLIPModel.from_pretrained(
        f"/home/gridsan/omoll/xmodexp/notebooks/models/clip-vit-base-patch32"
    )
    init_config = base_model.config
    init_state_dict = base_model.state_dict()

    processor = CLIPProcessor.from_pretrained(
        f"/home/gridsan/omoll/xmodexp/notebooks/models/clip-vit-base-patch32"
    )

    trainable = make_trainable(
        num_epochs=args.max_epochs,
        gpus_per_trial=gpus_per_trial,
        dataset=bird_dataset,
        init_config=init_config,
        init_state_dict=init_state_dict,
        processor=processor,
    )

    analysis = tune.run(
        trainable,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        local_dir=args.local_dir,  # must be in the shared File system, else change below
        sync_config=tune.SyncConfig(syncer=None),
        metric=metric,
        mode="min",
        config=config,
        num_samples=args.num_samples,
        trial_dirname_creator=lambda trial: trial.trial_id,  # avoid super long name with config
        scheduler=scheduler,
        log_to_file=True,
        progress_reporter=reporter,
        name=args.experiment_name,
        keep_checkpoints_num=1,
        checkpoint_score_attr=f"min-{metric}",
    )
