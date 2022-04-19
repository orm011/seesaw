import string
from seesaw.multiscale_index import MultiscaleIndex
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
import pandas as pd
from clip.model import build_model
from .clip_module import CLIPFineTunedModel, CLIPTx, MappedDataset
import torch
from .multiscale_index import add_iou_score
from ray.data.extensions import TensorArray
import torch.optim
from .clip_module import configure_optimizer


def join_vecs2annotations(db: MultiscaleIndex, dbidx, annotations):
    patch_box_df = db.get_data(dbidx)
    roi_box_df = pd.DataFrame.from_records([b.dict() for b in annotations])

    dfvec = add_iou_score(patch_box_df, roi_box_df)
    dfvec = dfvec.assign(
        descriptions=dfvec.best_box_idx.map(lambda idx: annotations[idx].description)
    )

    dfbox = add_iou_score(roi_box_df, patch_box_df)

    matched_vecs = np.stack(
        [dfvec.vectors.iloc[i].copy() for i in dfbox.best_box_idx.values]
    )
    dfbox = dfbox.assign(
        descriptions=dfbox.description, vectors=TensorArray(matched_vecs)
    )

    return dfvec, dfbox


def deduplicate_strings(string_list):
    s2id = {}
    sids = []
    for s in string_list:
        if s not in s2id:  # add new string to dict
            s2id[s] = len(s2id)

        sids.append(s2id[s])

    string_ids = np.array(sids)

    reverse_dict = {num: strng for (strng, num) in s2id.items()}
    id2string = np.array([reverse_dict[i] for i in range(len(s2id))])

    return {"strings": id2string, "indices": string_ids}


class LinearScorer(nn.Module):
    def __init__(self, init_weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(data=init_weight, requires_grad=True)
        self.bias = nn.Parameter(
            torch.tensor(0.0, device=init_weight.device), requires_grad=True
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(0.0, device=init_weight.device), requires_grad=True
        )

    def get_vec(self):
        return self.weight.detach().cpu().numpy()

    def forward(self, X: torch.Tensor):
        return (X @ self.weight.t()) * self.logit_scale.exp() + self.bias


class DynamicLinear(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.scorers = nn.ModuleDict()

    def add_scorer(self, name: str, init_weight: torch.Tensor):
        assert name not in self.scorers
        self.scorers.add_module(
            name, LinearScorer(init_weight=init_weight).to(self.device)
        )

    def get_vec(self, name):
        return self.scorers[name].get_vec()

    def forward(self, vecs: torch.Tensor):
        assert len(vecs.shape) == 2
        scores = []
        for (_, scorer) in self.scorers.items():
            scores.append(scorer(vecs))

        return torch.stack(scores).t()


### the way we pick vectors for training right now should be revisited
import transformers


def hinge(x, margin):  # 0 loss if x >= margin.
    return F.relu(-x + margin)


def rank_loss(ranking_scores, marked_accepted, margin):
    if 0 < marked_accepted.sum() < marked_accepted.shape[0]:
        pos_scores = ranking_scores[marked_accepted]
        neg_scores = ranking_scores[~marked_accepted]
        image_rank_losses = hinge(
            pos_scores.reshape(-1, 1) - neg_scores.reshape(1, -1), margin
        )
        return image_rank_losses.reshape(-1).mean()
    else:
        return None


class OnlineModel:
    def __init__(self, state_dict, config):
        if not torch.cuda.is_available():
            if config["device"].startswith("cuda"):
                print("Warning: no GPU available, using cpu instead")
                config = {**config, "device": "cpu"}  # overrule gpu if not available

        self.original_weights = {k: v.float() for (k, v) in state_dict.items()}

        self.device = config["device"]
        self.model = build_model(self.original_weights).float().to(self.device)
        self.linear_scorer = None
        self.config = config
        self._reset_model()
        self.mode = self.config["mode"]
        self.losses = []
        self._cache = {}

    def _reset_model(self):  # resets model and optimizers
        print("resetting model state")
        layers = ["text_projection"]
        reset_weights = {
            k: v.to(self.device)
            for (k, v) in self.original_weights.items()
            if k in layers
        }
        self.model.load_state_dict(reset_weights, strict=False)
        self.linear_scorer = DynamicLinear(self.device)
        print("done resetting model")

    def _encode_string(self, tokenized_strings):
        text_features = self.model.encode_text(tokenized_strings.to(self.device))
        text_features = F.normalize(text_features)
        return text_features

    def encode_string(self, string) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tokens = clip.tokenize([string])
            vecs = self._encode_string(tokens)
            return vecs.cpu().numpy()

    def compute_up_to(self, strings, layer) -> torch.Tensor:
        assert layer in ["text_projection", "full"]

        non_cached = []
        for s in strings:
            if s not in self._cache:
                non_cached.append(s)

        def closure(self, tokens):  # pass self.model
            # taken from model.encode_text
            x = self.token_embedding(tokens).type(
                self.dtype
            )  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]
            if layer == "text_projection":
                return x
            elif layer == "full":
                x = x @ self.text_projection
                return F.normalize(x)
            else:
                assert False

        if len(non_cached) > 0:
            tokens = clip.tokenize(non_cached).to(self.device)
            new_vecs = closure(self.model, tokens)
            for (s, v) in zip(non_cached, new_vecs):
                self._cache[s] = v

        ans = []
        for s in strings:
            ans.append(self._cache[s])

        return torch.stack(ans)

    def compute_from(self, x, layer) -> torch.Tensor:
        assert layer in ["text_projection", "full"]

        def closure(self, x):
            if layer == "text_projection":
                x = x @ self.text_projection
                return F.normalize(x)
            elif layer == "full":
                return x
            else:
                assert False

        return closure(self.model, x)

    def score_vecs(self, imagevecs):
        with torch.no_grad():
            self.model.eval()
            X = torch.from_numpy(imagevecs).to(self.device)
            logits = self.linear_scorer(X)
            if logits.shape[1] > 1:
                logits = logits.softmax(
                    dim=-1
                )  # want the score to stay within 0 to 1 range for visualization
            return logits[:, 0].cpu().numpy()

    def get_lookup_vec(self, str):
        return self.linear_scorer.get_vec(str)

    ### option 1: discount common mistakes via a softmax.
    ### assume only negatives get text description (including near misses)

    def prepare_annotated_pairs(
        self, imagevecs, marked_accepted, annotations, target_string
    ):
        """returns N image vecs, M unique (non empty) string vecs, and the string id of the annotation for each vec"""

        has_description = annotations != ""
        assert target_string != ""

        imagevecs = imagevecs[has_description]
        annotations = annotations[has_description]

        d = deduplicate_strings([target_string] + list(annotations))
        strings = d["strings"]
        string_ids = torch.from_numpy(d["indices"][1:]).to(self.device)
        imagevecs = torch.from_numpy(imagevecs).to(self.device)

        with torch.no_grad():
            self.model.eval()
            last_layer = "full" if self.mode == "linear" else "text_projection"
            stringvecs = self.compute_up_to(strings, last_layer)

        assert imagevecs.device == stringvecs.device
        assert stringvecs.shape[0] == strings.shape[0]
        return {
            "imagevecs": imagevecs,
            "stringvecs": stringvecs,
            "target": string_ids,
            "unique_strings": strings,
        }

    def update(self, imagevecs, marked_accepted, annotations, target_string):
        assert imagevecs.shape[0] == marked_accepted.shape[0]
        assert annotations.shape[0] == marked_accepted.shape[0]

        all_imagevecs = torch.from_numpy(imagevecs).to(self.device)
        r = self.prepare_annotated_pairs(
            imagevecs, marked_accepted, annotations, target_string
        )

        self._reset_model()
        self.model.train()
        self.linear_scorer.train()

        if self.mode == "linear":
            return self._update_linear(r, all_imagevecs, marked_accepted)
        elif self.mode == "finetune":
            return self._update_finetune(r, all_imagevecs, marked_accepted)
        else:
            assert False

    def _update_linear(self, description_data, all_imagevecs, marked_accepted):
        ## want mask for all vectors where the user has provided a better description
        ## ie. exclude empty descriptions '', and also exclude cases where the description
        ## is identical to the search query used as reference (we would be penalizing it)

        for (lookupstr, vec) in zip(
            description_data["unique_strings"], description_data["stringvecs"]
        ):
            assert lookupstr != ""
            self.linear_scorer.add_scorer(lookupstr, vec)

        # 2 groups: vectors modified more slowly
        # temps and biases more aggressively:
        pgs = [
            dict(
                params=[
                    p
                    for (name, p) in self.linear_scorer.named_parameters()
                    if name.endswith("weight")
                ],
                lr=0.001,
                weight_decay=0.0,
            ),
            dict(
                params=[
                    p
                    for (name, p) in self.linear_scorer.named_parameters()
                    if not name.endswith("weight")
                ],
                lr=0.002,
                weight_decay=0.0,
            ),
        ]

        opt = transformers.AdamW(pgs)
        lr_scheduler = transformers.get_constant_schedule_with_warmup(
            opt, num_warmup_steps=self.config["num_warmup_steps"]
        )

        def _description_loss(score_all_pairs, description_data):
            if len(description_data["unique_strings"]) > 1:
                assert score_all_pairs.shape[1] > 1
                return F.cross_entropy(
                    score_all_pairs, description_data["target"]
                ).mean()
            else:
                print("no textual annotationswith wich to make a loss")
                return None

        def training_step():
            self.model.train()

            score_all_pairs = self.linear_scorer(description_data["imagevecs"])
            label_rank_loss = _description_loss(score_all_pairs, description_data)

            ranking_scores = self.linear_scorer(all_imagevecs).log_softmax(dim=-1)[:, 0]
            image_rank_loss = rank_loss(
                ranking_scores, marked_accepted, margin=self.config["rank_margin"]
            )

            if label_rank_loss is None and image_rank_loss is None:
                return None

            loss1 = 0.0 if label_rank_loss is None else label_rank_loss
            loss2 = 0.0 if image_rank_loss is None else image_rank_loss

            loss = (1.0 - self.config["image_loss_weight"]) * loss1 + self.config[
                "image_loss_weight"
            ] * loss2
            return (loss, loss1, loss2)

        for _ in range(self.config["rounds"] + self.config["num_warmup_steps"]):
            opt.zero_grad()
            loss = training_step()
            if loss is None:
                print("no loss yet: no qualified labels")
                break
            (loss, l1, l2) = loss
            loss.backward()
            print(f"loss:{loss:.02f} label_loss: {l1:.02f} rank_loss: {l2:.02f}")
            opt.step()
            lr_scheduler.step()

        return []

    def _update_finetune(self, description_data, all_imagevecs, marked_accepted):
        r = configure_optimizer(self.model, self.config)
        opt: torch.optim.Optimizer = r["optimizer"]
        lr_scheduler = r["lr_scheduler"]

        d = description_data

        def _compute_label_loss(scores, target):
            if scores.shape[0] > 0 and scores.shape[1] > 1:
                # return hinge(scores[torch.arange(scores.shape[0]), target] - scores[:,0]) # needs to handle special case where target is 0
                return F.multi_margin_loss(
                    scores, target, margin=self.config["label_margin"]
                )
            else:
                return None

        def opt_closure():
            self.model.train()
            text_features = self.compute_from(d["stringvecs"], "text_projection")
            scores = d["imagevecs"] @ text_features.t()

            l1 = _compute_label_loss(scores, d["target"])

            rank_scores = (all_imagevecs @ text_features.t())[:, 0]
            l2 = rank_loss(
                rank_scores, marked_accepted, margin=self.config["rank_margin"]
            )

            if l1 is None and l2 is None:
                return None

            l1 = l1 if l1 is not None else 0.0
            l2 = l2 if l2 is not None else 0.0

            loss = (1.0 - self.config["image_loss_weight"]) * l1 + self.config[
                "image_loss_weight"
            ] * l2
            return loss, l1, l2

        for _ in range(self.config["rounds"] + self.config["num_warmup_steps"]):
            opt.zero_grad()
            loss = opt_closure()
            if loss is None:
                print("no loss yet: no qualified labels")
                break
            (loss, l1, l2) = loss
            loss.backward()
            print(f"loss:{loss:.02f} label_loss: {l1:.02f} rank_loss: {l2:.02f}")
            opt.step()
            lr_scheduler.step()
