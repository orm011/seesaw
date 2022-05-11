import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms as T
from pytorch_lightning.loggers import TensorBoardLogger

import sklearn
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset


class MatchedEmbeddingDataset(object):
    """Dataset for training a translation model as done in Recipe1M,
    generates positive and "negative" examples from a captioned dataset
    """

    def __init__(
        self,
        caption_embeddings,
        caption_ids,
        image_embeddings,
        fraction_positive,
        coco_cap=None,
    ):

        assert caption_embeddings.shape[0] == caption_ids.shape[0]
        self.caption_embeddings = caption_embeddings
        caption_ids = caption_ids.assign(pos=np.arange(caption_ids.shape[0]))
        self.caption_ids = caption_ids

        self.image_embeddings = image_embeddings
        # self.target_size = target_size
        self.fraction_positive = fraction_positive
        self.coco_cap = coco_cap

    def __len__(self):
        return self.image_embeddings.shape[0]

    def __getitem__(self, idx):
        ## pick idx^th image
        ## with prob p pick one of the correct captions
        ## with prob 1 - p pick a 'wrong' caption
        image_idx = idx
        if np.random.rand() < self.fraction_positive:
            caps = self.caption_ids[self.caption_ids["image_idx"] == idx]
            cap = caps.sample(n=1)
            caption_idx = cap.pos.iloc[0]
            target_sim = 1.0
        else:
            caps = self.caption_ids[self.caption_ids["image_idx"] != idx]
            cap = caps.sample(n=1)
            caption_idx = cap.pos.iloc[0]
            target_sim = 0.0

        if self.coco_cap is None:
            return (
                self.caption_embeddings[caption_idx],
                self.image_embeddings[image_idx],
                target_sim,
            )
        else:  # debug mode see actual image and caption
            return (
                self.caption_ids["captions"].iloc[caption_idx],
                self.coco_cap[image_idx][0],
                target_sim,
            )


class MatchedEmbeddingDataset1to1(object):
    """Dataset for training a translation model as done in Recipe1M,
    generates positive and "negative" examples from a captioned dataset
    """

    def __init__(
        self, caption_embeddings, image_embeddings, fraction_positive, coco_cap=None
    ):
        assert caption_embeddings.shape[0] == image_embeddings.shape[0]
        self.caption_embeddings = caption_embeddings
        self.image_embeddings = image_embeddings

        self.fraction_positive = fraction_positive
        self.coco_cap = coco_cap

    def __len__(self):
        return self.image_embeddings.shape[0]

    def __getitem__(self, idx):
        ## pick idx^th image
        ## with prob p pick one of the correct captions
        ## with prob 1 - p pick a 'wrong' caption
        image_idx = idx

        if np.random.rand() < self.fraction_positive:
            caption_idx = image_idx
            target_sim = 1.0
        else:
            caption_idx = image_idx
            while caption_idx == image_idx:
                caption_idx = np.random.randint(0, self.caption_embeddings.shape[0])

            orig_embed = self.caption_embeddings[image_idx]
            rand_embed = self.caption_embeddings[caption_idx]
            target_sim = (orig_embed @ rand_embed) / (
                np.linalg.norm(orig_embed) * np.linalg.norm(rand_embed)
            )
            target_sim = 0

        # if self.coco_cap is None:
        return (
            self.caption_embeddings[caption_idx],
            self.image_embeddings[image_idx],
            target_sim,
        )
        # else: # debug mode see actual image and caption
        # return (self.caption_ids['captions'].iloc[caption_idx], self.coco_cap[image_idx][0], target_sim)


## a learned key- 1 value lookup rather than hardcoded NN lookup
## using n_keys (k,v) pairs
# w = softmax( q @ K ) (dot vs 'keys' for similarity, focus on most similar items)
# w @ V (return weighted sum of values)


class KVMapping(nn.Module):
    def __init__(self, k_size, n_keys, v_size):
        super().__init__()
        self.Kmap = nn.Linear(k_size, n_keys, bias=False)
        self.attend = nn.Softmax(dim=-1)

        self.Vmap = nn.Linear(n_keys, v_size, bias=False)
        self.normalizer = np.sqrt(n_keys)

    def forward(self, qs):
        w1 = self.Kmap(qs) / self.normalizer
        weights = self.attend(w1)  # size B,n_keys

        # return something in V space
        return self.Vmap(weights)  # size B,v_size


class TextImageCrossModal(pl.LightningModule):
    def __init__(self, caption_vec_size, image_vec_size, n_keys, cm_val):
        super().__init__()
        self.save_hyperparameters("caption_vec_size", "image_vec_size", "n_keys")
        self.cm_val = cm_val

        #         cap_hsize = 2*caption_vec_size

        #         self.cap_translator = nn.Sequential(
        #             nn.Linear(caption_vec_size, cap_hsize),
        #             nn.Dropout(p=.3),
        #             nn.Sigmoid(),
        #             nn.Linear(cap_hsize, image_vec_size),
        #             nn.Dropout(p=.3)
        #         )
        self.cap_translator = lambda x: x

        # self.img_translator =  #nn.Sequential(#nn.LayerNorm(torch.Size([image_vec_size])),
        self.img_translator = KVMapping(
            k_size=image_vec_size,
            n_keys=n_keys,  # similar to kd tree values
            v_size=caption_vec_size,
        )
        # )

        #  self.img_translator = lambda x : x# worked better than training both at the same time
        ## may need to tune them separately
        self.loss = nn.CosineEmbeddingLoss()

    def from_string_vec(self, caption_emb):
        if isinstance(caption_emb, np.ndarray):
            caption_emb = torch.from_numpy(caption_emb)

        caption_emb = caption_emb.to(self.device)
        ret = self.cap_translator(caption_emb)
        return ret.detach().to("cpu").numpy()

    def from_image_vec(self, image_emb):
        if isinstance(image_emb, np.ndarray):
            image_emb = torch.from_numpy(image_emb)

        image_emb = image_emb.to(self.device)
        ret = self.img_translator(image_emb)
        return ret.detach().to("cpu").numpy()

    def forward(self, caption_emb, image_emb):
        return (self.cap_translator(caption_emb), self.img_translator(image_emb))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        caption_emb, image_emb, sims = train_batch
        xcap, xim = self.forward(caption_emb, image_emb)

        # cosine embedding loss expects 1,-1 format
        target = sims * 2 - 1
        train_loss = self.loss(xcap, xim, target)
        self.log("train_loss", train_loss)

        return train_loss

    def validation_step(self, val_batch, batch_idx, dataset_idx):
        if dataset_idx == 0:
            caption_emb, image_emb, sims = val_batch
            xcap, xim = self.forward(caption_emb, image_emb)

            target = sims * 2 - 1
            loss = self.loss(xcap, xim, target)
            return {"val_loss": loss}
        elif dataset_idx == 1:
            image_emb = val_batch[0]
            return {"val_emb": self.from_image_vec(image_emb)}

    def validation_epoch_end(self, validation_step_outputs):
        # capt vecs, img vecs, model (new)
        # for pred in validation_step_outputs:

        vloss = torch.stack([v["val_loss"] for v in validation_step_outputs[0]]).mean()
        self.log("val_loss", vloss)

        embedded_img_db = np.concatenate(
            [v["val_emb"] for v in validation_step_outputs[1]]
        )
        cm = CosineMetric(
            ref_vectors=self.cm_val.ref_vectors[: embedded_img_db.shape[0]],
            test_vectors=embedded_img_db,
            max_top_n=100,
        )

        ratios = []
        color_ratios = []
        for qstr, q_enc in zip(val_query_names, val_query_ref_vecs):
            _, _, ratio = cm.eval_testvec(
                q_enc, self.from_string_vec(q_enc), topn=100, weighted=True
            )
            ratios.append(ratio)
            if qstr in example_colors:
                color_ratios.append(ratio)

        self.log("val_retrieval", torch.tensor(ratios).mean())
        self.log("val_retrieval_color", torch.tensor(color_ratios).mean())


class CosineMetric(object):
    def __init__(self, ref_vectors, test_vectors, max_top_n=200):
        assert ref_vectors.shape[0] == test_vectors.shape[0]

        self.max_top_n = max_top_n
        self.ref_vectors = (
            ref_vectors / np.linalg.norm(ref_vectors, axis=1)[:, np.newaxis]
        )
        self.ref_vec_index = NearestNeighbors(metric="cosine", n_neighbors=max_top_n)
        self.ref_vec_index.fit(self.ref_vectors)

        self.test_vectors = test_vectors
        self.test_vec_index = NearestNeighbors(metric="cosine", n_neighbors=max_top_n)
        self.test_vec_index.fit(test_vectors)

    def compute_similarity(self, query_vec, positions):
        ref_vecs = self.ref_vectors[positions].transpose()
        nvec = query_vec / np.linalg.norm(query_vec)
        cosines = nvec @ ref_vecs  # self.ref_vectors.transpose()
        return cosines

    def eval_testvec(self, ref_vec, test_vec, topn, weighted=False):
        # actual ranking:
        assert topn <= self.max_top_n

        _, test_rnk = self.test_vec_index.kneighbors(test_vec.reshape(1, -1))
        return self.eval_ranking(ref_vec, test_rnk.reshape(-1)[:topn], weighted)

    def eval_ranking(self, ref_vec, test_ranking, weighted=False):

        topn = test_ranking.shape[0]

        if weighted:
            wt = np.sqrt(1.0 / (np.arange(topn) + 1))
            wt = wt / wt.sum()
        else:
            wt = np.ones(topn) / topn

        assert wt.shape[0] == topn
        assert np.isclose(wt.sum(), 1.0)
        assert wt.max() == wt[0]
        assert topn <= test_ranking.shape[0]

        optimal_cos_dist, rnk_ref = self.ref_vec_index.kneighbors(
            ref_vec.reshape(1, -1)
        )
        optimal_sims = 1 - optimal_cos_dist.reshape(-1)[:topn]  # sims[optimal_pos]
        optimal_total = optimal_sims @ wt  # .sum()

        actual_sims = self.compute_similarity(ref_vec, test_ranking)
        actual_total = actual_sims @ wt  # .sum()
        return actual_total, optimal_total, (actual_total / optimal_total)
