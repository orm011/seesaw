from .loop_base import *

class TextualLoop(LoopBase):
    def __init__(self, gdm, q, params):
        super().__init__(gdm, q, params)
        s = self.state
        p = self.params
        
        if self.params.interactive in ["textual"]:
            param_dict = gdm.global_cache.read_state_dict(
                "/home/gridsan/groups/fastai/omoll/seesaw_root/models/clip/ViT-B-32.pt",
                jit=True,
            )
            # s.model = OnlineModel(param_dict, p.method_config)


    def set_text_vec(self, tvec):
        raise NotImplementedError('implement me')

    def refine(self):
        p = self.params
        s = self.state

        if p.method_config["mode"] == "finetune":
            vec = s.model.encode_string(s.curr_str)
            rescore_m = lambda vecs: vecs @ vec.reshape(-1, 1)
        elif p.method_config["mode"] == "linear":
            if len(s.model.linear_scorer.scorers) == 0:  ## first time
                vec = s.model.encode_string(s.curr_str)
                s.model.linear_scorer.add_scorer(
                    s.curr_str, torch.from_numpy(vec.reshape(-1))
                )
            rescore_m = self.state.model.score_vecs
            vec = self.state.model.get_lookup_vec(s.curr_str)

        b = self.q.query_stateful(
            vector=vec,
            batch_size=p.batch_size,
            shortlist_size=p.shortlist_size,
            agg_method=p.agg_method,
            aug_larger=p.aug_larger,
            rescore_method=rescore_m,
        )

        return b

    def next_batch(self):
        p = self.params
        s = self.state
        if (
            "image_vector_strategy" not in p.dict()
            or p.image_vector_strategy == None
            or p.image_vector_strategy == "matched"
        ):
            vecs = []
            strs = []
            acc = []

            for dbidx in self.q.label_db.get_seen():
                annot = self.q.label_db.get(dbidx, format="box")
                assert annot is not None
                if len(annot) == 0:
                    continue

                dfvec, dfbox = join_vecs2annotations(self.q.index, dbidx, annot)
                # best_box_iou, best_box_idx

                ## vectors with overlap
                df = dfbox  # use boxes as guide for now
                mask_boxes = df.best_box_iou > p.method_config["vector_box_min_iou"]
                df = df[mask_boxes]
                if df.shape[0] > 0:
                    vecs.append(df.vectors.values)
                    strs.append(df.descriptions.values)
                    acc.append(df.marked_accepted.values)

            if len(vecs) == 0:
                print("no annotations for update... skipping")
                return

            all_vecs = np.concatenate(vecs)
            all_strs = np.concatenate(strs)
            marked_accepted = np.concatenate(acc)
        elif p.image_vector_strategy == "computed":
            vecs = []
            strs = []
            acc = []
            # annot = self.q.label_db.get(dbidx, format='box')
            for dbidx in self.q.label_db.get_seen():
                annot = self.q.label_db.get(dbidx, format="box")
                if len(annot) == 0:
                    continue

                vecs.append(self.compute_image_activations(dbidx, annot))
                strs.append()

            pass
        else:
            assert False, "unknown image vec strategy"

        losses = s.model.update(all_vecs, marked_accepted, all_strs, s.curr_str)
        print("done with update", losses)