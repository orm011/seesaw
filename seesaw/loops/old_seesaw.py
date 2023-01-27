from .loop_base import *
from .point_based import *
from ..search_loop_models import adjust_vec, adjust_vec2
from ..pairwise_rank_loss import VecState
from ..search_loop_tools import *
from ..search_loop_models import *
import sklearn
import sklearn.linear_model

class OldSeesaw(PointBased):
    def __init__(
        self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams
    ):
        super().__init__(gdm, q, params)
        p = self.params
        assert p.interactive == 'pytorch'

    def set_text_vec(self, tvec):
        super().set_text_vec(tvec)

        s = self.state
        p = self.params

        if self.params.method_config.get("model_type", None) == "multirank2":
            self.state.vec_state = VecState(
                tvec,
                margin=p.loss_margin,
                opt_class=torch.optim.SGD,
                opt_params={"lr": p.learning_rate},
            )

    def refine(self):
        """
        update based on vector. box dict will have every index from idx batch, including empty dfs.
        """
        s = self.state
        p = self.params

        Xt, yt = self.q.getXy()

        if (yt.shape[0] == 0) or (yt.max() == yt.min()):
            pass # nothing to do yet.

        if p.interactive == "sklearn":
            lr = sklearn.linear_model.LogisticRegression(
                class_weight="balanced"
            )
            lr.fit(Xt, yt)
            s.tvec = lr.coef_.reshape(1, -1)
        elif p.interactive == "pytorch":
            prob = yt.sum() / yt.shape[0]
            w = np.clip((1 - prob) / prob, 0.1, 10.0)

            cfg = p.method_config

            if cfg["model_type"] == "logistic":
                mod = PTLogisiticRegression(
                    Xt.shape[1],
                    learning_ratep=p.learning_rate,
                    C=0,
                    positive_weight=w,
                )
                if cfg["warm_start"] == "warm":
                    iv = torch.from_numpy(s.tvec)
                    iv = iv / iv.norm()
                    mod.linear.weight.data = iv.type(mod.linear.weight.dtype)
                elif cfg["warm_start"] == "default":
                    pass

                fit_reg(
                    mod=mod,
                    X=Xt.astype("float32"),
                    y=yt.astype("float"),
                    batch_size=p.minibatch_size,
                )
                s.tvec = mod.linear.weight.detach().numpy().reshape(1, -1)
            elif cfg["model_type"] in ["cosine", "multirank"]:
                for i in range(cfg["num_epochs"]):
                    s.tvec = adjust_vec(
                        s.tvec,
                        Xt,
                        yt,
                        learning_rate=cfg["learning_rate"],
                        max_examples=cfg["max_examples"],
                        minibatch_size=cfg["minibatch_size"],
                        loss_margin=cfg["loss_margin"],
                    )
            elif cfg["model_type"] in ["multirank2"]:
                npairs = yt.sum() * (1 - yt).sum()
                max_iters = (
                    math.ceil(
                        min(npairs, cfg["max_examples"])
                        // cfg["minibatch_size"]
                    )
                    * cfg["num_epochs"]
                )
                print("max iters this round would have been", max_iters)
                # print(s.vec_state.)

                # vecs * niters = number of vector seen.
                # n vec seen <= 10000
                # niters <= 10000/vecs
                max_vec_seen = 10000
                n_iters = math.ceil(max_vec_seen / Xt.shape[0])
                n_steps = np.clip(n_iters, 20, 200)

                # print(f'steps for this iteration {n_steps}. num vecs: {Xt.shape[0]} ')
                # want iters * vecs to be const..
                # eg. dota. 1000*100*30

                for _ in range(n_steps):
                    loss = s.vec_state.update(Xt, yt)
                    if loss == 0:  # gradient is 0 when loss is 0.
                        print("loss is 0, breaking early")
                        break

                s.tvec = s.vec_state.get_vec()
            elif cfg["model_type"] == "solver":
                s.tvec = adjust_vec2(s.tvec, Xt, yt, **p.solver_opts)
            else:
                assert False, "model type"
        else:
            assert False