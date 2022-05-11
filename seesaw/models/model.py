import torch.nn as nn
import torch.nn.functional as F
import torch
import transformers
import clip


def preprocess(tup, factor):
    """meant to preprocess dict with {path, dbidx,image}"""
    ptx = PyramidTx(tx=non_resized_transform(224), factor=factor, min_size=224)
    ims, sfs = ptx(tup["image"])
    acc = []
    for zoom_level, (im, sf) in enumerate(zip(ims, sfs), start=1):
        acc.append(
            {
                "file_path": tup["file_path"],
                "dbidx": tup["dbidx"],
                "image": im,
                "scale_factor": sf,
                "zoom_level": zoom_level,
            }
        )

    return acc


def postprocess_results(acc):
    flat_acc = {
        "iis": [],
        "jjs": [],
        "dbidx": [],
        "vecs": [],
        "zoom_factor": [],
        "zoom_level": [],
        "file_path": [],
    }
    flat_vecs = []

    # {'accs':accs, 'sf':sf, 'dbidx':dbidx, 'zoom_level':zoom_level}
    for item in acc:
        acc0, sf, dbidx, zl, fp = itemgetter(
            "accs", "scale_factor", "dbidx", "zoom_level", "file_path"
        )(item)
        acc0 = acc0.squeeze(0)
        acc0 = acc0.transpose((1, 2, 0))

        iis, jjs = np.meshgrid(
            np.arange(acc0.shape[0], dtype=np.int16),
            np.arange(acc0.shape[1], dtype=np.int16),
            indexing="ij",
        )
        # iis = iis.reshape(-1, acc0)
        iis = iis.reshape(-1)
        jjs = jjs.reshape(-1)
        acc0 = acc0.reshape(-1, acc0.shape[-1])
        imids = np.ones_like(iis) * dbidx
        zf = np.ones_like(iis) * (1.0 / sf)
        zl = np.ones_like(iis) * zl

        flat_acc["iis"].append(iis)
        flat_acc["jjs"].append(jjs)
        flat_acc["dbidx"].append(imids)
        flat_acc["vecs"].append(acc0)
        flat_acc["zoom_factor"].append(zf.astype("float32"))
        flat_acc["zoom_level"].append(zl.astype("int16"))
        flat_acc["file_path"].append([fp] * iis.shape[0])

    flat = {}
    for k, v in flat_acc.items():
        flat[k] = np.concatenate(v)

    vecs = flat["vecs"]
    del flat["vecs"]

    vec_meta = pd.DataFrame(flat)
    # vecs = vecs.astype('float32')
    # vecs = vecs/(np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-6)
    vec_meta = vec_meta.assign(**get_boxes(vec_meta), vectors=TensorArray(vecs))
    return vec_meta.drop(["iis", "jjs"], axis=1)


class BatchInferModel:
    def __init__(self, model, device):
        self.device = device
        self.model = model

    def __call__(self, batch):
        with torch.no_grad():
            res = []
            for tup in batch:
                im = tup["image"]
                del tup["image"]
                vectors = (
                    self.model(preprocessed_image=im.unsqueeze(0).to(self.device))
                    .to("cpu")
                    .numpy()
                )
                tup["accs"] = vectors
                res.append(tup)

        if len(res) == 0:
            return []
        else:
            return [postprocess_results(res)]


class NormalizedEmbedding(nn.Module):
    def __init__(self, emb_mod):
        super().__init__()
        self.mod = emb_mod

    def forward(self, X):
        tmp = self.mod(X)
        with torch.cuda.amp.autocast():
            return F.normalize(tmp, dim=1).type(tmp.dtype)


def trace_emb_jit(output_path):
    device = torch.device("cuda:0")
    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    ker = NormalizedEmbedding(model.visual)
    ker = ker.eval()

    example = torch.randn(
        (10, 3, 224, 224), dtype=torch.half, device=torch.device("cuda:0")
    )
    with torch.no_grad():
        jitmod = torch.jit.trace(ker, example)

    out = ker(example)
    print(out.dtype)
    jitmod.save(output_path)


def clip_loader(variant="ViT-B/32"):
    def fun(device, jit_path=None):
        if jit_path == None:
            model, _ = clip.load(variant, device=device, jit=False)
            ker = NormalizedEmbedding(model.visual)
        else:
            ker = torch.jit.load(jit_path, map_location=device)

        return ker

    return fun


class HGFaceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model.get_image_features(images)


def huggingface_loader(variant="./models/clip-vit-base-patch32"):
    ## output must be extracted and then normalized
    def fun(device, jit_path=None):
        model = transformers.CLIPModel.from_pretrained(variant)
        return HGFaceWrapper(model)

    return fun


class ImageEmbedding(nn.Module):
    def __init__(self, device, jit_path=None):
        super().__init__()
        self.device = device

        # if loader_function is None:
        #     loader_function = clip_loader()
        ker = huggingface_loader(variant=jit_path)(device)

        kernel_size = 224  # changes with variant
        self.model = SlidingWindow(
            ker, kernel_size=kernel_size, stride=kernel_size // 2, center=True
        ).to(self.device)

    def forward(self, *, preprocessed_image):
        return self.model(preprocessed_image)
