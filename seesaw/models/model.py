import torch.nn as nn
import torch.nn.functional as F
import torch
import transformers
from .embeddings import SlidingWindow
from ..definitions import resolve_path


class NormalizedEmbedding(nn.Module):
    def __init__(self, emb_mod):
        super().__init__()
        self.mod = emb_mod

    def forward(self, X):
        tmp = self.mod(X)
        with torch.cuda.amp.autocast():
            return F.normalize(tmp, dim=1).type(tmp.dtype)

def trace_emb_jit(output_path):
    import clip
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
    import clip
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
        feats = self.model.get_image_features(images)
        return F.normalize(feats, dim=1).type(feats.dtype)

def huggingface_loader(variant):
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
        path = resolve_path(jit_path)
        ker = huggingface_loader(variant=path)(device)

        kernel_size = 224  # changes with variant
        self.model = SlidingWindow(
            ker, kernel_size=kernel_size, stride=kernel_size // 2, center=True
        ).to(self.device)

    def forward(self, *, preprocessed_image):
        return self.model(preprocessed_image)
