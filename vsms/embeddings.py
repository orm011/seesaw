
import torchvision
import torch
import pandas as pd
import os
import numpy as np
import PIL.Image
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import torchvision.transforms as T
import pyroaring as pr
import typing
from sentence_transformers import SentenceTransformer
from .cross_modal_embedding import TextImageCrossModal


import clip

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = torchvision.models.resnet50(pretrained=True)
        del self.resnet.fc
        del self.resnet.avgpool
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.eval()  # set to eval.

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.pooling(x)
        return x

class XEmbedding(object):
    """
    Common interface to operator that maps a
    vectors from (bert) text embedding space to vectors in
    resnet embedding space
    """

    def __init__(self, model : TextImageCrossModal,
                 language_model, image_model, image_tx, image_vec_mean):
        self.image_tx = image_tx
        self.language_model = language_model
        self.image_model = image_model.eval()
        self.translator = model.eval().to('cpu')
        # self.string_vec_mean = torch.from_numpy(string_vec_mean).reshape(1,-1) # not used right now
        self.image_vec_mean = torch.from_numpy(image_vec_mean).reshape(1,-1)

    def from_string(self, string=None, str_vec=None, numpy=True):
        with torch.no_grad():
            if string is not None:
                str_vec = self.language_model.encode([string], convert_to_tensor=True)
            elif isinstance(str_vec, np.ndarray):
                str_vec = torch.from_numpy(str_vec)
                if len(str_vec.shape) == 1:  # single vector
                    str_vec = str_vec.view(1, -1)

            # str_vec = str_vec - self.string_vec_mean # not done during training for the already embedded strings..
            return self.translator.from_string_vec(str_vec)

    def from_image(self, image=None, img_vec=None, numpy=True):
        with torch.no_grad():
            if image is not None:
                txim = self.image_tx(image)
                img_vec = self.image_model(txim.unsqueeze(0)).squeeze().unsqueeze(0)
                # breakpoint()
            elif isinstance(img_vec, np.ndarray):
                img_vec = torch.from_numpy(img_vec)
                if len(img_vec.shape) == 1:
                    img_vec = img_vec.view(1, -1)

            img_vec = img_vec - self.image_vec_mean
            return self.translator.from_image_vec(img_vec)

    def from_raw(self, data: typing.Union[str, PIL.Image.Image]):
        if isinstance(data, str):
            return self.from_string(string=data)
        elif isinstance(data,PIL.Image.Image):
            return self.from_image(image=data)
        else:
            assert False


class VariableSizeNet(nn.Module):
    def __init__(self, base : clip.model.ModifiedResNet, **kwargs):
        super().__init__()
        assert isinstance(base, clip.model.ModifiedResNet), 'maybe use mod.visual?'
        self.base = base
        self.attnpool = ManualPooling(base.attnpool, kernel_size=7, **kwargs)
                
    def forward(self, x):
        base = self.base
        
        def stem(self, x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(base.conv1.weight.dtype)
        x = stem(base, x)
        x = base.layer1(x)
        x = base.layer2(x)
        x = base.layer3(x)
        x = base.layer4(x)
        x = self.attnpool(x)

        return x

class ClipFeatureExtractor(nn.Module):
    def __init__(self, clip_model, device:str, resize_px : int, square_crop:bool, pool_stride:int):
        super().__init__()
        self.preproc = make_clip_transform(resize_px, square_crop=square_crop)
        extractor = VariableSizeNet(clip_model.visual, stride=pool_stride, center=True)
        self.extractor = extractor.to(device)
        self.device = device
        
    def forward(self, tensor):
        return self.extractor(tensor)
        
    def process(self, image):
        with torch.no_grad():
            tensor = self.preproc(image).to(self.device).unsqueeze(0)
            return self.extractor.eval()(tensor).cpu()

def load_embedding_model() -> XEmbedding :
    resnet50 = ResNetFeatureExtractor()
    distilbert = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    model = TextImageCrossModal(caption_vec_size=768,  # ,train_cap_embeddings.shape[1],
                                image_vec_size=2048,  # train_img_embeddings.shape[1],
                                n_keys=2000, cm_val=None)
    d2 = torch.load('./data/lookup_model_2k_capacity_image_emb.pth')
    model.load_state_dict(d2)

    tx = T.Compose([
        T.Lambda(lambda x: x.convert(mode='RGB')),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    caption_embeddings_mean = np.load('./data/caption_embeddings_mean.npy')
    image_embeddings_mean = np.load('./data/image_embeddings_mean.npy')

    embedding = XEmbedding(model=model,
                           language_model=distilbert,
                           image_model=resnet50,
                           image_tx=tx,
                           image_vec_mean=image_embeddings_mean)
    return embedding

def load_clip_embedding(variant, tx, device='cpu', jit=False):
    assert variant in ["ViT-B/32"]
    print('ignoring tx, jit arguments')
    return CLIPWrapper(device=device)

class ManualPooling(nn.Module):
    def __init__(self, kernel, kernel_size, stride=None, center=False):
        super().__init__()
        self.kernel = kernel
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.center = center

    def forward(self, tensor):
        vecs = tensor
        width_size=self.kernel_size
        layer = self.kernel
        stride_size = self.stride
        center = self.center

        # assume vecs are *,H,W
        h,w = vecs.shape[-2:]

        ## if we are forced to cut part off, lets center where we take the pools
        iis = list(range(0,h-width_size+1,stride_size))
        jjs = list(range(0,w-width_size+1,stride_size))

        if center:
            offset_h = (h - (iis[-1] + width_size))//2
            offset_w = (w - (jjs[-1] + width_size))//2
        else:
            offset_h = 0
            offset_w = 0

        output = []
        for ii in iis:
            for jj in jjs:
                cut = vecs[...,
                           ii + offset_h:ii + width_size + offset_h,
                           jj + offset_w:jj + width_size + offset_w]
                assert cut.shape[-2] == width_size
                assert cut.shape[-1] == width_size
                out = layer(cut)
                output.append(out)

        v = torch.stack(output, dim=-1)
        v = v.reshape(output[0].shape + (len(iis), len(jjs),))
        return v

def test_pooling():
    tsizes = [7,8,9,11,12,13,16]
    veclist = []
    for i in tsizes:
        for j in tsizes:
            veclist.append(torch.randn(1,3,i,j))

    assert len(veclist) > 0
    kernel_sizes = [6,7]
    strides = [None, 2,3,4]
    
    avg_pool = lambda x : nn.AdaptiveAvgPool2d(1)(x).squeeze(-1).squeeze(-1)
    for v in veclist:
        for stride in strides:
            for ks in kernel_sizes:
                reference_pooling = nn.AvgPool2d(kernel_size=ks, stride=stride)
                test_pooling = ManualPooling(kernel=avg_pool, 
                                              kernel_size=ks, stride=stride, center=False)
                center_test = ManualPooling(kernel=avg_pool, 
                                              kernel_size=ks, stride=stride, center=True)

                target = reference_pooling(v)
                test = test_pooling(v) 
                center = center_test(v)
                assert test.shape == center.shape
                assert test.shape == target.shape
                assert torch.isclose(test, target, atol=1e-6).all()    
test_pooling()


import torchvision.transforms as T
class ImTransform(object):
    def __init__(self, visual_xforms, tensor_xforms):
        self.visual_tx = T.Compose(visual_xforms)
        self.tensor_tx = T.Compose(tensor_xforms)
        self.full = T.Compose([self.visual_tx, self.tensor_tx])
            
    def __call__(self, img):
        return self.full(img)
    
    
def make_clip_transform(n_px, square_crop=False):
    maybe_crop = [T.CenterCrop(n_px)] if square_crop else []
    return ImTransform(visual_xforms=[T.Resize(n_px, interpolation=PIL.Image.BICUBIC)]
                                    + maybe_crop + [lambda image: image.convert("RGB")],
                       tensor_xforms=[T.ToTensor(),
                                      T.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                                (0.26862954, 0.26130258, 0.27577711))])

class CLIPWrapper(XEmbedding):
    def __init__(self, device):
        tx = make_clip_transform(n_px=224, square_crop=False)
        tx2 = T.Compose([tx, lambda x : x.type(torch.float16)])
        variant = "ViT-B/32"
        kernel_size = 224 # changes with variant
        assert variant in ["ViT-B/32", "RN50"]
        model, _ = clip.load(variant, device=device,  jit=True)
        self.model = model.eval()
        self.preprocess = tx
        self.device = device
        self.pooled_model = nn.Sequential(ManualPooling(model.visual.eval(),
                     kernel_size=kernel_size, 
                     stride=kernel_size//2, 
                     center=True),nn.AdaptiveAvgPool2d(1)).eval()

    def from_string(self, *, string=None, str_vec=None, numpy=True):
        if str_vec is not None:
            return str_vec
        else:
            with torch.no_grad():
                text = clip.tokenize([string]).to(self.device)
                text_features = self.model.encode_text(text)
                return text_features.cpu().numpy().reshape(1,-1)

    def from_image(self, *, preprocessed_image=None, image=None, img_vec=None, numpy=True):
        if img_vec is not None:
            return img_vec
        elif (image is not None) or (preprocessed_image is not None):
            if image is not None:
                tensor = self.preprocess(image)
            elif preprocessed_image is not None:
                tensor = preprocessed_image

            with torch.no_grad():
                image_features = self.pooled_model(tensor.unsqueeze(0).to(self.device))
                return image_features.cpu().numpy().reshape(1,-1)

