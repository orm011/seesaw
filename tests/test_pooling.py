import torch
import torch.nn as nn
from seesaw.embeddings import ManualPooling, SlidingWindow

def check_pooling(pm):
    tsizes = [7,8,9,11,12,13,16]
    veclist = []
    for i in tsizes:
        for j in tsizes:
            veclist.append(torch.randn(1,3,i,j))

    assert len(veclist) > 0
    kernel_sizes = [6,7]
    # strides = [None, 2,3,4]
    avg_pool = lambda x : nn.AdaptiveAvgPool2d(1)(x).squeeze(-1).squeeze(-1)
    for v in veclist:
        for ks in kernel_sizes:
            stride = ks//2
            reference_pooling = nn.AvgPool2d(kernel_size=ks, stride=stride)
            test_pooling = pm(kernel=avg_pool, 
                                            kernel_size=ks, stride=stride, center=False)
            center_test = pm(kernel=avg_pool, 
                                            kernel_size=ks, stride=stride, center=True)

            target = reference_pooling(v)
            test = test_pooling(v) 
            center = center_test(v)
            assert test.shape == center.shape, f'{test.shape} {center.shape}'
            assert test.shape == target.shape, f'{test.shape} {target.shape}'
            assert torch.isclose(test, target, atol=1e-6).all()    

check_pooling(ManualPooling)
check_pooling(SlidingWindow)