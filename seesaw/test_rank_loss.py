from seesaw.rank_loss import quick_pairwise_gradient, ref_signed_inversions, ref_pairwise_rank_loss, ref_pairwise_rank_loss_gradient
import torch

## run this with pytest  -vvl  ./seesaw/seesaw/test_rank_loss.py

_test_cases = [

    # # corner case empty array
    dict(target = [],
         scores = [],
         margin = 10.,
         inversions = [],
         rank_loss = [],
         gradient = [],
    ),

    # corner case 1 elt
    dict(target = [1],
         scores = [.5],
         margin = 1.,
         inversions = [0],
         rank_loss = [0.],
         gradient = [0.],
    ),

    #   correct basic pair
    dict(target = [0., 1.],
         scores = [.1, .2],
         margin = 0.,
         inversions = torch.zeros((2,2)),
         rank_loss = torch.zeros(2),
         gradient = torch.zeros(2),
    ),

    # correct basic pair with no margin violation
    dict(target = [0., 1.],
         scores = [.09, .2],
         margin = 0.1,
         inversions = torch.zeros((2,2)),
         rank_loss = torch.zeros(2),
         gradient = torch.zeros(2),
    ),

    # correct basic pair in different input order
    dict(target = [2., 1.],
         scores = [.2, .09],
         margin = 0.1,
         inversions = torch.zeros((2,2)),
         rank_loss = torch.zeros(2),
         gradient = torch.zeros(2),
    ),

    # correct basic pair with margin violation
    dict(target = [0., 1.],
         scores = [.11, .2],
         margin = .1,
         inversions = [[0, -1], # need to increase score
                        [1., 0]
         ],
         rank_loss = [[0, .01], # small loss after taking maring off.01
                      [.01, 0.]
         ],
         gradient = [1., -1.], 
    ),

    # basic pair at margin boundary : want gradient to separate them
    dict(target = [0., 1.],
         scores = [.1, .2],
         margin = .1, # note margin. want to make it larger than margin so that 
         # when margin is 0, values are still not equal
         inversions = [[0, -1], # need to increase score
                        [1., 0]
         ],
         rank_loss = [[0, .0], # loss is technically zero
                      [.0, 0.]
         ],
         gradient = [1., -1.],  # but dont want points to get stuck here 
    ),

    # inversion
    dict(target = [0., 1.],
         scores = [.2, .1],
         margin = 0.,
         inversions = [[0, -1],   # negative when element is scored lower than it should be
                        [1, 0.]], # positive when element is scored higher than it should be
         rank_loss = [[0, .1],  
                      [.1, 0]], # rank loss is always positive.

         gradient = [1, -1.], # decrease score for larger one, increase score for smaller one
    ),

    # same inversion but inputs in different order
    dict(target = [ 1., 0.],
         scores = [.1, .2],
         margin = 0.,
         inversions = [[0, 1],   # negative when element is scored lower than it should be
                        [-1, 0.]], # positive when element is scored higher than it should be
         rank_loss = [[0, .1],  
                      [.1, 0]], # rank loss is always positive.

         gradient = [-1, 1.], # same output but in different order
    ),

    # inversion in pair with an extra margin issue
    dict(target = [0., 1.],
         scores = [.2, .1],
         margin = .11,
         inversions = [[0, -1], 
                        [1, 0.]],
         rank_loss = [[0, .21],  # loss is bigger
                      [.21, 0]],
         gradient = [1, -1.], # gradient stays the same
    ),

    # repeated target value with different scores
    # no losses even if margin
    dict(target = [1., 1.],
         scores = [.1, .2],
         margin = .11,
         inversions = torch.zeros((2,2)),
         rank_loss = torch.zeros(2),
         gradient = torch.zeros(2),
    ),

    # repeated target value different order
    # also no losses
    dict(target = [1., 1.],
         scores = [.2, .1],
         margin = .11,
         inversions = torch.zeros((2,2)),
         rank_loss = torch.zeros(2),
         gradient = torch.zeros(2),
    ),

    # same scores, zero margin, still violation
    dict(target = [1., 2.],
         scores = [.1, .1],
         margin = 0.,
         inversions = [[0, -1], 
                        [1, 0.]],
         rank_loss = [[0, .0],  # loss value is zero here
                      [.0, 0]],
         gradient = [1, -1.], # despite loss being right at 0, we should make the first score lower, second one higher 
    ),

    # full inversion of array. loss should reflect
    dict(target = [1., 2., 3.],
         scores = [.3, .2, .1],
         margin = 0.,
         inversions = [[0,  -1, -1], 
                        [1, 0., -1],
                        [1, 1, 0]],
         rank_loss = [[0, .1,  .2],  # loss is zero here, but we are at the boundary
                      [.1, .0, .1],
                      [.2, .1, 0.]
                      ],

         gradient = [2, 0., -2.], # note middle vector has zero gradient
    ),

    # mixed case with duplicate targets
    dict(target = [0, 0, 1, 1],
         scores = [0,.2,.1,.3],
         margin = 0.,
         inversions = [ [0,  0,  0, 0], 
                        [0,  0, -1, 0],
                        [0,  1,  0, 0],
                        [0,  0,  0, 0]
                    ],
         rank_loss = [[0,  0,  0, 0],  # loss is zero here, but we are at the boundary
                      [0,  0, .1, 0],
                      [0, .1,  0, 0],
                      [0,  0,  0, 0]
                      ],

         gradient = [0, 1., -1., 0.], # second vector has zero gradient, but other two have larger
    ),

    # degenerate score case
    dict( target = [0, 1, 2],
            scores = [0, 0, 0],
            margin = 0,
            inversions = [
                    [0, -1, -1],
                    [1, 0, -1],
                    [1, 1, 0],
            ],
            rank_loss = torch.zeros((3,3)),
            gradient = [2, 0, -2],
    )
]

def get_test_cases():
    ret = []
    for d in _test_cases:
        d2 = {k:(v.float() if torch.is_tensor(v) else torch.tensor(v).float())
                 for (k,v) in d.items() }
        ret.append(d2)
    return ret
    
def test_ref_inversions():
    for test in get_test_cases():
        computed = ref_signed_inversions(test['target'], scores=test['scores'], margin=test['margin'])
        expected = test['inversions']
        assert (computed == expected).all(), f'{computed=} {expected=}'

def test_ref_pairwise_loss():
    for test in get_test_cases():
        computed = ref_pairwise_rank_loss(test['target'], scores=test['scores'], margin=test['margin'])
        expected = test['rank_loss']
        assert torch.isclose(computed, expected).all(), f'{computed=} {expected=}'

def test_ref_pairwise_rank_loss_gradient():
    for test in get_test_cases():
        computed = ref_pairwise_rank_loss_gradient(test['target'], scores=test['scores'], margin=test['margin'])
        expected = test['gradient']
        assert torch.isclose(computed, expected).all(), f'{computed=} {expected=}'


def test_quick_pairwise_rank_loss_gradient_zero_margin():
    for test in get_test_cases():
        if test['margin'] == 0:
            computed = quick_pairwise_gradient(test['target'], scores=test['scores'], margin=test['margin'])
            expected = test['gradient']
            assert torch.isclose(computed, expected).all(), f'{computed=} {expected=}'
