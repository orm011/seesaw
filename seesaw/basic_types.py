from pydantic import BaseModel
from typing import Optional, List

class Box(BaseModel):
    x1 : float
    y1 : float
    x2 : float
    y2 : float
    description: Optional[str]
    marked_accepted = False

# TODO switch annotations to this class 
# make box only about boundaries
class Annotation(BaseModel):
    box : Box
    description: Optional[str]
    marked_accepted = False

class ActivationData(BaseModel):
    box : Box
    score : float

class Imdata(BaseModel):
    url : str
    dbidx : int
    boxes : Optional[List[Box]] # None means not labelled (neutral). [] means positively no boxes.
    activations : Optional[List[ActivationData]]
    marked_accepted = False

def is_accepted(imdata : Imdata):
  return any(map(lambda box : box.marked_accepted, imdata.boxes))

class IndexSpec(BaseModel):
    d_name:str 
    i_name:str
    m_name:Optional[str]
    c_name:Optional[str] # ground truth category (for lvis benchmark)

class SessionParams(BaseModel):
    index_spec : IndexSpec
    interactive : str
    method_config : Optional[dict] # changes from method to method (interactive)
    warm_start : str
    batch_size : int
    minibatch_size : int
    learning_rate : float
    max_examples : int
    loss_margin : float
    num_epochs : int
    model_type : str

class SessionState(BaseModel):
    params : SessionParams
    gdata : List[List[Imdata]]
    timing : List[float]
    reference_categories : List[str]

class BenchParams(BaseModel):
    name : str
    ground_truth_category : str
    qstr : str
    provide_textual_feedback : bool = False
    n_batches : int # max number of batches to run
    max_results : int # stop when this numbrer of results is found
    max_feedback : Optional[int]
    box_drop_prob : float

class BenchResult(BaseModel):
    nimages: int
    ntotal: int
    session: SessionState
    run_info : dict
    total_time : float

class BenchSummary(BaseModel):
    bench_params : BenchParams
    session_params : SessionParams
    timestamp : str
    result : Optional[BenchResult]