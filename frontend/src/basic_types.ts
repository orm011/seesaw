/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface ActivationData {
  box: Box;
  score: number;
}
export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  description?: string;
  marked_accepted?: boolean;
}
export interface Annotation {
  box: Box;
  description?: string;
  marked_accepted?: boolean;
}
export interface BenchParams {
  name: string;
  ground_truth_category: string;
  qstr: string;
  provide_textual_feedback?: boolean;
  n_batches: number;
  max_results: number;
  max_feedback?: number;
  box_drop_prob: number;
}
export interface BenchResult {
  nimages: number;
  ntotal: number;
  session: SessionState;
  run_info: {
    [k: string]: unknown;
  };
  total_time: number;
}
export interface SessionState {
  params: SessionParams;
  gdata: Imdata[][];
  timing: number[];
  reference_categories: string[];
  query_string?: string;
}
export interface SessionParams {
  index_spec: IndexSpec;
  interactive: string;
  batch_size: number;
  agg_method?: string;
  shortlist_size?: number;
  method_config?: {
    [k: string]: unknown;
  };
  image_vector_strategy?: string;
}
export interface IndexSpec {
  d_name: string;
  i_name: string;
  c_name?: string;
}
export interface Imdata {
  url: string;
  dbidx: number;
  boxes?: Box[];
  activations?: ActivationData[];
}
export interface BenchSummary {
  bench_params: BenchParams;
  session_params: SessionParams;
  timestamp: string;
  result?: BenchResult;
}
