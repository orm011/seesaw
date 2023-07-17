mamba activate seesaw

CONFIG=~/seesaw/scripts/configs/pseudo_label_lr.yaml

#    --dryrun --dryrun_max_iter=5 \
python ./seesaw/scripts/run_bench.py \
    --num_cpus=8 --mem_gbs=16 \
    --output_dir ~/test_new_trainer \
    --root_dir ~/fastai_shared/omoll/seesaw_root2 \
    $CONFIG