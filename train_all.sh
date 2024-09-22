#!/bin/bash

experiment_suffix="final_version"

for ds in "iemocap" "meld"; do
    ./train_concat.sh --dataset ${ds} --experiment_suffix ${experiment_suffix}
    ./train_late_fusion.sh --dataset ${ds} --experiment_suffix ${experiment_suffix}
    ./train_merge.sh --dataset ${ds} --experiment_suffix ${experiment_suffix}
done
