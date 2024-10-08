#!/bin/bash

TRAIN=True
TEST=True
ABLATION=True

WINDOW=12

# dataset="meld"
dataset="iemocap"
model="LLaMA2-base"

experiment_suffix="final_version"

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset)
            dataset=$2
            shift
            ;;
        --experiment_suffix)
            experiment_suffix=$2
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

experiment="merge/final/$dataset/$model/$experiment_suffix"

LANGUAGE_MODEL="/home/fock/code/MultiModalInstructERC/models/language/$model"
LORA_ADAPTER="/home/fock/code/MultiModalInstructERC/models/language/adapter/$dataset/$model"
ACOUSTIC_MODEL="/home/fock/code/MultiModalInstructERC/models/acoustic/wav2vec2/wav2vec2-large-robust-12-ft-emotion-msp-dim"
OUTPUT_PATH="/home/fock/code/MultiModalInstructERC/experiments/multimodal/$experiment/"

DS_BASE="/home/fock/code/MultiModalInstructERC/datasets/$dataset"
if [ $dataset = "meld" ]; then
    DS_TRAIN_PATH="$DS_BASE/train_sent_emo.csv"
    DS_DEV_PATH="$DS_BASE/dev_sent_emo.csv"
    DS_TEST_PATH="$DS_BASE/test_sent_emo.csv"

    s1_weight_decay=1e-2
    s1_lr=1e-5
    s1_min_lr_ratio=0.5
    s1_epochs=10

elif [ $dataset = "iemocap" ]; then
    DS_TRAIN_PATH="$DS_BASE/iemocap.csv"
    DS_DEV_PATH="$DS_BASE/iemocap.csv"
    DS_TEST_PATH="$DS_BASE/iemocap.csv"

    s1_weight_decay=5e-3
    s1_lr=3e-5
    s1_min_lr_ratio=1.0
    s1_epochs=20

else
    echo "Invalid dataset"
    exit 1
fi

stage_1_path=$OUTPUT_PATH"stage_1/"
stage_2_path=$OUTPUT_PATH"stage_2/"
stage_3_path=$OUTPUT_PATH"stage_3/"

output_path=$stage_3_path

if [ "$TRAIN" = "True" ]; then
    echo "Running stage 1"
    accelerate launch ./run_scripts/main_merge.py \
        --batch_size 8 \
        --gradient_accumulation_steps 4 \
        --llm_id $LANGUAGE_MODEL \
        --acoustic_id $ACOUSTIC_MODEL \
        --adapter_id $LORA_ADAPTER \
        --output_path $stage_1_path \
        --train_dataset $DS_TRAIN_PATH \
        --test_dataset $DS_TEST_PATH \
        --dev_dataset $DS_DEV_PATH \
        --task "normal" \
        --epochs $s1_epochs \
        --lr $s1_lr \
        --weight_decay $s1_weight_decay \
        --min_lr_ratio $s1_min_lr_ratio \
        --warmup_ratio 0.2 \
        --stage 1 \
        --window_size 1

    if [ $? -ne 0 ]; then
        echo "An error occurred. Terminating."
        exit 1
    fi

    output_path=$stage_1_path

    echo "Running stage 2"
    accelerate launch ./run_scripts/main_merge.py \
        --batch_size 2 \
        --gradient_accumulation_steps 16 \
        --llm_id $LANGUAGE_MODEL \
        --acoustic_id $ACOUSTIC_MODEL \
        --adapter_id $LORA_ADAPTER \
        --output_path $stage_2_path \
        --checkpoint_path $stage_1_path \
        --train_dataset $DS_TRAIN_PATH \
        --test_dataset $DS_TEST_PATH \
        --dev_dataset $DS_DEV_PATH \
        --task "normal" \
        --stage 2 \
        --window_size 1 \
        --train_llm \
        --epochs 20 \
        --lr 1e-5 \
        --warmup_ratio 0.3 \
        --weight_decay 5e-3 \
        --window_size 1 \
        --train_llm \
        --lora_dim 4 \
        --lora_alpha 16 \
        --lora_dropout 0.3 \
        --lora_module_name ".*?[qkv]_proj" \
        --min_lr_ratio 0.5 \
        --do_auxiliary_task \
        --time_till_aux 5 \
        --ignore_loss_till 13

    if [ $? -ne 0 ]; then
        echo "An error occurred. Terminating."
        exit 1
    fi

    output_path=$stage_2_path

    # echo "Copying adapter files from stage 2 to stage 3"
    # mkdir -p $stage_3_path
    # for file in $(ls $stage_2_path | grep "^adapter"); do
    #     cp $stage_2_path/$file $stage_3_path
    # done

    echo "Running stage 3"
    accelerate launch ./run_scripts/main_merge.py \
        --batch_size 2 \
        --gradient_accumulation_steps 16 \
        --llm_id $LANGUAGE_MODEL \
        --acoustic_id $ACOUSTIC_MODEL \
        --adapter_id $LORA_ADAPTER \
        --output_path $stage_3_path \
        --checkpoint_path $stage_2_path \
        --train_dataset $DS_TRAIN_PATH \
        --test_dataset $DS_TEST_PATH \
        --dev_dataset $DS_DEV_PATH \
        --task "normal" \
        --epochs 20 \
        --lr 1 \
        --min_lr_ratio 0.1 \
        --warmup_ratio 0.1 \
        --stage 3 \
        --window_size $WINDOW

    if [ $? -ne 0 ]; then
        echo "An error occurred. Terminating."
        exit 1
    fi

    output_path=$stage_3_path

fi

if [ "$TEST" = "True" ]; then

    echo "Running evaluation"
    python run_scripts/main_merge.py \
        --evaluation \
        --llm_id $LANGUAGE_MODEL \
        --acoustic_id $ACOUSTIC_MODEL \
        --adapter_id $LORA_ADAPTER \
        --output_path $output_path \
        --test_dataset $DS_TEST_PATH \
        --window_size $WINDOW \
        --batch_size 1

fi

if [ "$ABLATION" = "True" ]; then

    echo "Running ablation"
    # Performance of audio_only s2
    python run_scripts/main_merge.py \
        --evaluation \
        --llm_id $LANGUAGE_MODEL \
        --acoustic_id $ACOUSTIC_MODEL \
        --adapter_id $LORA_ADAPTER \
        --output_path $stage_2_path \
        --test_dataset $DS_TEST_PATH \
        --window_size 1 \
        --batch_size 1 \
        --audio_only
    
    # Performance of audio_only 23
    python run_scripts/main_merge.py \
        --evaluation \
        --llm_id $LANGUAGE_MODEL \
        --acoustic_id $ACOUSTIC_MODEL \
        --adapter_id $LORA_ADAPTER \
        --output_path $stage_3_path \
        --test_dataset $DS_TEST_PATH \
        --window_size 1 \
        --batch_size 1 \
        --audio_only
    
    # Performance before stage 3
    python run_scripts/main_merge.py \
        --evaluation \
        --llm_id $LANGUAGE_MODEL \
        --acoustic_id $ACOUSTIC_MODEL \
        --adapter_id $LORA_ADAPTER \
        --output_path $stage_2_path \
        --test_dataset $DS_TEST_PATH \
        --window_size $WINDOW \
        --batch_size 1 
fi
