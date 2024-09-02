#!/bin/bash

TEST_ONLY=False

WINDOW=12


# dataset="meld"
dataset="iemocap"
model="LLaMA2-base"

experiment="mlp/late_fusion/$dataset/$model/text_only"

LANGUAGE_MODEL="/home/fock/code/MultiModalInstructERC/models/language/$model"
LORA_ADAPTER="/home/fock/code/MultiModalInstructERC/models/language/adapter/$dataset/$model"
ACOUSTIC_MODEL="/home/fock/code/MultiModalInstructERC/models/acoustic/wav2vec2/wav2vec2-large-robust-12-ft-emotion-msp-dim"
OUTPUT_PATH="/home/fock/code/MultiModalInstructERC/experiments/multimodal/$experiment/"

DS_BASE="/home/fock/code/MultiModalInstructERC/datasets/$dataset"
if [ $dataset = "meld" ]; then
    DS_TRAIN_PATH="$DS_BASE/train_sent_emo.csv"
    DS_DEV_PATH="$DS_BASE/dev_sent_emo.csv"
    DS_TEST_PATH="$DS_BASE/test_sent_emo.csv"

elif [ $dataset = "iemocap" ]; then
    DS_TRAIN_PATH="$DS_BASE/iemocap.csv"
    DS_DEV_PATH="$DS_BASE/iemocap.csv"
    DS_TEST_PATH="$DS_BASE/iemocap.csv"

else
    echo "Invalid dataset"
    exit 1
fi


stage_1_path=$OUTPUT_PATH"stage_1/"
stage_2_path=$OUTPUT_PATH"stage_2/"
stage_3_path=$OUTPUT_PATH"stage_3/"

output_path=$stage_1_path

if [ $TEST_ONLY = False ]; then
    echo "Running stage 1"
    accelerate launch ./run_scripts/main_late_fusion.py \
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
        --epochs 4 \
        --lr 1e-5 \
        --warmup_ratio 0.05 \
        --weight_decay 0.0 \
        --min_lr_ratio 0.2 \
        --stage 1 \
        --window_size $WINDOW 
        # --resume_training \


    if [ $? -ne 0 ]; then
        echo "An error occurred. Terminating."
        exit 1
    fi

    output_path=$stage_1_path

    cp $stage_1_path"best_model.pth" $stage_2_path

    echo "Running stage 2"
    accelerate launch ./run_scripts/main_late_fusion.py \
        --batch_size 8 \
        --gradient_accumulation_steps 4 \
        --llm_id $LANGUAGE_MODEL \
        --acoustic_id $ACOUSTIC_MODEL \
        --adapter_id $LORA_ADAPTER \
        --output_path $stage_2_path \
        --checkpoint_path $stage_1_path \
        --train_dataset $DS_TRAIN_PATH \
        --test_dataset $DS_TEST_PATH \
        --dev_dataset $DS_DEV_PATH \
        --task "normal" \
        --epochs 15 \
        --lr 1e-4 \
        --min_lr_ratio 0.2 \
        --warmup_ratio 0.1 \
        --weight_decay 0.0 \
        --stage 2 \
        --window_size $WINDOW \

    if [ $? -ne 0 ]; then
        echo "An error occurred. Terminating."
        exit 1
    fi

    output_path=$stage_2_path

    cp $stage_2_path"best_model.pth" $stage_3_path
    
    
    echo "Running stage 3"
    accelerate launch ./run_scripts/main_late_fusion.py \
        --batch_size 8 \
        --gradient_accumulation_steps 4 \
        --llm_id $LANGUAGE_MODEL \
        --acoustic_id $ACOUSTIC_MODEL \
        --adapter_id $LORA_ADAPTER \
        --output_path $stage_3_path \
        --checkpoint_path $stage_2_path \
        --train_dataset $DS_TRAIN_PATH \
        --test_dataset $DS_TEST_PATH \
        --dev_dataset $DS_DEV_PATH \
        --task "normal" \
        --epochs 15 \
        --lr 1e-5 \
        --min_lr_ratio 0.2 \
        --warmup_ratio 0.1 \
        --weight_decay 0.0 \
        --stage 3 \
        --window_size $WINDOW \

    if [ $? -ne 0 ]; then
        echo "An error occurred. Terminating."
        exit 1
    fi

    output_path=$stage_3_path


fi

echo "Running evaluation"
python run_scripts/main_late_fusion.py \
    --evaluation \
    --llm_id $LANGUAGE_MODEL \
    --acoustic_id $ACOUSTIC_MODEL \
    --adapter_id $LORA_ADAPTER \
    --output_path $output_path \
    --test_dataset $DS_TEST_PATH \
    --dev_dataset $DS_DEV_PATH \
    --window_size $WINDOW \
    --batch_size 8
