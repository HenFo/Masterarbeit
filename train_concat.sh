#!/bin/bash

TRAIN=True
TEST=True
ABLATION=True


WINDOW=12


# dataset="meld"
dataset="iemocap"
model="LLaMA2-base"

experiment="concat/$dataset/$model/mlp/audio_instruction"

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

output_path=$stage_2_path

if [ $TRAIN = True ]; then
    echo "Running stage 1"
    accelerate launch ./run_scripts/main_concat.py \
        --batch_size 2 \
        --gradient_accumulation_steps 16 \
        --llm_id $LANGUAGE_MODEL \
        --acoustic_id $ACOUSTIC_MODEL \
        --adapter_id $LORA_ADAPTER \
        --output_path $stage_1_path \
        --train_dataset $DS_TRAIN_PATH \
        --test_dataset $DS_TEST_PATH \
        --dev_dataset $DS_DEV_PATH \
        --task "normal" \
        --epochs 20 \
        --lr 2e-5 \
        --stage 1 \
        --window_size $WINDOW

    if [ $? -ne 0 ]; then
        echo "An error occurred. Terminating."
        exit 1
    fi

    output_path=$stage_1_path


    echo "Running stage 2"
    accelerate launch ./run_scripts/main_concat.py \
        --batch_size 2 \
        --gradient_accumulation_steps 16 \
        --llm_id $LANGUAGE_MODEL \
        --acoustic_id $ACOUSTIC_MODEL \
        --adapter_id $LORA_ADAPTER \
        --output_path $output_path \
        --checkpoint_path $stage_1_path \
        --train_dataset $DS_TRAIN_PATH \
        --test_dataset $DS_TEST_PATH \
        --dev_dataset $DS_DEV_PATH \
        --task "normal" \
        --epochs 20 \
        --lr 2e-5 \
        --train_llm \
        --stage 2 \
        --window_size $WINDOW \
        --lora_dim 16 \
        --lora_alpha 16 \
        --lora_dropout 0.1

    if [ $? -ne 0 ]; then
        echo "An error occurred. Terminating."
        exit 1
    fi


fi


if [ $TEST = True ]; then

    echo "Running evaluation"
    python ./run_scripts/main_concat.py \
        --evaluation True \
        --llm_id $LANGUAGE_MODEL \
        --acoustic_id $ACOUSTIC_MODEL \
        --adapter_id $LORA_ADAPTER \
        --output_path $output_path \
        --test_dataset $DS_TEST_PATH \
        --window_size $WINDOW \
        --batch_size 1

fi
