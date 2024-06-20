#!/bin/bash

WINDOW=5

LANGUAGE_MODEL="/home/fock/code/MultiModalInstructERC/models/language/LLaMA2"
LORA_ADAPTER="/home/fock/code/MultiModalInstructERC/models/language/adapter/InstructERC_unbalanced"
ACOUSTIC_MODEL="/home/fock/code/MultiModalInstructERC/models/acoustic/wav2vec2/wav2vec2-large-robust-12-ft-emotion-msp-dim"
OUTPUT_PATH="/home/fock/code/MultiModalInstructERC/experiments/multimodal/mlp/concat/"

DS_TRAIN_PATH="/home/fock/code/MultiModalInstructERC/datasets/meld/train_sent_emo.csv"
DS_DEV_PATH="/home/fock/code/MultiModalInstructERC/datasets/meld/dev_sent_emo.csv"
DS_TEST_PATH="/home/fock/code/MultiModalInstructERC/datasets/meld/test_sent_emo.csv"

stage_1_path=$OUTPUT_PATH"stage_1/"
output_path=$OUTPUT_PATH"stage_2/"

accelerate launch main.py \
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
    --deepspeed_config "deepspeed_config.json" \
    --epochs 12 \
    --lr 2e-5 \
    --stage 1 \
    --window_size $WINDOW

accelerate launch main.py \
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
    --deepspeed_config "deepspeed_config.json" \
    --epochs 15 \
    --lr 2e-5 \
    --train_llm True \
    --stage 2 \
    --window_size $WINDOW

cp $stage_1_path"best_model.pth" $output_path"best_model.pth"