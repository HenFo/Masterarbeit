#!/bin/bash

TEST_ONLY=True

WINDOW=10

experiment="mlp/concat/interpolate"

LANGUAGE_MODEL="/home/fock/code/MultiModalInstructERC/models/language/LLaMA2"
LORA_ADAPTER="/home/fock/code/MultiModalInstructERC/models/language/adapter/InstructERC_unbalanced"
ACOUSTIC_MODEL="/home/fock/code/MultiModalInstructERC/models/acoustic/wav2vec2/wav2vec2-large-robust-12-ft-emotion-msp-dim"
OUTPUT_PATH="/home/fock/code/MultiModalInstructERC/experiments/multimodal/"$experiment"/"

DS_TRAIN_PATH="/home/fock/code/MultiModalInstructERC/datasets/meld/train_sent_emo.csv"
DS_DEV_PATH="/home/fock/code/MultiModalInstructERC/datasets/meld/dev_sent_emo.csv"
DS_TEST_PATH="/home/fock/code/MultiModalInstructERC/datasets/meld/test_sent_emo.csv"

stage_1_path=$OUTPUT_PATH"stage_1/"
output_path=$OUTPUT_PATH"stage_2/"

if [ $TEST_ONLY = False ]; then
    # echo "Running stage 1"
    # accelerate launch ./run_scripts/main_concat.py \
    #     --batch_size 2 \
    #     --gradient_accumulation_steps 16 \
    #     --llm_id $LANGUAGE_MODEL \
    #     --acoustic_id $ACOUSTIC_MODEL \
    #     --adapter_id $LORA_ADAPTER \
    #     --output_path $stage_1_path \
    #     --train_dataset $DS_TRAIN_PATH \
    #     --test_dataset $DS_TEST_PATH \
    #     --dev_dataset $DS_DEV_PATH \
    #     --task "normal" \
    #     --deepspeed_config "deepspeed_config.json" \
    #     --epochs 20 \
    #     --time_till_aux 7 \
    #     --lr 2e-5 \
    #     --stage 1 \
    #     --window_size $WINDOW

    # if [ $? -ne 0 ]; then
    #     echo "An error occurred. Terminating."
    #     exit 1
    # fi
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
        --deepspeed_config "deepspeed_config.json" \
        --epochs 20 \
        --lr 2e-5 \
        --train_llm \
        --stage 2 \
        --window_size $WINDOW \
        --lora_dim 32 \
        --lora_alpha 32 \
        --lora_dropout 0.1 
        # --do_auxilary_task \
        # --time_till_aux 10 \
        # --include_target_text_percentage_decay 0.3

    if [ $? -ne 0 ]; then
        echo "An error occurred. Terminating."
        exit 1
    fi
    
    cp $stage_1_path"best_model.pth" $output_path"best_model.pth"


fi

echo "Running evaluation"
python ./run_scripts/main_concat.py \
    --evaluation True \
    --llm_id $LANGUAGE_MODEL \
    --acoustic_id $ACOUSTIC_MODEL \
    --adapter_id $LORA_ADAPTER \
    --output_path $stage_1_path \
    --test_dataset $DS_TEST_PATH \
    --window_size $WINDOW \
    --batch_size 1

