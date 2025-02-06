cd src/open-r1-multimodal # We edit the grpo.py and grpo_trainer.py in open-r1 repo.

# follow open-r1-multimodal to install the packages.
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_ENTITY="mmo1"
export WANDB_PROJECT=GRPO
export WANDB_API_KEY="71003e04759f7b311ea38af1141f25e78f0614a0"
export WANDB_RUN_NAME=Qwen-VL-2B-GRPO-$(date +%Y-%m-%d-%H-%M-%S)
wandb login $WANDB_API_KEY

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir checkpoints/${WANDB_RUN_NAME} \
    --model_name_or_path /mnt/zhangh/sicong/mmo1/checkpoints/Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name /mnt/zhangh/sicong/mmo1/GRPO_MM/data/lmms-lab/multimodal-open-r1-8k-verified \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name $WANDB_RUN_NAME \
    --save_steps 100 \
    --save_only_model true
