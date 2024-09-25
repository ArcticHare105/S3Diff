accelerate launch --num_processes=4 --gpu_ids="0,1,2,3" --main_process_port 29300 src/train_s3diff.py \
    --sd_path="path_to_checkpoints/sd-turbo" \
    --de_net_path="assets/mm-realsr/de_net.pth" \
    --output_dir="./output" \
    --resolution=512 \
    --train_batch_size=4 \
    --enable_xformers_memory_efficient_attention \
    --viz_freq 25
