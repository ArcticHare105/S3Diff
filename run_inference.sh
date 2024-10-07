accelerate launch --num_processes=1 --gpu_ids="0," --main_process_port 29300 src/inference_s3diff.py \
    --sd_path="path_to_checkpoints/sd-turbo" \
    --de_net_path="assets/mm-realsr/de_net.pth" \
    --pretrained_path="path_to_checkpoints_folder/model_30001.pkl" \
    --output_dir="./output" \
    --ref_path="path_to_ground_truth_folder" \
    --align_method="wavelet"
