export HF_HOME=/scratch/dkhasha1/bzhang90/huggingface
export HF_DATASETS_CACHE=/scratch/dkhasha1/bzhang90/huggingface/datasets
export HF_HUB_CACHE=/scratch/dkhasha1/bzhang90/huggingface/hub

CUDA_VISIBLE_DEVICES=0,1 python label_stock_news_meaningful.py \
  --model_name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --root_dir /scratch/dkhasha1/bzhang90/data_mining_fall_2025/hourly_output_test \
  --non_meaningful_json /scratch/dkhasha1/bzhang90/data_mining_fall_2025/non_meaningful_test_files_report.json \
  --meaningful_output_dir /scratch/dkhasha1/bzhang90/data_mining_fall_2025/hourly_output_test_meaningful \
  --batch_size 1000
