#!/bin/bash
#SBATCH --partition=gpu_min12gb
#SBATCH --qos=gpu_min12gb
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=baseline_densenet201.out
#SBATCH --error=baseline_densenet201.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Multi-modal B-cos Networks"
# echo "Training Catalogue Type: E"
# python src/main_multimodal.py \
#  --gpu_id 0 \
#  --config_json 'config/multimodal/E/baseline_densenet201.json' \
#  --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
#  --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-12-14_20-33-56/bin/model_final.pt' \
#  --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E'
# echo "Finished"
echo "Testing Catalogue Type: E"
python src/main_multimodal.py \
 --gpu_id 0 \
 --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
 --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-12-14_20-33-56/bin/model_final.pt' \
 --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-12-22_22-05-05/' \
 --train_or_test 'test' \
 --verbose
echo "Finished"

# echo "Training Catalogue Type: F"
# python src/main_multimodal.py \
#  --gpu_id 0 \
#  --config_json 'config/multimodal/F/baseline_densenet201.json' \
#  --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
#  --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-12-15_19-46-09/bin/model_final.pt' \
#  --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F'
# echo "Finished"
echo "Testing Catalogue Type: F"
python src/main_multimodal.py \
 --gpu_id 0 \
 --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
 --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-12-15_19-46-09/bin/model_final.pt' \
 --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-12-22_22-13-12/' \
 --train_or_test 'test' \
 --verbose
echo "Finished"