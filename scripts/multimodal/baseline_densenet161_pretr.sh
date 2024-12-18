#!/bin/bash
#SBATCH --partition=gpu_min12gb
#SBATCH --qos=gpu_min12gb
#SBATCH --job-name=cind_breloai_bcos_ret
#SBATCH --output=baseline_densenet161_pretr.out
#SBATCH --error=baseline_densenet161_pretr.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Multi-modal B-cos Networks"
echo "Training Catalogue Type: E"
python src/main_multimodal.py \
 --gpu_id 0 \
 --config_json 'config/multimodal/E/baseline_densenet161_pretr.json' \
 --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
 --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-12-08_19-18-32/bin/model_final.pt' \
 --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E'
echo "Finished"
# echo "Testing Catalogue Type: E"
# python src/main_multimodal.py \
#  --gpu_id 0 \
#  --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
#  --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-11-16_22-11-38/bin/model_final.pt' \
#  --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-11-18_00-33-23/' \
#  --train_or_test 'test' \
#  --verbose
# echo "Finished"

echo "Training Catalogue Type: F"
python src/main_multimodal.py \
 --gpu_id 0 \
 --config_json 'config/multimodal/F/baseline_densenet161_pretr.json' \
 --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
 --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-12-09_20-08-57/bin/model_final.pt' \
 --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F'
echo "Finished"
# echo "Testing Catalogue Type: F"
# python src/main_multimodal.py \
#  --gpu_id 0 \
#  --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
#  --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-11-17_04-54-50/bin/model_final.pt' \
#  --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-11-18_00-40-38/' \
#  --train_or_test 'test' \
#  --verbose
# echo "Finished"