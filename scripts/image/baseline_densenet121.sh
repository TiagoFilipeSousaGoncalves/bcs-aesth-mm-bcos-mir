#!/bin/bash
#SBATCH --partition=gpu_min24gb
#SBATCH --qos=gpu_min24gb
#SBATCH --job-name=cind_breloai_bc_ret
#SBATCH --output=baseline_densenet121.out
#SBATCH --error=baseline_densenet121.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Bcos Networks"
# echo "Training Catalogue Type: E"
# python src/main_image.py \
#  --gpu_id 0 \
#  --config_json 'config/image/E/baseline_densenet121.json' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
#  --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E' \
#  --train_or_test 'train'
# echo "Finished"
echo "Testing Catalogue Type: E"
python src/main_image.py \
 --gpu_id 0 \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
 --train_or_test 'test' \
 --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-12-06_17-50-15/' \
 --verbose
echo "Finished"

# echo "Training Catalogue Type: F"
# python src/main_image.py \
#  --gpu_id 0 \
#  --config_json 'config/image/F/baseline_densenet121.json' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
#  --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F' \
#  --train_or_test 'train'
# echo "Finished"
echo "Testing Catalogue Type: F"
python src/main_image.py \
 --gpu_id 0 \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
 --train_or_test 'test' \
 --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-12-07_09-28-23/' \
 --verbose
echo "Finished"