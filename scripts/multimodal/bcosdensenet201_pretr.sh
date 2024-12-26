#!/bin/bash
#SBATCH --partition=gpu_min32gb
#SBATCH --qos=gpu_min32gb
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=bcosdensenet201_pretr.out
#SBATCH --error=bcosdensenet201_pretr.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Multi-modal B-cos Networks"
echo "Training Catalogue Type: E"
python src/main_multimodal.py \
 --gpu_id 0 \
 --config_json 'config/multimodal/E/bcosdensenet201_pretr.json' \
 --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
 --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-12-15_10-38-06/bin/model_final.pt' \
 --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E'
echo "Finished"
# echo "Testing Catalogue Type: E"
# python src/main_multimodal.py \
#  --gpu_id 0 \
#  --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
#  --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-12-15_10-38-06/bin/model_final.pt' \
#  --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/TBA/' \
#  --train_or_test 'test' \
#  --verbose
# echo "Finished"

echo "Training Catalogue Type: F"
python src/main_multimodal.py \
 --gpu_id 0 \
 --config_json 'config/multimodal/F/bcosdensenet201_pretr.json' \
 --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
 --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-12-16_18-57-00/bin/model_final.pt' \
 --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F'
echo "Finished"
# echo "Testing Catalogue Type: F"
# python src/main_multimodal.py \
#  --gpu_id 0 \
#  --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
#  --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-12-16_18-57-00/bin/model_final.pt' \
#  --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/TBA/' \
#  --train_or_test 'test' \
#  --verbose
# echo "Finished"