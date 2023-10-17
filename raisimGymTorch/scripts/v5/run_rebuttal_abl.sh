#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -e '004_sugarbox_noreg' -c 'cfg.yaml' -o 3 -nr 1 --mode 'retrain' -w '2022-01-11-08-47-52/full_3000.pt' -sd 'data_rebuttal' -p
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -e '052_clamp_noreg' -c 'cfg.yaml' -o 20 -nr 1 --mode 'retrain' -w '2022-01-11-08-47-11/full_3000.pt' -sd 'data_rebuttal' -p

#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -e '004_sugarbox_noreg' -c 'cfg.yaml' -o 3  --mode 'retrain' -w '2022-01-11-08-47-52/full_3000.pt' -sd 'data_rebuttal' -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -e '052_clamp_noreg' -c 'cfg.yaml' -o 20  --mode 'retrain' -w '2022-01-11-08-47-11/full_3000.pt' -sd 'data_rebuttal' -p

#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -e '002_can_nopose' -c 'cfg_nohierarchy.yaml' -o 1 -nr 1 --mode 'retrain' -w '2022-01-11-08-38-04/full_3000.pt' -sd 'data_rebuttal' -p
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -e '004_sugarbox_nopose' -c 'cfg_nohierarchy.yaml' -o 3 -nr 1 --mode 'retrain' -w '2022-01-11-08-44-41/full_3000.pt' -sd 'data_rebuttal' -p
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -e '052_clamp_nopose' -c 'cfg_nohierarchy.yaml' -o 20 -nr 1 --mode 'retrain' -w '2022-01-11-08-51-25/full_3000.pt' -sd 'data_rebuttal' -p
#
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -e '002_can_nopose' -c 'cfg_nohierarchy.yaml' -o 1  --mode 'retrain' -w '2022-01-11-08-38-04/full_3000.pt' -sd 'data_rebuttal' -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -e '004_sugarbox_nopose' -c 'cfg_nohierarchy.yaml' -o 3  --mode 'retrain' -w '2022-01-11-08-44-41/full_3000.pt' -sd 'data_rebuttal' -p
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -e '052_clamp_nopose' -c 'cfg_nohierarchy.yaml' -o 20 --mode 'retrain' -w '2022-01-11-08-51-25/full_3000.pt' -sd 'data_rebuttal' -p

python raisimGymTorch/env/envs/mano_eval/runner_eval.py -e '002_can_contact_pruned' -c 'cfg_contactpruned.yaml' -o 1 -nr 1 --mode 'retrain' -w '2022-01-11-08-47-46/full_3000.pt' -sd 'data_rebuttal' -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -e '004_sugarbox_contact_pruned' -c 'cfg_contactpruned.yaml' -o 3 -nr 1 --mode 'retrain' -w '2022-01-11-08-08-48/full_3000.pt' -sd 'data_rebuttal' -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -e '052_clamp_contact_pruned' -c 'cfg_contactpruned.yaml' -o 20 -nr 1 --mode 'retrain' -w '2022-01-11-08-44-09/full_3000.pt' -sd 'data_rebuttal' -p

python raisimGymTorch/env/envs/mano_eval/eval_scores.py -e '002_can_contact_pruned' -c 'cfg_contactpruned.yaml' -o 1  --mode 'retrain' -w '2022-01-11-08-47-46/full_3000.pt' -sd 'data_rebuttal' -p
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -e '004_sugarbox_contact_pruned' -c 'cfg_contactpruned.yaml' -o 3  --mode 'retrain' -w '2022-01-11-08-08-48/full_3000.pt' -sd 'data_rebuttal' -p
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -e '052_clamp_contact_pruned' -c 'cfg_contactpruned.yaml' -o 20 --mode 'retrain' -w '2022-01-11-08-44-09/full_3000.pt' -sd 'data_rebuttal' -p