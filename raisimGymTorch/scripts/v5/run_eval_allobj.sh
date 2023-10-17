#!/bin/sh

python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 1 -e '002_can_grasptta'         -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-32-38/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 2 -e '003_crackerbox_grasptta'  -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-21-52-12/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 3 -e '004_sugarbox_grasptta'    -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-25-09/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 4 -e '005_tomatocan_grasptta'   -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-05-31/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 6 -e '007_tunacan_grasptta'     -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-21-47-43/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 7 -e '008_puddingbox_grasptta'  -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-21-51-15/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 8 -e '009_gelatinbox_grasptta'  -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-09-51/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 9 -e '010_pottedcan_grasptta'   -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-26-31/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 10 -e '011_banana_grasptta'     -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-21-54-13/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 11 -e '019_pitcher_grasptta'    -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-14-27/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 12 -e '021_bleach_grasptta'     -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-30-05/full_2600.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 13 -e '024_bowl_grasptta'       -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-26-25/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 14 -e '025_mug_grasptta'        -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-21-51-40/full_2800.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 15 -e '035_powerdrill_grasptta' -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-18-57/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 16 -e '036_woodblock_grasptta'  -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-21-57-42/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 17 -e '037_scissors_grasptta'   -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-18-56/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 18 -e '040_marker_grasptta'     -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-35-43/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 20 -e '052_clamp_grasptta'      -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-21-56-11/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 21 -e '061_foam_grasptta'       -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-29-02/full_3000.pt' -gtta -p -t


python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 1 -e '002_can_grasptta'         -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-32-38/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 2 -e '003_crackerbox_grasptta'  -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-21-52-12/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 3 -e '004_sugarbox_grasptta'    -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-25-09/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 4 -e '005_tomatocan_grasptta'   -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-05-31/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 5 -e '006_mustard_grasptta'     -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-01-18/full_2000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 6 -e '007_tunacan_grasptta'     -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-21-47-43/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 7 -e '008_puddingbox_grasptta'  -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-21-51-15/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 8 -e '009_gelatinbox_grasptta'  -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-09-51/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 9 -e '010_pottedcan_grasptta'   -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-26-31/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 10 -e '011_banana_grasptta'     -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-21-54-13/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 11 -e '019_pitcher_grasptta'    -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-14-27/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 12 -e '021_bleach_grasptta'     -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-30-05/full_2600.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 13 -e '024_bowl_grasptta'       -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-26-25/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 14 -e '025_mug_grasptta'        -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-21-51-40/full_2800.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 15 -e '035_powerdrill_grasptta' -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-18-57/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 16 -e '036_woodblock_grasptta'  -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-21-57-42/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 17 -e '037_scissors_grasptta'   -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-18-56/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 18 -e '040_marker_grasptta'     -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-35-43/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 20 -e '052_clamp_grasptta'      -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-21-56-11/full_3000.pt' -gtta -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 21 -e '061_foam_grasptta'       -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_corrected' -w '2022-01-21-22-29-02/full_3000.pt' -gtta -p -t
#
#python raisimGymTorch/env/envs/mano_gen/runner.py -o 1 -e '002_can_contact_pruned_allcontact'   -c 'cfg_contactpruned.yaml' --mode 'retrain' -sd 'data_rebuttal' -w '2022-01-19-16-34-39/full_3000.pt'
#python raisimGymTorch/env/envs/mano_pred/runner.py -o 1 -e '002_can_grasptta_reg'         -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_rebuttal' -w '2022-01-19-09-14-22/full_3000.pt'
#python raisimGymTorch/env/envs/mano_pred/runner.py -o 1 -e '002_can_grasptta_reg_allcontact'         -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_rebuttal' -w '2022-01-19-09-25-27/full_3000.pt' -ac
#python raisimGymTorch/env/envs/mano_pred/runner.py -o 1 -e '002_can_grasptta_reg_allcontact_mi'         -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_rebuttal' -w '2022-01-19-13-06-47/full_3000.pt' -ac -mi
#python raisimGymTorch/env/envs/mano_pred/runner.py -o 1 -e '002_can_grasptta_reg_nopose_allcontact_mi'         -c 'cfg_reg_nopose.yaml' --mode 'retrain' -sd 'data_rebuttal' -w '2022-01-19-13-08-09/full_3000.pt' -ac -mi
#python raisimGymTorch/env/envs/mano_pred/runner.py -o 1 -e '002_can_gtta_allcontact'         -c 'cfg.yaml' --mode 'retrain' -sd 'data_rebuttal' -w '2022-01-19-19-21-28/full_3000.pt' -ac
#python raisimGymTorch/env/envs/mano_pred/runner.py -o 1 -e '002_can_gtta_allcontact_mi'         -c 'cfg.yaml' --mode 'retrain' -sd 'data_rebuttal' -w '2022-01-19-19-53-26/full_3000.pt' -ac -mi
#python raisimGymTorch/env/envs/mano_pred/runner.py -o 1 -e '002_can_gtta_nopose'         -c 'cfg_nopose.yaml' --mode 'retrain' -sd 'data_rebuttal' -w '2022-01-19-22-08-54/full_2600.pt'
#python raisimGymTorch/env/envs/mano_pred/runner.py -o 1 -e '002_can_gtta_nopose_mi'         -c 'cfg_nopose.yaml' --mode 'retrain' -sd 'data_rebuttal' -w '2022-01-19-19-51-13/full_1400.pt' -mi
#python raisimGymTorch/env/envs/mano_pred/runner.py -o 1 -e '002_can_gtta_reg'         -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_rebuttal' -w '2022-01-19-22-08-48/full_3000.pt'
#python raisimGymTorch/env/envs/mano_pred/runner.py -o 1 -e '002_can_gtta_reg_mi'         -c 'cfg_reg.yaml' --mode 'retrain' -sd 'data_rebuttal' -w '2022-01-19-22-08-47/full_3000.pt' -mi
#python raisimGymTorch/env/envs/mano_pred/runner.py -o 1 -e '002_can_gtta_reg_nopose'         -c 'cfg_reg_nopose.yaml' --mode 'retrain' -sd 'data_rebuttal' -w '2022-01-19-20-50-35/full_2200.pt'
#python raisimGymTorch/env/envs/mano_pred/runner.py -o 1 -e '002_can_gtta_reg_nopose_mi'         -c 'cfg_reg_nopose.yaml' --mode 'retrain' -sd 'data_rebuttal' -w '2022-01-19-22-08-46/full_3000.pt' -mi


#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 1 -e '002_can_grasptta' -c 'cfg.yaml'           -w '2021-11-11-11-45-51/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 2 -e '003_crackerbox_grasptta' -c 'cfg.yaml'  -w '2021-11-13-22-31-11/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 3 -e '004_sugarbox_grasptta' -c 'cfg.yaml'      -w '2021-11-11-11-46-18/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 4 -e '005_tomatocan_grasptta' -c 'cfg.yaml'     -w '2021-11-11-11-46-23/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 5 -e '006_mustard_grasptta' -c 'cfg.yaml'       -w '2021-11-11-20-54-39/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 6 -e '007_tunacan_grasptta' -c 'cfg.yaml'       -w '2021-11-11-11-46-41/full_2400.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 7 -e '008_puddingbox_grasptta' -c 'cfg.yaml'    -w '2021-11-11-20-54-38/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 8 -e '009_gelatinbox_grasptta' -c 'cfg.yaml'    -w '2021-11-11-20-54-49/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 9 -e '010_pottedcan_grasptta' -c 'cfg.yaml'     -w '2021-11-11-20-54-36/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 10 -e '011_banana_grasptta' -c 'cfg.yaml'       -w '2021-11-11-20-54-38/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 11 -e '019_pitcher_grasptta' -c 'cfg.yaml'      -w '2021-11-12-00-34-58/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 12 -e '021_bleach_grasptta' -c 'cfg.yaml'       -w '2021-11-12-02-33-58/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 13 -e '024_bowl_grasptta' -c 'cfg.yaml'         -w '2021-11-11-21-51-03/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 14 -e '025_mug_grasptta' -c 'cfg.yaml'          -w '2021-11-11-21-34-34/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 15 -e '035_powerdrill_grasptta' -c 'cfg.yaml'   -w '2021-11-12-03-27-15/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 16 -e '036_woodblock_grasptta' -c 'cfg.yaml'    -w '2021-11-12-06-29-38/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 17 -e '037_scissors_grasptta' -c 'cfg.yaml'     -w '2021-11-11-21-34-32/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 18 -e '040_marker_grasptta' -c 'cfg.yaml'       -w '2021-11-11-21-53-48/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 20 -e '052_clamp_grasptta' -c 'cfg.yaml'        -w '2021-11-11-23-41-21/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 21 -e '061_foam_grasptta' -c 'cfg.yaml'      -w '2021-11-11-11-47-46/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t

#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 1 -e '002_can_grasptta' -c 'cfg.yaml'           -w '2021-11-11-11-45-51/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 2 -e '003_crackerbox_grasptta' -c 'cfg.yaml'  -w '2021-11-13-22-31-11/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 3 -e '004_sugarbox_grasptta' -c 'cfg.yaml'      -w '2021-11-11-11-46-18/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 4 -e '005_tomatocan_grasptta' -c 'cfg.yaml'     -w '2021-11-11-11-46-23/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 5 -e '006_mustard_grasptta' -c 'cfg.yaml'       -w '2021-11-11-20-54-39/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 6 -e '007_tunacan_grasptta' -c 'cfg.yaml'       -w '2021-11-11-11-46-41/full_2400.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 7 -e '008_puddingbox_grasptta' -c 'cfg.yaml'    -w '2021-11-11-20-54-38/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 8 -e '009_gelatinbox_grasptta' -c 'cfg.yaml'    -w '2021-11-11-20-54-49/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 9 -e '010_pottedcan_grasptta' -c 'cfg.yaml'     -w '2021-11-11-20-54-36/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 10 -e '011_banana_grasptta' -c 'cfg.yaml'       -w '2021-11-11-20-54-38/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 11 -e '019_pitcher_grasptta' -c 'cfg.yaml'      -w '2021-11-12-00-34-58/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 12 -e '021_bleach_grasptta' -c 'cfg.yaml'       -w '2021-11-12-02-33-58/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 13 -e '024_bowl_grasptta' -c 'cfg.yaml'         -w '2021-11-11-21-51-03/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 14 -e '025_mug_grasptta' -c 'cfg.yaml'          -w '2021-11-11-21-34-34/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 15 -e '035_powerdrill_grasptta' -c 'cfg.yaml'   -w '2021-11-12-03-27-15/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 16 -e '036_woodblock_grasptta' -c 'cfg.yaml'    -w '2021-11-12-06-29-38/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 17 -e '037_scissors_grasptta' -c 'cfg.yaml'     -w '2021-11-11-21-34-32/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 18 -e '040_marker_grasptta' -c 'cfg.yaml'       -w '2021-11-11-21-53-48/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 20 -e '052_clamp_grasptta' -c 'cfg.yaml'        -w '2021-11-11-23-41-21/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t
#python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 21 -e '061_foam_grasptta' -c 'cfg.yaml'      -w '2021-11-11-11-47-46/full_3000.pt' -p -sd 'data_grasptta'  -gtta -t

#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 1 -e '002_can_grasptta' -c 'cfg.yaml'           -w '2021-11-11-11-45-51/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 2 -e '003_crackerbox_grasptta' -c 'cfg.yaml'    -w '2021-11-11-11-52-45/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 3 -e '004_sugarbox_grasptta' -c 'cfg.yaml'      -w '2021-11-11-11-46-18/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 4 -e '005_tomatocan_grasptta' -c 'cfg.yaml'     -w '2021-11-11-11-46-23/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 5 -e '006_mustard_grasptta' -c 'cfg.yaml'       -w '2021-11-11-20-54-39/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 6 -e '007_tunacan_grasptta' -c 'cfg.yaml'       -w '2021-11-11-11-46-41/full_2400.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 7 -e '008_puddingbox_grasptta' -c 'cfg.yaml'    -w '2021-11-11-20-54-38/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 8 -e '009_gelatinbox_grasptta' -c 'cfg.yaml'    -w '2021-11-11-20-54-49/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 9 -e '010_pottedcan_grasptta' -c 'cfg.yaml'     -w '2021-11-11-20-54-36/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 10 -e '011_banana_grasptta' -c 'cfg.yaml'       -w '2021-11-11-20-54-38/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 11 -e '019_pitcher_grasptta' -c 'cfg.yaml'      -w '2021-11-12-00-34-58/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 12 -e '021_bleach_grasptta' -c 'cfg.yaml'       -w '2021-11-12-02-33-58/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 13 -e '024_bowl_grasptta' -c 'cfg.yaml'         -w '2021-11-11-21-51-03/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 14 -e '025_mug_grasptta' -c 'cfg.yaml'          -w '2021-11-11-21-34-34/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 15 -e '035_powerdrill_grasptta' -c 'cfg.yaml'   -w '2021-11-12-03-27-15/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 16 -e '036_woodblock_grasptta' -c 'cfg.yaml'    -w '2021-11-12-06-29-38/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 17 -e '037_scissors_grasptta' -c 'cfg.yaml'     -w '2021-11-11-21-34-32/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 18 -e '040_marker_grasptta' -c 'cfg.yaml'       -w '2021-11-11-21-53-48/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 20 -e '052_clamp_grasptta' -c 'cfg.yaml'        -w '2021-11-11-23-41-21/full_3000.pt' -p -sd 'data_grasptta'  -gtta
#python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 21 -e '061_foam_grasptta' -c 'cfg.yaml'      -w '2021-11-11-11-47-46/full_3000.pt' -p -sd 'data_grasptta'  -gtta