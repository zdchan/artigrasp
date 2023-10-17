#!/bin/sh
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 1 -e '002_can_extstate_v4_3_nocontact' -c 'cfg_nocontact.yaml'      -sd 'data_v4_3' -w '2021-11-08-22-23-01/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 3 -e '004_sugarbox_extstate_v4_3_nocontact' -c 'cfg_nocontact.yaml' -sd 'data_v4_3'  -w '2021-11-08-22-23-03/full_3000.pt' -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 20 -e '052_clamp_extstate_v4_3_nocontact' -c 'cfg_nocontact.yaml'   -sd 'data_v4_3' -w '2021-11-08-22-22-57/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 1 -e '002_can_extstate_v4_3_norg' -c 'cfg_norg.yaml'                -sd 'data_v4_3'  -w '2021-11-08-23-14-09/full_1800.pt' -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 3 -e '004_sugarbox_extstate_v4_3_norg' -c 'cfg_norg.yaml'           -sd 'data_v4_3' -w '2021-11-08-23-14-08/full_1000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 20 -e '052_clamp_extstate_v4_3_norg' -c 'cfg_norg.yaml'             -sd 'data_v4_3'   -w '2021-11-08-23-14-07/full_3000.pt' -p

python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 1 -e '002_can_extstate_v4_3_nocontact' -c 'cfg_nocontact.yaml'       -sd 'data_v4_3' -w '2021-11-08-22-23-01/full_3000.pt'  -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 3 -e '004_sugarbox_extstate_v4_3_nocontact' -c 'cfg_nocontact.yaml'  -sd 'data_v4_3'  -w '2021-11-08-22-23-03/full_3000.pt' -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 20 -e '052_clamp_extstate_v4_3_nocontact' -c 'cfg_nocontact.yaml'    -sd 'data_v4_3'  -w '2021-11-08-22-22-57/full_3000.pt' -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 1 -e '002_can_extstate_v4_3_norg' -c 'cfg_norg.yaml'                 -sd 'data_v4_3'  -w '2021-11-08-23-14-09/full_1800.pt' -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 3 -e '004_sugarbox_extstate_v4_3_norg' -c 'cfg_norg.yaml'            -sd 'data_v4_3'  -w '2021-11-08-23-14-08/full_1000.pt' -p -t
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 20 -e '052_clamp_extstate_v4_3_norg' -c 'cfg_norg.yaml'              -sd 'data_v4_3'  -w '2021-11-08-23-14-07/full_3000.pt' -p -t

python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 1 -e '002_can_extstate_v4_3_nocontact' -c 'cfg_nocontact.yaml'      -sd 'data_v4_3' -w '2021-11-08-22-23-01/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 3 -e '004_sugarbox_extstate_v4_3_nocontact' -c 'cfg_nocontact.yaml' -sd 'data_v4_3'  -w '2021-11-08-22-23-03/full_3000.pt' -p
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 20 -e '052_clamp_extstate_v4_3_nocontact' -c 'cfg_nocontact.yaml'   -sd 'data_v4_3' -w '2021-11-08-22-22-57/full_3000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 1 -e '002_can_extstate_v4_3_norg' -c 'cfg_norg.yaml'                -sd 'data_v4_3'  -w '2021-11-08-23-14-09/full_1800.pt' -p
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 3 -e '004_sugarbox_extstate_v4_3_norg' -c 'cfg_norg.yaml'           -sd 'data_v4_3' -w '2021-11-08-23-14-08/full_1000.pt'  -p
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 20 -e '052_clamp_extstate_v4_3_norg' -c 'cfg_norg.yaml'             -sd 'data_v4_3'   -w '2021-11-08-23-14-07/full_3000.pt' -p

python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 1 -e '002_can_extstate_v4_3_nocontact' -c 'cfg_nocontact.yaml'       -sd 'data_v4_3' -w '2021-11-08-22-23-01/full_3000.pt'  -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 3 -e '004_sugarbox_extstate_v4_3_nocontact' -c 'cfg_nocontact.yaml'  -sd 'data_v4_3'  -w '2021-11-08-22-23-03/full_3000.pt' -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 20 -e '052_clamp_extstate_v4_3_nocontact' -c 'cfg_nocontact.yaml'    -sd 'data_v4_3'  -w '2021-11-08-22-22-57/full_3000.pt' -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 1 -e '002_can_extstate_v4_3_norg' -c 'cfg_norg.yaml'                 -sd 'data_v4_3'  -w '2021-11-08-23-14-09/full_1800.pt' -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 3 -e '004_sugarbox_extstate_v4_3_norg' -c 'cfg_norg.yaml'            -sd 'data_v4_3'  -w '2021-11-08-23-14-08/full_1000.pt' -p -t
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 20 -e '052_clamp_extstate_v4_3_norg' -c 'cfg_norg.yaml'              -sd 'data_v4_3'  -w '2021-11-08-23-14-07/full_3000.pt' -p -t


