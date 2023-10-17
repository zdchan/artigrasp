#!/bin/sh
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 10 -e '011_banana_extstate_nopose_v4_2_np' -c 'cfg_nopose.yaml' -sd 'data_v4'     -w '2021-11-08-00-04-51/full_2800.pt' -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 13 -e '024_bowl_extstate_nopose_v4_2_np' -c 'cfg_nopose.yaml' -sd 'data_v4'       -w '2021-11-08-00-04-51/full_3000.pt' -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 14 -e '025_mug_extstate_nopose_v4_2_np' -c 'cfg_nopose.yaml' -sd 'data_v4'        -w '2021-11-08-00-04-50/full_3000.pt' -p 
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 15 -e '035_powerdrill_extstate_nopose_v4_2_np' -c 'cfg_nopose.yaml' -sd 'data_v4' -w '2021-11-08-00-04-53/full_2200.pt' -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 17 -e '037_scissors_extstate_nopose_v4_2_np' -c 'cfg_nopose.yaml' -sd 'data_v4'   -w '2021-11-08-00-05-03/full_2000.pt' -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 18 -e '040_marker_extstate_nopose_v4_2_np' -c 'cfg_nopose.yaml' -sd 'data_v4'     -w '2021-11-08-00-05-00/full_2200.pt' -p
python raisimGymTorch/env/envs/mano_eval/runner_eval.py -o 20 -e '052_clamp_extstate_nopose_v4_2_np' -c 'cfg_nopose.yaml' -sd 'data_v4'      -w '2021-11-08-00-26-30/full_2600.pt' -p

python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 10 -e '011_banana_extstate_nopose_v4_2_np' -c 'cfg_nopose.yaml' -sd 'data_v4'     -w '2021-11-08-00-04-51/full_2800.pt' -p
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 13 -e '024_bowl_extstate_nopose_v4_2_np' -c 'cfg_nopose.yaml' -sd 'data_v4'       -w '2021-11-08-00-04-51/full_3000.pt' -p
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 14 -e '025_mug_extstate_nopose_v4_2_np' -c 'cfg_nopose.yaml' -sd 'data_v4'        -w '2021-11-08-00-04-50/full_3000.pt' -p 
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 15 -e '035_powerdrill_extstate_nopose_v4_2_np' -c 'cfg_nopose.yaml' -sd 'data_v4' -w '2021-11-08-00-04-53/full_2200.pt' -p
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 17 -e '037_scissors_extstate_nopose_v4_2_np' -c 'cfg_nopose.yaml' -sd 'data_v4'   -w '2021-11-08-00-05-03/full_2000.pt' -p
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 18 -e '040_marker_extstate_nopose_v4_2_np' -c 'cfg_nopose.yaml' -sd 'data_v4'     -w '2021-11-08-00-05-00/full_2200.pt' -p
python raisimGymTorch/env/envs/mano_eval/eval_scores.py -o 20 -e '052_clamp_extstate_nopose_v4_2_np' -c 'cfg_nopose.yaml' -sd 'data_v4'      -w '2021-11-08-00-26-30/full_2600.pt' -p
