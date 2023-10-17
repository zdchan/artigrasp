#!/bin/sh
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 1 -e '002_can_extstate_v4_3' -c 'cfg_world.yaml'           -w '2021-11-09-15-50-55/full_3000.pt' -whl '2021-11-14-18-31-13/full_1000.pt' -p -sd 'data_all' -m 'retrain'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 3 -e '004_sugarbox_extstate_v4_3' -c 'cfg_world.yaml'      -w '2021-11-09-15-49-15/full_3000.pt' -whl '2021-11-15-12-51-35/full_900.pt' -p -sd 'data_all' -m 'retrain'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 6 -e '007_tunacan_extstate_v5_s2' -c 'cfg_world.yaml'      -w '2021-11-11-00-01-12/full_2400.pt' -whl '2021-11-15-01-25-48/full_1000.pt' -p -sd 'data_v5' -m 'retrain'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 11 -e '019_pitcher_extstate_v5' -c 'cfg_world.yaml'        -w '2021-11-09-21-24-54/full_3000.pt' -whl '2021-11-15-01-25-44/full_1000.pt' -p -sd 'data_v5' -m 'retrain'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 20 -e '052_clamp_extstate_v5' -c 'cfg_world.yaml'          -w '2021-11-09-21-25-25/full_3000.pt' -whl '2021-11-15-01-25-43/full_1000.pt' -p -sd 'data_v5' -m 'retrain'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 21 -e '061_foam_extstate_v5_s2' -c 'cfg_world.yaml'        -w '2021-11-11-00-01-23/full_2800.pt' -whl '2021-11-14-18-31-13/full_1000.pt' -p -sd 'data_v5' -m 'retrain'

python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 1 -e '002_can_extstate_v4_3' -c 'cfg_world.yaml'        -w '2021-11-09-15-50-55/full_3000.pt' -whl '2021-11-15-09-31-25/full_900.pt' -p -sd 'data_all' -m 'retrain'
python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 6 -e '007_tunacan_extstate_v5_s2' -c 'cfg_world.yaml'   -w '2021-11-11-00-01-12/full_2400.pt' -whl '2021-11-14-16-39-24/full_400.pt' -p -sd 'data_all' -m 'retrain'
python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 20 -e '052_clamp_extstate_v5' -c 'cfg_world.yaml'          -w '2021-11-09-21-25-25/full_3000.pt' -whl '2021-11-15-01-25-43/full_1000.pt' -p -sd 'data_v5' -m 'retrain'
python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 21 -e '061_foam_extstate_v5_s2' -c 'cfg_world.yaml'     -w '2021-11-11-00-01-23/full_2800.pt' -whl '2021-11-15-17-02-27/full_900.pt' -p -sd 'data_all' -m 'retrain'
python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 11 -e '019_pitcher_extstate_v5' -c 'cfg_world.yaml'        -w '2021-11-09-21-24-54/full_3000.pt' -whl '2021-11-15-01-25-44/full_1000.pt' -p -sd 'data_v5' -m 'retrain'
python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 3 -e '004_sugarbox_extstate_v4_3' -c 'cfg_world.yaml'      -w '2021-11-09-15-49-15/full_3000.pt' -whl '2021-11-15-12-51-35/full_900.pt' -p -sd 'data_all' -m 'retrain'

#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 1 -e '002_can_extstate_v4_3' -c 'cfg_pd.yaml'         -w '2021-11-09-15-50-55/full_3000.pt'  -p -sd 'data_all' -m 'retrain' -pd --freeze
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 3 -e '004_sugarbox_extstate_v4_3' -c 'cfg_pd.yaml'    -w '2021-11-09-15-49-15/full_3000.pt'  -p -sd 'data_all' -m 'retrain' -pd --freeze
##python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 5 -e '006_mustard_extstate_v5' -c 'cfg_pd.yaml'       -w '2021-11-10-04-16-23/full_3000.pt'  -p -sd 'data_v5'-m 'retrain' -pd --freeze
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 6 -e '007_tunacan_extstate_v5_s2' -c 'cfg_pd.yaml'    -w '2021-11-11-00-01-12/full_2400.pt'  -p -sd 'data_v5' -m 'retrain' -pd --freeze
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 11 -e '019_pitcher_extstate_v5' -c 'cfg_pd.yaml'      -w '2021-11-09-21-24-54/full_3000.pt'  -p -sd 'data_v5' -m 'retrain' -pd --freeze
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 20 -e '052_clamp_extstate_v5' -c 'cfg_pd.yaml'        -w '2021-11-09-21-25-25/full_3000.pt'  -p -sd 'data_v5' -m 'retrain' -pd --freeze
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 21 -e '061_foam_extstate_v5_s2' -c 'cfg_pd.yaml'      -w '2021-11-11-00-01-23/full_2800.pt'  -p -sd 'data_v5' -m 'retrain' -pd --freeze
#
#
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 1 -e '002_can_extstate_v4_3' -c 'cfg_pd.yaml'         -w '2021-11-09-15-50-55/full_3000.pt'  -p -sd 'data_all' -m 'retrain' -pd
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 3 -e '004_sugarbox_extstate_v4_3' -c 'cfg_pd.yaml'    -w '2021-11-09-15-49-15/full_3000.pt'  -p -sd 'data_all' -m 'retrain' -pd
##python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 5 -e '006_mustard_extstate_v5' -c 'cfg_pd.yaml'       -w '2021-11-10-04-16-23/full_3000.pt'  -p -sd 'data_v5'-m 'retrain' -pd
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 6 -e '007_tunacan_extstate_v5_s2' -c 'cfg_pd.yaml'    -w '2021-11-11-00-01-12/full_2400.pt'  -p -sd 'data_v5' -m 'retrain' -pd
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 11 -e '019_pitcher_extstate_v5' -c 'cfg_pd.yaml'      -w '2021-11-09-21-24-54/full_3000.pt'  -p -sd 'data_v5' -m 'retrain' -pd
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 20 -e '052_clamp_extstate_v5' -c 'cfg_pd.yaml'        -w '2021-11-09-21-25-25/full_3000.pt'  -p -sd 'data_v5' -m 'retrain' -pd
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy.py -o 21 -e '061_foam_extstate_v5_s2' -c 'cfg_pd.yaml'      -w '2021-11-11-00-01-23/full_2800.pt'  -p -sd 'data_v5' -m 'retrain' -pd
#
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 1 -e '002_can_extstate_v4_3' -c 'cfg_world_sparse.yaml'           -w '2021-11-11-11-45-51/full_3000.pt' -whl '2021-11-14-18-31-13/full_1000.pt' -p -sd 'data_all'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 3 -e '004_sugarbox_extstate_v4_3' -c 'cfg_world_sparse.yaml'      -w '2021-11-09-15-49-15/full_3000.pt' -whl '2021-11-14-18-31-13/full_1000.pt' -p -sd 'data_all'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 5 -e '006_mustard_extstate_v5' -c 'cfg_world_sparse.yaml'         -w '2021-11-10-04-16-23/full_3000.pt' -whl '2021-11-14-18-31-13/full_1000.pt' -p -sd 'data_v5'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 6 -e '007_tunacan_extstate_v5_s2' -c 'cfg_world_sparse.yaml'      -w '2021-11-11-00-01-12/full_2400.pt' -whl '2021-11-14-18-31-13/full_1000.pt' -p -sd 'data_v5'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 11 -e '019_pitcher_extstate_v5' -c 'cfg_world_sparse.yaml'        -w '2021-11-09-21-24-54/full_3000.pt' -whl '2021-11-15-01-25-43/full_1000.pt' -p -sd 'data_v5' -m 'retrain'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 20 -e '052_clamp_extstate_v5' -c 'cfg_world_sparse.yaml'          -w '2021-11-09-21-25-25/full_3000.pt' -whl '2021-11-14-18-31-13/full_1000.pt' -p -sd 'data_v5'  -m 'retrain'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 21 -e '061_foam_extstate_v5_s2' -c 'cfg_world_sparse.yaml'        -w '2021-11-11-00-01-23/full_2800.pt'    -whl '2021-11-14-18-31-13/full_1000.pt' -p -sd 'data_v5'

#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 1 -e '002_can_extstate_v4_3' -c 'cfg_world_nopose.yaml'           -w '2021-11-11-11-45-51/full_3000.pt' -whl '2021-11-14-18-31-13/full_1000.pt' -p -sd 'data_all'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 3 -e '004_sugarbox_extstate_v4_3' -c 'cfg_world_nopose.yaml'      -w '2021-11-09-15-49-15/full_3000.pt' -whl '2021-11-14-18-31-13/full_1000.pt' -p -sd 'data_all'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 5 -e '006_mustard_extstate_v5' -c 'cfg_world_nopose.yaml'         -w '2021-11-10-04-16-23/full_3000.pt' -whl '2021-11-14-18-31-13/full_1000.pt' -p -sd 'data_v5'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 6 -e '007_tunacan_extstate_v5_s2' -c 'cfg_world_nopose.yaml'      -w '2021-11-11-00-01-12/full_2400.pt' -whl '2021-11-14-18-31-13/full_1000.pt' -p -sd 'data_v5'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 11 -e '019_pitcher_extstate_v5' -c 'cfg_world_nopose.yaml'        -w '2021-11-09-21-24-54/full_3000.pt' -whl '2021-11-15-01-25-44/full_1000.pt' -p -sd 'data_v5' -m 'retrain'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 20 -e '052_clamp_extstate_v5' -c 'cfg_world_nopose.yaml'          -w '2021-11-09-21-25-25/full_3000.pt' -whl '2021-11-14-18-31-13/full_1000.pt' -p -sd 'data_v5'
#python raisimGymTorch/env/envs/mano_hierarchy/runner_hierarchy_test.py -o 21 -e '061_foam_extstate_v5_s2' -c 'cfg_world_nopose.yaml'        -w '2021-11-11-00-01-23/full_2800.pt'    -whl '2021-11-14-18-31-13/full_1000.pt' -p -sd 'data_v5'

