from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import compose_eval as mano
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.compose_eval import NormalSampler

import os
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import argparse
from raisimGymTorch.helper import rotations, label_gen_final

data_version = "chiral_220223"
ref_r = [0.09566994, 0.00638343, 0.0061863]
ref_l = [-0.09566994, 0.00638343, 0.0061863]
path_mean_r = os.path.join("./../rsc/mano_double/right_pose_mean.txt")
path_mean_l = os.path.join("./../rsc/mano_double/left_pose_mean.txt")
pose_mean_r = np.loadtxt(path_mean_r)
pose_mean_l = np.loadtxt(path_mean_l)


weight_path_articulate_l = '/../../../general_two/2023-05-15-10-18-54/full_400_l.pt'
weight_path_articulate_r = '/../../../general_two/2023-05-15-10-18-54/full_400_r.pt'



exp_name = "floating_evaluation"

ref_frame = 0
height_desk = 0.5
xpos_desk = 0.3

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg_reg.yaml')
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default=exp_name)
parser.add_argument('-w', '--weight', type=str, default='2022-06-13-17-21-55/full_3000_l.pt')
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-pr', '--prior', action="store_true")
parser.add_argument('-o', '--obj_id', type=int, default=1)
parser.add_argument('-t', '--test', action="store_true")
parser.add_argument('-mc', '--mesh_collision', action="store_true")
parser.add_argument('-ao', '--all_objects', action="store_true")
parser.add_argument('-ev', '--evaluate', action="store_true")
parser.add_argument('-to', '--test_object_set', type=int, default=-1)
parser.add_argument('-ac', '--all_contact', action="store_true")
parser.add_argument('-seed', '--seed', type=int, default=1)
parser.add_argument('-itr', '--num_iterations', type=int, default=9001)
parser.add_argument('-nr', '--num_repeats', type=int, default=5)
parser.add_argument('-debug', '--debug', action="store_true")
parser.add_argument('-lr', '--log_rewards', action="store_true")
parser.add_argument('-obj', '--obj', help='which obj to test', type=str, default="box")
parser.add_argument('-test', '--test_set', help='test or train set', action="store_true")
parser.add_argument('-grasp', '--grasp', action="store_true")
parser.add_argument('-record', '--record', help='record the state', action="store_true")

args = parser.parse_args()
mode = args.mode
# weight_path = args.weight
cfg_grasp = args.cfg
record = args.record


print(f"Configuration file: \"{args.cfg}\"")
print(f"Experiment name: \"{args.exp_name}\"")

# task specification
task_name = args.exp_name
# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

if args.logdir is None:
    exp_path = home_path
else:
    exp_path = args.logdir

# config
cfg = YAML().load(open(task_path + '/cfgs/' + args.cfg, 'r'))

if args.seed != 1:
    cfg['seed'] = args.seed


start_offset = 0
reward_clip = -2.0

activations = nn.LeakyReLU
output_activation = nn.Tanh

output_act = False
inference_flag = False
all_obj_train = True if args.all_objects else False
all_train = False
test_inference = args.test
train_obj_id = args.obj_id


obj_name = args.obj
test = args.test_set
if test:
    print("test set")
else:
    print("train set")
arti_labels_list, grasp_labels_list = label_gen_final.label_eval_floating(obj_name, test)

label_list = grasp_labels_list

num_envs = len(arti_labels_list) * len(grasp_labels_list)
num_repeats = len(arti_labels_list) * len(grasp_labels_list)


shuffle_label = []
for env_id in range(num_envs):
    shuffle_label.append(label_list[0])
processed_data, obj_list, left_kind_list, right_kind_list = label_gen_final.pose_gen(shuffle_label, num_repeats, False)
print(obj_list)

stage_dim = processed_data[0]
stage_pos = processed_data[1]
stage_dim[:, 0] = 3.0
stage_dim[:, 1] = 3.0
obj_pose_reset = processed_data[2]
qpos_reset_r = processed_data[3]
qpos_reset_l = processed_data[4]
final_obj_angle = processed_data[5]
final_obj_pos = processed_data[6]
final_obj_pos_r = processed_data[7]
final_ee_r = processed_data[8]
final_ee_l = processed_data[9]
final_pose_r = processed_data[10]
final_pose_l = processed_data[11]
final_qpos_r = processed_data[12]
final_qpos_l = processed_data[13]
final_contacts_r = processed_data[14]
final_contacts_l = processed_data[15]
final_obj_euler = processed_data[16]

num_envs = final_qpos_r.shape[0]
cfg['environment']['hand_model_r'] = "rhand_mano_meshcoll.urdf" if args.mesh_collision else "rhand_mano_demo.urdf"
cfg['environment']['hand_model_l'] = "lhand_mano_meshcoll.urdf" if args.mesh_collision else "lhand_mano_demo.urdf"
cfg['environment']['num_envs'] = 1 if args.evaluate else num_envs
if args.debug:
    cfg['environment']['num_envs'] = 1
cfg["testing"] = True if test_inference else False
print('num envs', num_envs)

# Environment definition
env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'])
env.add_stage(stage_dim, stage_pos)

obj_path_list = []
for obj_item in obj_list:
    obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)

# Setting dimensions from environments
n_act = final_qpos_r[0].shape[0]
ob_dim_grasp = env.num_obs_r
ob_dim_grasp_l = env.num_obs_l
ob_dim_critic = ob_dim_grasp
ob_dim_critic_l = ob_dim_grasp_l
act_dim = env.num_acts

# Training
grasp_steps = 260
arti_steps = 400
if obj_name == 'microwave' or obj_name == 'notebook' or obj_name == 'waffleiron':
    transit_steps = 500
else:
    transit_steps = 300
n_steps_r = grasp_steps + transit_steps + transit_steps + transit_steps + arti_steps
n_steps_l = grasp_steps + transit_steps + transit_steps + transit_steps + arti_steps
total_steps_r = n_steps_r * env.num_envs

# RL network
actor_l = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_grasp_l, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

critic_l = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_critic_l, 1), device)

actor_r = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_grasp, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_critic, 1), device)


test_dir = True

saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[], test_dir=test_dir)

ppo_l = PPO.PPO(actor=actor_l,
                critic=critic_l,
                num_envs=num_envs,
                num_transitions_per_env=n_steps_r,
                num_learning_epochs=4,
                gamma=0.996,
                lam=0.95,
                num_mini_batches=4,
                device=device,
                log_dir=saver.data_dir,
                shuffle_batch=False
                )

ppo_r = PPO.PPO(actor=actor_r,
                critic=critic_r,
                num_envs=num_envs,
                num_transitions_per_env=n_steps_r,
                num_learning_epochs=4,
                gamma=0.996,
                lam=0.95,
                num_mini_batches=4,
                device=device,
                log_dir=saver.data_dir,
                shuffle_batch=False
                )

load_param(saver.data_dir + weight_path_articulate_l, env, actor_l, critic_l, ppo_l.optimizer, saver.data_dir, cfg_grasp)
load_param(saver.data_dir + weight_path_articulate_r, env, actor_r, critic_r, ppo_r.optimizer, saver.data_dir, cfg_grasp)

success = np.zeros((num_envs, 1), 'float32')
success_srict = np.zeros((num_envs, 1), 'float32')
success_grasp = np.zeros((num_envs, 1), 'float32')
success_arrive = np.zeros((num_envs, 1), 'float32')
success_open = np.zeros((num_envs, 1), 'float32')
success_ori = np.zeros((num_envs, 1), 'float32')

final_pos_error = np.zeros((num_envs, 1), 'float32')
final_angle_error = np.zeros((num_envs, 1), 'float32')
final_arti_angle_error = np.zeros((num_envs, 1), 'float32')

if record:
    trans_obj = np.zeros((num_envs, 3))
    rot_obj = np.zeros((num_envs, 3))
    angle_obj = np.zeros((num_envs, 1))

    target_trans_obj = np.zeros((num_envs, 3))
    target_rot_obj = np.zeros((num_envs, 3))
    target_angle_obj = np.zeros((num_envs, 1))

    trans_r = np.zeros((num_envs, 3))
    rot_r = np.zeros((num_envs, 3))
    pose_r = np.zeros((num_envs, 45))
    trans_l = np.zeros((num_envs, 3))
    rot_l = np.zeros((num_envs, 3))
    pose_l = np.zeros((num_envs, 45))


for update in range(1):
    opened = np.zeros((num_envs, 1), 'float32')
    grasped = np.zeros((num_envs, 1), 'float32')
    arrived = np.zeros((num_envs, 1), 'float32')
    ori_get = np.zeros((num_envs, 1), 'float32')

    motion_phase = 0

    start = time.time()

    grasp_label = []
    for env_id in range(num_envs):
        grasp_label.append(grasp_labels_list[int(env_id / (len(arti_labels_list)))])
    processed_data, _, left_kind_list, right_kind_list = label_gen_final.pose_gen(grasp_label, num_repeats, False)
    stage_dim = processed_data[0]
    stage_pos = processed_data[1]
    grasp_obj_pose_reset = processed_data[2]
    grasp_qpos_reset_r = processed_data[3]
    grasp_qpos_reset_l = processed_data[4]
    grasp_final_obj_angle = processed_data[5]
    grasp_final_obj_pos = processed_data[6]
    grasp_final_obj_pos_r = processed_data[7]
    grasp_final_ee_r = processed_data[8]
    grasp_final_ee_l = processed_data[9]
    grasp_final_pose_r = processed_data[10]
    grasp_final_pose_l = processed_data[11]
    grasp_final_qpos_r = processed_data[12]
    grasp_final_qpos_l = processed_data[13]
    grasp_final_contacts_r = processed_data[14]
    grasp_final_contacts_l = processed_data[15]
    grasp_final_obj_euler = processed_data[16]

    arti_label = []
    for env_id in range(num_envs):
        arti_label.append(arti_labels_list[env_id % len(arti_labels_list)])
    arti_data, _, left_kind_list_arti, right_kind_list_arti = label_gen_final.pose_gen(arti_label, num_repeats, False)

    arti_obj_pose_reset = arti_data[2]
    arti_qpos_reset_r = arti_data[3]
    arti_qpos_reset_l = arti_data[4]
    arti_final_obj_angle = arti_data[5]
    arti_final_obj_pos = arti_data[6]
    arti_final_obj_pos_r = arti_data[7]
    arti_final_ee_r = arti_data[8]
    arti_final_ee_l = arti_data[9]
    arti_final_pose_r = arti_data[10]
    arti_final_pose_l = arti_data[11]
    arti_final_qpos_r = arti_data[12]
    arti_final_qpos_l = arti_data[13]
    arti_final_contacts_r = arti_data[14]
    arti_final_contacts_l = arti_data[15]
    arti_final_obj_euler = arti_data[16]



    env.reset_state(grasp_qpos_reset_r,
                    grasp_qpos_reset_l,
                    np.zeros((num_envs, 51), 'float32'),
                    np.zeros((num_envs, 51), 'float32'),
                    grasp_obj_pose_reset
                    )

    left_kinds = np.float32(left_kind_list.copy())
    right_kinds = np.float32(right_kind_list.copy())

    obs_r, obs_l = env.observe()
    obs_r = obs_r.astype('float32')
    obs_l = obs_l.astype('float32')
    global_state = env.get_global_state().astype('float32')

    z_bias = 0.15
    pos_bias = np.random.uniform(-0.1, -0.05, size=(num_envs, 2)).astype('float32')
    angle_ref = np.random.uniform(0.5, 0.6, size=(num_envs, 1)).astype('float32')
    angle_reset = grasp_obj_pose_reset[:, -1].reshape(-1, 1)
    angle = (angle_reset > 1.0) * (angle_reset - angle_ref) + (angle_reset <= 1.0) * angle_ref
    ori = np.random.uniform(-0.4, 0.4, size=(num_envs, 1)).astype('float32')

    for step in range(n_steps_r):
        if step < grasp_steps:
            if motion_phase == 0:
                print("start grasp")
                motion_phase += 1

                env.set_goals_r(grasp_final_obj_pos,
                                grasp_final_ee_r,
                                grasp_final_pose_r,
                                grasp_final_qpos_r,
                                )

                random_noise_angle = final_obj_angle.copy()
                random_noise_angle[:] = 0

                env.set_goals(random_noise_angle,
                              grasp_final_obj_pos,
                              grasp_final_ee_r,
                              grasp_final_ee_l,
                              grasp_final_pose_r,
                              grasp_final_pose_l,
                              grasp_final_qpos_r,
                              grasp_final_qpos_l,
                              grasp_final_contacts_r,
                              grasp_final_contacts_l,
                              )

                obj_pos_goal = grasp_obj_pose_reset.copy()
                obj_pos_goal[:, 2] += z_bias
                obj_pos_goal[:, 1] += pos_bias[:, 1]
                obj_pos_goal[:, 0] += pos_bias[:, 0]
                euler = rotations.quat2euler(obj_pos_goal[:, 3:7])
                euler[:, 1] += ori[:, 0]
                obj_pos_goal[:, 3:7] = rotations.euler2quat(euler)
                env.set_obj_goal(random_noise_angle, obj_pos_goal)

                env.set_imitation_goals(grasp_qpos_reset_l, grasp_qpos_reset_r, grasp_obj_pose_reset)

                for i in range(num_envs):
                    if left_kinds[i] == 7:
                        left_kinds[i] = 11
                    if right_kinds[i] == 7:
                        right_kinds[i] = 11
                env.control_switch_all(left_kinds, right_kinds)

        elif step < grasp_steps + transit_steps:
            current_grasped_state = (-global_state[:, 5] > 0.1).reshape(-1, 1)
            grasped += current_grasped_state - current_grasped_state * grasped
            if motion_phase == 1:
                print("start synthesis")
                motion_phase += 1

                synthesis_right_kinds = right_kinds.copy()
                synthesis_left_kinds = left_kinds.copy()
                for i in range(num_envs):
                    if synthesis_right_kinds[i] == 9:
                        synthesis_right_kinds[i] = 1
                    if synthesis_left_kinds[i] == 9:
                        synthesis_left_kinds[i] = 1
                env.control_switch_all(synthesis_left_kinds, synthesis_right_kinds)

        elif step < grasp_steps + transit_steps + transit_steps:
            if motion_phase == 2:
                print("start synthesis")
                motion_phase += 1

                obj_pos_goal = grasp_obj_pose_reset.copy()
                obj_pos_goal[:, 1] += pos_bias[:, 1]
                obj_pos_goal[:, 0] += pos_bias[:, 0]
                euler = rotations.quat2euler(obj_pos_goal[:, 3:7])
                euler[:, 1] += ori[:, 0]
                obj_pos_goal[:, 3:7] = rotations.euler2quat(euler)
                env.set_obj_goal(random_noise_angle, obj_pos_goal)
                synthesis_right_kinds = right_kinds.copy()
                synthesis_left_kinds = left_kinds.copy()
                for i in range(num_envs):
                    if synthesis_right_kinds[i] == 9:
                        synthesis_right_kinds[i] = 1
                    if synthesis_left_kinds[i] == 9:
                        synthesis_left_kinds[i] = 1
                env.control_switch_all(synthesis_left_kinds, synthesis_right_kinds)
            current_arrive_state = (global_state[:, 0] < 0.05).reshape(-1, 1)
            arrived += current_arrive_state - current_arrive_state * arrived

            current_obj_quat = global_state[:, -5:-1]
            current_obj_euler = rotations.quat2euler(current_obj_quat)
            current_ori_get = (np.abs(euler[:, 1] - current_obj_euler[:, 1]) < 0.2).reshape(-1, 1)
            ori_get += current_ori_get - current_ori_get * ori_get

        elif step < grasp_steps + transit_steps + transit_steps + transit_steps/4:
            if motion_phase == 3:
                final_pos_error = global_state[:, 0].reshape(-1, 1)
                final_angle_error = np.abs(euler[:, 1] - current_obj_euler[:, 1]).reshape(num_envs, 1)

                print("drop object")
                motion_phase += 1

                env.set_goals_r(arti_final_obj_pos,
                                arti_final_ee_r,
                                arti_final_pose_r,
                                arti_final_qpos_r,
                                )

                random_noise_angle = angle.copy()

                env.set_goals(random_noise_angle,
                              arti_final_obj_pos,
                              arti_final_ee_r,
                              arti_final_ee_l,
                              arti_final_pose_r,
                              arti_final_pose_l,
                              arti_final_qpos_r,
                              arti_final_qpos_l,
                              arti_final_contacts_r,
                              arti_final_contacts_l,
                              )
                arti_qpos_reset_l_new = arti_qpos_reset_l.copy()
                arti_qpos_reset_r_new = arti_qpos_reset_r.copy()
                arti_qpos_reset_l_new[:, :] = 0
                arti_qpos_reset_r_new[:, :] = 0

                env.set_imitation_goals(arti_qpos_reset_l_new, arti_qpos_reset_r_new, arti_obj_pose_reset)

                env.set_obj_goal(random_noise_angle, obj_pos_goal)

                left_kinds_arti = np.float32(left_kind_list_arti.copy())
                right_kinds_arti = np.float32(right_kind_list_arti.copy())
                transit_kinds_l = left_kinds_arti.copy()
                transit_kinds_r = right_kinds_arti.copy()
                if obj_name == 'mixer' or obj_name == 'ketchup':
                    transit_kinds_l[:] = 7
                else:
                    transit_kinds_l[:] = 13
                transit_kinds_r[:] = 13

                env.control_switch_all(transit_kinds_l, transit_kinds_r)

        elif step < grasp_steps + transit_steps + transit_steps + transit_steps/2:
            if motion_phase == 4:
                print("go to arti pos")
                motion_phase += 1
                arti_qpos_reset_l_new = arti_qpos_reset_l.copy()
                arti_qpos_reset_r_new = arti_qpos_reset_r.copy()
                arti_qpos_reset_l_new[:, :] = 0
                arti_qpos_reset_r_new[:, :] = 0
                arti_qpos_reset_l_new[:, 0] += 0.5
                arti_qpos_reset_r_new[:, 0] -= 0.5

                left_kinds_arti = np.float32(left_kind_list_arti.copy())
                right_kinds_arti = np.float32(right_kind_list_arti.copy())
                transit_kinds_l = left_kinds_arti.copy()
                transit_kinds_r = right_kinds_arti.copy()
                if obj_name == 'mixer' or obj_name == 'ketchup':
                    transit_kinds_l[:] = 7
                else:
                    transit_kinds_l[:] = 13
                transit_kinds_r[:] = 13
                env.set_imitation_goals(arti_qpos_reset_l_new, arti_qpos_reset_r_new, arti_obj_pose_reset)
                env.control_switch_all(transit_kinds_l, transit_kinds_r)

        elif step < grasp_steps + transit_steps + transit_steps + transit_steps/4*3:
            if motion_phase == 5:
                print("go to arti pos")
                motion_phase += 1
                arti_qpos_reset_l_new = arti_qpos_reset_l.copy()
                arti_qpos_reset_r_new = arti_qpos_reset_r.copy()
                arti_qpos_reset_l_new[:, :] = 0
                arti_qpos_reset_r_new[:, :] = 0
                arti_qpos_reset_l_new[:, 2] += 0.3
                arti_qpos_reset_r_new[:, 2] += 0.3
                arti_qpos_reset_l_new[:, 1] += 0.3
                arti_qpos_reset_r_new[:, 1] += 0.3

                left_kinds_arti = np.float32(left_kind_list_arti.copy())
                right_kinds_arti = np.float32(right_kind_list_arti.copy())
                transit_kinds_l = left_kinds_arti.copy()
                transit_kinds_r = right_kinds_arti.copy()
                if obj_name == 'mixer'or obj_name == 'ketchup':
                    transit_kinds_l[:] = 7
                else:
                    transit_kinds_l[:] = 13
                transit_kinds_r[:] = 13
                env.set_imitation_goals(arti_qpos_reset_l_new, arti_qpos_reset_r_new, arti_obj_pose_reset)
                env.control_switch_all(transit_kinds_l, transit_kinds_r)

        elif step < grasp_steps + transit_steps + transit_steps + transit_steps + 100:
            if motion_phase == 6:
                print("go to arti pos")
                motion_phase += 1

                if obj_name == 'notebook' or obj_name == 'mixer' or obj_name == 'ketchup':
                    weight_path_articulate_r = '/../../../multi_obj_arti/2023-05-14-16-30-13/full_8600_r.pt'
                    load_param(saver.data_dir + weight_path_articulate_r, env, actor_r, critic_r, ppo_r.optimizer,
                               saver.data_dir, cfg_grasp)

                left_kinds_arti = np.float32(left_kind_list_arti.copy())
                right_kinds_arti = np.float32(right_kind_list_arti.copy())
                transit_kinds_l = left_kinds_arti.copy()
                transit_kinds_r = right_kinds_arti.copy()
                transit_kinds_l[:] = 7
                transit_kinds_r[:] = 11
                env.set_imitation_goals(arti_qpos_reset_l, arti_qpos_reset_r, arti_obj_pose_reset)
                env.control_switch_all(transit_kinds_l, transit_kinds_r)
        else:
            if motion_phase == 7:
                obj_angle = global_state[:, 3].copy()
                env.reset_right_hand(arti_qpos_reset_l, arti_qpos_reset_r, arti_obj_pose_reset)
                print("start arti")

                motion_phase += 1
                left_kinds_arti[:] = 7
                env.control_switch_all(left_kinds_arti, right_kinds_arti)
            current_opened_state = (np.abs(global_state[:, 3].reshape(num_envs, 1) - angle[:]) < 0.5).reshape(-1, 1)
            opened += current_opened_state - current_opened_state * opened

        obs_r, obs_l = env.observe()
        obs_r = obs_r.astype('float32')
        obs_l = obs_l.astype('float32')
        global_state = env.get_global_state().astype('float32')

        action_r = actor_r.architecture.architecture(torch.from_numpy(obs_r.astype('float32')).to(device))
        action_r = action_r.cpu().detach().numpy()

        action_l = actor_l.architecture.architecture(torch.from_numpy(obs_l.astype('float32')).to(device))
        action_l = action_l.cpu().detach().numpy()

        frame_start = time.time()

        _, reward_l, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))


        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    if record:
        for iii in range(num_envs):
            trans_r[iii, :] = global_state[iii, 6:9] - ref_r
            axis, hand_angle = rotations.quat2axisangle(global_state[iii, 9:13])
            rot_r[iii, :] = axis * hand_angle
            pose_r[iii, :] = global_state[iii, 19:64] - pose_mean_r

            trans_l[iii, :] = global_state[iii, 64:67] - ref_l
            axis, hand_angle = rotations.quat2axisangle(global_state[iii, 67:71])
            rot_l[iii, :] = axis * hand_angle
            pose_l[iii, :] = global_state[iii, 77:122] - pose_mean_l

            trans_obj[iii, :] = global_state[iii, -8:-5]
            axis, obj_angle = rotations.quat2axisangle(global_state[iii, -5:-1])
            rot_obj[iii, :] = axis * obj_angle
            angle_obj[iii, :] = global_state[iii, -1:]

            target_trans_obj[iii, :] = obj_pos_goal[iii, :3]
            target_axis, target_angle = rotations.quat2axisangle(obj_pos_goal[iii, 3:7])
            target_rot_obj[iii, :] = target_axis * target_angle
            target_angle_obj[iii, :] = angle[iii, :]

        data = {}
        data[f'{obj_name}'] = {}
        data[f'{obj_name}_gt'] = {}
        data['right_hand'] = {}
        data['left_hand'] = {}
        data[f'{obj_name}']['trans'] = np.float32(trans_obj)
        data[f'{obj_name}']['rot'] = np.float32(rot_obj)
        data[f'{obj_name}']['angle'] = np.float32(angle_obj)
        data[f'{obj_name}_gt']['trans'] = np.float32(target_trans_obj)
        data[f'{obj_name}_gt']['rot'] = np.float32(target_rot_obj)
        data[f'{obj_name}_gt']['angle'] = np.float32(target_angle_obj)
        data['right_hand']['trans'] = np.float32(trans_r)
        data['right_hand']['rot'] = np.float32(rot_r)
        data['right_hand']['pose'] = np.float32(pose_r)
        data['left_hand']['trans'] = np.float32(trans_l)
        data['left_hand']['rot'] = np.float32(rot_l)
        data['left_hand']['pose'] = np.float32(pose_l)
        np.save(f"./data_all/recorded/composed_ours/{obj_name}.npy", data)


    final_arti_angle_error = np.abs(global_state[:, 3] - angle[:, 0]).reshape(num_envs, 1)
    not_fall_temp = (global_state[:, 0] < 1.0).reshape(-1, 1) * (global_state[:, 3] < 2.0).reshape(-1, 1)
    success_arrive += arrived
    success_ori += ori_get
    success_open += opened * not_fall_temp
    success += arrived * opened * ori_get * not_fall_temp

print("end")
print("average success arrive rate", np.sum(success_arrive) / num_envs)
print("average success open rate", np.sum(success_open) / num_envs)
print("average success ori rate", np.sum(success_ori) / num_envs)
final_pos_error = np.clip(final_pos_error, 0, 0.1)
final_angle_error = np.clip(final_angle_error, 0, 0.3)
final_arti_angle_error = np.clip(final_arti_angle_error, 0, 1.0)
print("average pos error", np.sum(final_pos_error) / num_envs)
print("average angle error", np.sum(final_angle_error) / num_envs)
print("average arti angle error", np.sum(final_arti_angle_error) / num_envs)
print("average success rate", np.sum(success) / num_envs)
print("ave success rate on successful arrival", np.sum(success) / np.sum(success_arrive))
print("ave success rate on successful grasp", np.sum(success) / np.sum(success_arrive * success_ori))