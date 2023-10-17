from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import general_two as mano
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.general_two import NormalSampler

import os
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import argparse
from raisimGymTorch.helper import rotations, label_gen_final
import random

data_version = "chiral_220223"


weight_path_articulate_l = '/../left_fixed/2023-05-04-16-51-56/full_6600_l.pt'
weight_path_articulate_r = '/../multi_obj_arti/2023-05-04-16-43-43/full_4300_r.pt'

exp_name = "general_two"

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
parser.add_argument('-nr', '--num_repeats', type=int, default=1)
parser.add_argument('-debug', '--debug', action="store_true")
parser.add_argument('-lr', '--log_rewards', action="store_true")
parser.add_argument('-random','--random', help='randomized goal obj angle', action="store_true")
parser.add_argument('-renew', '--renew', help='update labels every iteration', action="store_true")

args = parser.parse_args()
mode = args.mode
cfg_grasp = args.cfg


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

num_envs = cfg['environment']['num_envs']
start_offset = 0
pre_grasp_steps = 100
trail_steps = 200
reward_clip = -2.0

num_repeats = args.num_repeats
activations = nn.LeakyReLU
output_activation = nn.Tanh

output_act = False
inference_flag = False
all_obj_train = True if args.all_objects else False
all_train = False
test_inference = args.test
train_obj_id = args.obj_id


test = False
original_labels_arti, original_labels_grasp, shuffle_label = label_gen_final.label_train(num_repeats, True, test)
processed_data, obj_list, left_kind_list, right_kind_list = label_gen_final.pose_gen(shuffle_label, num_repeats, False)
print(obj_list)

stage_dim = processed_data[0]
stage_pos = processed_data[1]
obj_pose_reset2 = processed_data[2]
qpos_reset_r2 = processed_data[3]
qpos_reset_l2 = processed_data[4]
final_obj_angle2 = processed_data[5]
final_obj_pos2 = processed_data[6]
final_obj_pos_r2 = processed_data[7]
final_ee_r2 = processed_data[8]
final_ee_l2 = processed_data[9]
final_pose_r2 = processed_data[10]
final_pose_l2 = processed_data[11]
final_qpos_r2 = processed_data[12]
final_qpos_l2 = processed_data[13]
final_contacts_r2 = processed_data[14]
final_contacts_l2 = processed_data[15]
final_obj_euler2 = processed_data[16]

num_envs = final_qpos_r2.shape[0]
cfg['environment']['hand_model_r'] = "rhand_mano_meshcoll.urdf" if args.mesh_collision else "rhand_mano.urdf"
cfg['environment']['hand_model_l'] = "lhand_mano_meshcoll.urdf" if args.mesh_collision else "lhand_mano.urdf"
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
n_act = final_qpos_r2[0].shape[0]
ob_dim_grasp = env.num_obs_r
ob_dim_critic = ob_dim_grasp
act_dim = env.num_acts

# Training
grasp_steps = pre_grasp_steps
trans_steps = 0
n_steps_r = grasp_steps + trail_steps + trans_steps
n_steps_l = trail_steps + grasp_steps + trans_steps
total_steps_r = n_steps_r * env.num_envs

avg_rewards = []
contact_rewards = []
impulse_rewards = []
pos_rewards = []
pose_rewards = []
falling_rewards = []
rel_obj_rewards = []
body_vel_rewards = []
body_qvel_rewards = []

# RL network
actor_l = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_grasp, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

critic_l = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_critic, 1), device)

actor_r = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_grasp, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_critic, 1), device)

test_dir = True

saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp", task_path + "/runner_eval.py"], test_dir=test_dir)

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

env.reset_state(qpos_reset_r2,
                qpos_reset_l2,
                np.zeros((num_envs,51),'float32'),
                np.zeros((num_envs,51),'float32'),
                obj_pose_reset2
                )


for update in range(args.num_iterations):
    start = time.time()
    ### Evaluate trained model visually (note always the first environment gets visualized)

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor_l.architecture.state_dict(),
            'actor_distribution_state_dict': actor_l.distribution.state_dict(),
            'critic_architecture_state_dict': critic_l.architecture.state_dict(),
            'optimizer_state_dict': ppo_l.optimizer.state_dict(),
        }, saver.data_dir + "/full_" + str(update) + '_l.pt')

        env.save_scaling(saver.data_dir, str(update))

    # if args.renew:
    renew_labels = []
    obj_name = obj_list[0]
    kind_flag = random.random()
    if kind_flag < 0.5:
        labels_len = len(original_labels_arti[f"{obj_name}_list"])
        temp_label = original_labels_arti[f"{obj_name}_list"][random.randint(0, labels_len - 1)]
    else:
        labels_len = len(original_labels_grasp[f"{obj_name}_list"])
        temp_label = original_labels_grasp[f"{obj_name}_list"][random.randint(0, labels_len - 1)]
    renew_labels.append(temp_label)

    processed_data, obj_list, left_kind_list, right_kind_list = label_gen_final.pose_gen(renew_labels, num_repeats,
                                                                                         False)
    stage_dim = processed_data[0]
    stage_pos = processed_data[1]
    obj_pose_reset2 = processed_data[2]
    qpos_reset_r2 = processed_data[3]
    qpos_reset_l2 = processed_data[4]
    final_obj_angle2 = processed_data[5]
    final_obj_pos2 = processed_data[6]
    final_obj_pos_r2 = processed_data[7]
    final_ee_r2 = processed_data[8]
    final_ee_l2 = processed_data[9]
    final_pose_r2 = processed_data[10]
    final_pose_l2 = processed_data[11]
    final_qpos_r2 = processed_data[12]
    final_qpos_l2 = processed_data[13]
    final_contacts_r2 = processed_data[14]
    final_contacts_l2 = processed_data[15]
    final_obj_euler2 = processed_data[16]

    env.set_goals_r2(final_obj_pos2,
                     final_ee_r2,
                     final_pose_r2,
                     final_qpos_r2,
                     final_contacts_r2
                     )

    env.set_goals(final_obj_angle2,
                  final_obj_pos2,
                  final_ee_r2,
                  final_ee_l2,
                  final_pose_r2,
                  final_pose_l2,
                  final_qpos_r2,
                  final_qpos_l2,
                  final_contacts_r2,
                  final_contacts_l2,
                  )

    # if args.random:
    random_noise_angle = final_obj_angle2.copy()
    rand_temp = np.float32(np.random.uniform(0.5, 1.5, (num_envs, 1)).copy())
    for i in range(num_envs):
        if right_kind_list[i] == 8 or left_kind_list[i] == 8:
            random_noise_angle[i] = rand_temp[i]
        else:
            random_noise_angle[i] = final_obj_angle2[i]
    print(f"ave goal angle: {np.sum(np.abs(random_noise_angle)) / num_envs}")
    env.set_goals(random_noise_angle,
                  final_obj_pos2,
                  final_ee_r2,
                  final_ee_l2,
                  final_pose_r2,
                  final_pose_l2,
                  final_qpos_r2,
                  final_qpos_l2,
                  final_contacts_r2,
                  final_contacts_l2,
                  )

    ### Add some noise to initial hand position
    random_noise_pos = np.float32(np.random.uniform([-0.01, -0.01, -0.001], [0.01, 0.01, 0.001], (num_envs, 3)).copy())
    random_noise_qpos = np.float32(np.random.uniform(-0.2, 0.2, (num_envs, 48)).copy())
    random_noise_pos_l = np.float32(np.random.uniform([-0.01, -0.01, 0.0], [0.01, 0.01, 0.0], (num_envs, 3)).copy())
    random_noise_qpos_l = np.float32(np.random.uniform(-0.2, 0.2, (num_envs, 48)).copy())


    qpos_noisy_reset_r = qpos_reset_r2.copy()
    qpos_noisy_reset_l = qpos_reset_l2.copy()

    env.reset_state(qpos_noisy_reset_r,
                    qpos_noisy_reset_l,
                    np.zeros((num_envs, 51), 'float32'),
                    np.zeros((num_envs, 51), 'float32'),
                    obj_pose_reset2
                    )

    left_kinds = np.float32(left_kind_list.copy())
    right_kinds = np.float32(right_kind_list.copy())

    env.control_switch_all(left_kinds, right_kinds)

    for step in range(n_steps_r):

        obs_r, obs_l = env.observe()
        obs_r = obs_r.astype('float32')
        obs_l = obs_l.astype('float32')

        action_r = actor_r.architecture.architecture(torch.from_numpy(obs_r.astype('float32')).to(device))
        action_r = action_r.cpu().detach().numpy()

        action_l = actor_l.architecture.architecture(torch.from_numpy(obs_l.astype('float32')).to(device))
        action_l = action_l.cpu().detach().numpy()

        frame_start = time.time()

        _, reward_l, dones = env.step2(action_r.astype('float32'), action_l.astype('float32'))


        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    print("end")

if args.store_video:
    print('store video')
    env.stop_video_recording()
    env.turn_off_visualization()
