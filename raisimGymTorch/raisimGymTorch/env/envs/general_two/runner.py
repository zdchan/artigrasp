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
from raisimGymTorch.helper import rotations, label_gen_new, label_gen_final, frame_dict

data_version = "chiral_220223"

weight_path_articulate_l = '/../../left_fixed/2023-05-04-16-51-56/full_6600_l.pt'
weight_path_articulate_r = '/../../multi_obj_arti/2023-05-04-16-43-43/full_4300_r.pt'


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
parser.add_argument('-nr', '--num_repeats', type=int, default=2000)
parser.add_argument('-debug', '--debug', action="store_true")
parser.add_argument('-lr', '--log_rewards', action="store_true")
parser.add_argument('-re', '--load_trained_policy', action="store_true")
parser.add_argument('-random','--random', help='randomized goal obj angle', action="store_true")
parser.add_argument('-renew', '--renew', help='update labels every iteration', action="store_true")

args = parser.parse_args()
mode = args.mode
weight_path = args.weight
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
grasp_steps = 100
trail_steps = 200
reward_clip = -2.0

num_repeats = args.num_repeats
activations = nn.LeakyReLU
output_activation = nn.Tanh

test_inference = args.test


# get goal labels
original_labels_arti, original_labels_grasp, shuffle_label = label_gen_final.label_train(num_repeats, False, False)
processed_data, obj_list, left_kind_list, right_kind_list = label_gen_final.pose_gen(shuffle_label, num_repeats, False)
print(obj_list)
print(left_kind_list) #
print(right_kind_list)

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
trans_steps = 0
n_steps_r = grasp_steps + trail_steps + trans_steps
n_steps_l = trail_steps + grasp_steps + trans_steps
total_steps_r = n_steps_r * env.num_envs
total_steps_l = n_steps_l * env.num_envs

avg_rewards_r = []
contact_rewards_r = []
impulse_rewards_r = []
pos_rewards_r = []
pose_rewards_r = []
rel_obj_vel_rewards_r = []
body_vel_rewards_r = []
body_qvel_rewards_r = []
obj_angle_rewards_r = []
obj_avel_rewards_r = []
obj_vel_rewards_r = []
obj_qvel_rewards_r = []
ave_angle_rewards = []


avg_rewards_l = []
contact_rewards_l = []
impulse_rewards_l = []
pos_rewards_l = []
pose_rewards_l = []
rel_obj_vel_rewards_l = []
rel_obj_pos_rewards_l = []
body_vel_rewards_l = []
body_qvel_rewards_l = []
obj_vel_rewards_l = []
obj_qvel_rewards_l = []
obj_pos_rewards_l = []
obj_pose_rewards_l = []

# RL network
actor_l = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_grasp, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

critic_l = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_critic, 1), device)

actor_r = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_grasp, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_critic, 1), device)

if mode == 'retrain':
    test_dir = True
else:
    test_dir = False

saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp",
                                       task_path + "/runner.py"], test_dir=test_dir)

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
if args.load_trained_policy:
    load_param(saver.data_dir.split('eval')[0] + weight_path_articulate_l, env, actor_l, critic_l, ppo_l.optimizer, saver.data_dir, cfg_grasp)
    load_param(saver.data_dir.split('eval')[0] + weight_path_articulate_r, env, actor_r, critic_r, ppo_r.optimizer, saver.data_dir, cfg_grasp)


env.reset_state(qpos_reset_r2,
                qpos_reset_l2,
                np.zeros((num_envs,51),'float32'),
                np.zeros((num_envs,51),'float32'),
                obj_pose_reset2
                )

env.set_goals_r2(final_obj_pos2.astype('float32'),
                 final_ee_r2.astype('float32'),
                 final_pose_r2.astype('float32'),
                 final_qpos_r2.astype('float32'),
                 final_contacts_r2.astype('float32'),
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
env.reset_state(qpos_reset_r2,
                qpos_reset_l2,
                np.zeros((num_envs, 51), 'float32'),
                np.zeros((num_envs, 51), 'float32'),
                obj_pose_reset2
                )
if args.debug:
    time.sleep(1200)

for update in range(args.num_iterations):
    start = time.time()

    opened = np.zeros((num_envs, 1), 'float32')

    reward_ll_sum_r = 0
    reward_ll_sum_l = 0
    reward_angle_sum = 0
    done_sum = 0
    average_dones = 0.

    contact_reward_r = 0
    impulse_reward_r = 0
    pos_reward_r = 0
    pose_reward_r = 0
    rel_obj_vel_reward_r = 0
    body_vel_reward_r = 0
    body_qvel_reward_r = 0

    obj_angle_reward_r = 0
    obj_avel_reward_r = 0
    obj_vel_reward_r = 0
    obj_qvel_reward_r = 0

    contact_reward_l = 0
    impulse_reward_l = 0
    pos_reward_l = 0
    pose_reward_l = 0
    rel_obj_vel_reward_l = 0
    rel_obj_pos_reward_l = 0
    body_vel_reward_l = 0
    body_qvel_reward_l = 0

    obj_vel_reward_l = 0
    obj_qvel_reward_l = 0
    obj_pos_reward_l = 0
    obj_pose_reward_l = 0


    ### Evaluate trained model visually (note always the first environment gets visualized)

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor_r.architecture.state_dict(),
            'actor_distribution_state_dict': actor_r.distribution.state_dict(),
            'critic_architecture_state_dict': critic_r.architecture.state_dict(),
            'optimizer_state_dict': ppo_r.optimizer.state_dict(),
        }, saver.data_dir + "/full_" + str(update) + '_r.pt')

        torch.save({
            'actor_architecture_state_dict': actor_l.architecture.state_dict(),
            'actor_distribution_state_dict': actor_l.distribution.state_dict(),
            'critic_architecture_state_dict': critic_l.architecture.state_dict(),
            'optimizer_state_dict': ppo_l.optimizer.state_dict(),
        }, saver.data_dir + "/full_" + str(update) + '_l.pt')

        env.save_scaling(saver.data_dir, str(update))

    random_noise_angle = final_obj_angle2.copy()
    ave_angle_show = final_obj_angle2.copy()
    rand_temp = np.float32(np.random.uniform(0.5, 1.5, (num_envs, 1)).copy())
    for i in range(num_envs):
        if right_kind_list[i] == 8 or left_kind_list[i] == 8:
            random_noise_angle[i] = rand_temp[i]
            ave_angle_show[i] = rand_temp[i]
        else:
            random_noise_angle[i] = final_obj_angle2[i]
            ave_angle_show[i] = 0
    print(f"ave goal angle: {np.sum(np.abs(ave_angle_show)) / (num_envs*0.5)}")
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

    is_arti = (right_kinds[:] == 8).reshape(-1, 1)

    env.control_switch_all(left_kinds, right_kinds)
    # 7: the label doesn't contain this hand
    # 8: the hand is for articulation
    # 9: the hand is for grasping


    for step in range(n_steps_r):
        obs_r, obs_l = env.observe()
        obs_r = obs_r.astype('float32')
        obs_l = obs_l.astype('float32')

        action_r = ppo_r.act(obs_r)

        action_l = ppo_l.act(obs_l)

        reward_r, reward_l, dones = env.step2(action_r.astype('float32'), action_l.astype('float32'))
        reward_r = np.clip(reward_r, -10.0, 10.0)
        reward_l = np.clip(reward_l, -10.0, 10.0)

        ppo_r.step(value_obs=obs_r, rews=reward_r, dones=dones)
        ppo_l.step(value_obs=obs_l, rews=reward_l, dones=dones)

        current_state = (np.abs(obs_r[:, -3] - obj_pose_reset2[:, -1]) > 0.3).reshape(-1, 1)
        opened[:] += (current_state - current_state * opened) * is_arti

        reward_ll_sum_r = reward_ll_sum_r + np.sum(reward_r)
        reward_ll_sum_l = reward_ll_sum_l + np.sum(reward_l)
        reward_angle_sum += np.sum(np.abs(obs_r[:, -1]))

    obs_r, obs_l = env.observe()
    obs_r = obs_r[:].astype('float32')
    obs_l = obs_l.astype('float32')

    final_angle_diff = np.abs(obs_r[:, -1]).reshape(-1, 1)
    succeed_arti = opened * current_state

    switch_mode = False

    # update policy
    ppo_r.update(actor_obs=obs_r, value_obs=obs_r, log_this_iteration=update % 10 == 0, update=update)
    ppo_l.update(actor_obs=obs_l, value_obs=obs_l, log_this_iteration=update % 10 == 0, update=update)

    average_ll_performance_l = reward_ll_sum_l / total_steps_l
    average_ll_performance_r = reward_ll_sum_r / total_steps_r
    average_angle_reward = reward_angle_sum / (total_steps_r * 0.5)

    avg_rewards_r.append(average_ll_performance_r)
    avg_rewards_l.append(average_ll_performance_l)
    ave_angle_rewards.append(average_angle_reward)

    actor_r.distribution.enforce_minimum_std((torch.ones(act_dim) * 0.2).to(device))
    actor_l.distribution.enforce_minimum_std((torch.ones(act_dim) * 0.2).to(device))

    end = time.time()

    mean_file_name_r = saver.data_dir + "/rewards_r.txt"
    mean_file_name_l = saver.data_dir + "/rewards_l.txt"
    angle_diff_file_name = saver.data_dir + "/angle_diff.txt"

    np.savetxt(mean_file_name_r, avg_rewards_r)
    np.savetxt(mean_file_name_l, avg_rewards_l)
    np.savetxt(angle_diff_file_name, ave_angle_rewards)

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward_r: ", '{:0.10f}'.format(average_ll_performance_r)))
    print('{:<40} {:>6}'.format("average ll reward_l: ", '{:0.10f}'.format(average_ll_performance_l)))
    # print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps_r / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps_r / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('{:<40} {:>6}'.format("angle diff: ", '{:0.10f}'.format(average_angle_reward)))
    print('{:<40} {:>6}'.format("success arti rate: ", '{:0.10f}'.format(np.sum(succeed_arti) / (num_envs * 0.5))))
    print('{:<40} {:>6}'.format("ave final success arti angle error: ", '{:0.10f}'.format(
        np.sum(final_angle_diff * succeed_arti) / (np.sum(succeed_arti) + 1e-6))))

    print('std: ')
    print(np.exp(actor_r.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')

    if np.isnan(np.isnan(np.exp(actor_r.distribution.std.cpu().detach().numpy())).any()):
        print('resetting env')
        env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
                     cfg['environment'])

