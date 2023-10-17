from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import fixed_arti_evaluation as mano
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.fixed_arti_evaluation import NormalSampler
import os.path as op

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

exp_name = "fixed_arti_evaluation"

y_adjust_l = 0
z_adjust_l = 0

ref_frame = 0
height_desk = 0.5
xpos_desk = 0.3

weight_saved = '/../../../multi_obj_arti/2023-05-14-16-30-13/full_8600_r.pt'

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg_reg.yaml')
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default=exp_name)
parser.add_argument('-w', '--weight', type=str, default=weight_saved)
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
parser.add_argument('-itr', '--num_iterations', type=int, default=3001)
parser.add_argument('-nr', '--num_repeats', type=int, default=5)
parser.add_argument('-debug', '--debug', action="store_true")
parser.add_argument('-lr', '--log_rewards', action="store_true")
parser.add_argument('-obj', '--obj', help='which obj to test', type=str, default="box")
parser.add_argument('-test', '--test_set', help='test or train set', action="store_true")

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

obj_name = args.obj
test = args.test_set
if test:
    print("test set")
else:
    print("train set")

arti_labels_list = label_gen_final.label_eval_fixed(obj_name, test)
arti_label = []
for env_id in range(num_envs):
    arti_label.append(arti_labels_list[0])

processed_data, obj_list, left_kind_list, right_kind_list = label_gen_final.pose_gen(arti_label, num_repeats, False)
print(obj_list)

stage_dim = processed_data[0]
stage_pos = processed_data[1]
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
    obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}_fixed_base.urdf"))
env.load_multi_articulated(obj_path_list)

# Setting dimensions from environments
n_act = final_qpos_r[0].shape[0]
ob_dim_r = env.num_obs_r
act_dim = env.num_acts

# Training
grasp_steps = pre_grasp_steps
n_steps_r = grasp_steps + trail_steps
total_steps_r = n_steps_r * env.num_envs

# RL network

actor_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_r, 1), device)

test_dir = True

saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[], test_dir=test_dir)

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

load_param(saver.data_dir+weight_path, env, actor_r, critic_r, ppo_r.optimizer, saver.data_dir, cfg_grasp)

final_ave_error = np.zeros((num_envs, 1), 'float32')
final_ave_error_once = np.zeros((num_envs, 1), 'float32')
final_ave_error_all = np.zeros((num_envs, 1), 'float32')
succeed = np.zeros((num_envs, 1), 'float32')
succeed_once = np.zeros((num_envs, 1), 'float32')

for update in range(len(arti_labels_list)):
    start = time.time()
    reward_angle_sum = np.zeros((num_envs, 1), 'float32')
    opened = np.zeros((num_envs, 1), 'float32')

    arti_label = []
    for env_id in range(num_envs):
        arti_label.append(arti_labels_list[update])

    processed_data, obj_list, left_kind_list, right_kind_list = label_gen_final.pose_gen(arti_label, num_repeats, False)

    stage_dim = processed_data[0]
    stage_pos = processed_data[1]
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

    env.set_goals(final_obj_angle,
                  final_obj_pos,
                  final_ee_r,
                  final_ee_l,
                  final_pose_r,
                  final_pose_l,
                  final_qpos_r,
                  final_qpos_l,
                  final_contacts_r,
                  final_contacts_l,
                  )

    ### Add some noise to goal angle
    random_noise_angle = final_obj_angle.copy()
    for i in range(num_envs):
        random_noise_angle[i] = 0.5 + i * 1 / (num_envs - 1)

    env.set_goals(random_noise_angle,
                  final_obj_pos,
                  final_ee_r,
                  final_ee_l,
                  final_pose_r,
                  final_pose_l,
                  final_qpos_r,
                  final_qpos_l,
                  final_contacts_r,
                  final_contacts_l,
                  )

    ### Add some noise to initial hand position
    qpos_noisy_reset_r = qpos_reset_r.copy()
    qpos_noisy_reset_l = qpos_reset_l.copy()

    env.reset_state(qpos_noisy_reset_r,
                    qpos_noisy_reset_l,
                    np.zeros((num_envs, 51), 'float32'),
                    np.zeros((num_envs, 51), 'float32'),
                    obj_pose_reset
                    )

    left_kinds = np.float32(left_kind_list.copy())
    right_kinds = np.float32(right_kind_list.copy())

    env.control_switch_all(left_kinds, right_kinds)

    for step in range(n_steps_r):
        obs_r, _ = env.observe()
        obs_r = obs_r[:, :].astype('float32')
        if np.isnan(obs_r).any():
            np.savetxt(saver.data_dir + "/nan_obs.txt", obs_r)

        action_r = actor_r.architecture.architecture(torch.from_numpy(obs_r.astype('float32')).to(device))
        action_l = torch.zeros_like(action_r)

        action_grasp_r = action_r.cpu().detach().numpy()
        action_grasp_l = action_l.cpu().detach().numpy()
        frame_start = time.time()

        reward_r, _, dones = env.step(action_grasp_r.astype('float32'), action_grasp_l.astype('float32'))

        reward_angle_sum += np.abs(obs_r[:, -1].reshape(-1, 1))
        current_opened_state = (np.abs(obs_r[:, -3] - obj_pose_reset[:, -1]) > 0.3).reshape(-1, 1)
        opened += current_opened_state - current_opened_state * opened

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    print("end")
    opened -= (obs_r[:, 3] > 2).reshape(-1, 1) * opened
    print("succeed once: ", opened.transpose())
    print("succeed: ", (opened * current_opened_state).transpose())
    print("average angle differnence: ", (reward_angle_sum / n_steps_r).transpose())
    print("final angle differnence: ", (np.abs(obs_r[:, -1]).reshape(-1, 1)).transpose())
    succeed_once += opened
    succeed += opened * current_opened_state
    final_ave_error += np.abs(obs_r[:, -1]).reshape(-1, 1) * opened * current_opened_state
    final_ave_error_once += np.abs(obs_r[:, -1]).reshape(-1, 1) * opened
    final_ave_error_all += np.abs(obs_r[:, -1]).reshape(-1, 1)

print(" ")
print("succeed once rate: ", (succeed_once / len(arti_labels_list)).transpose())
print("success rate: ", (succeed / len(arti_labels_list)).transpose())
print("final average angle differnence succeed once: ", (final_ave_error_once / (succeed_once + 1e-6)).transpose())
print("final average angle differnence succeed: ", (final_ave_error / (succeed + 1e-6)).transpose())
print("final average angle differnence all: ", (final_ave_error_all / (len(arti_labels_list))).transpose())
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

print(" ")
result_list = {}
direct_save = "./data_all/evaluation_result"
if not op.exists(direct_save):
		os.system(f"mkdir -p {direct_save}")
print("succeed once rate: ", np.sum(succeed_once / len(arti_labels_list)) / num_envs)
print("success rate: ", np.sum(succeed / len(arti_labels_list)) / num_envs)
print("final average angle differnence succeed once: ", np.sum(final_ave_error_once / (succeed_once + 1e-6)) / num_envs)
print("final average angle differnence succeed: ", np.sum(final_ave_error / (succeed + 1e-6)) / num_envs)
print("final average angle differnence all: ", np.sum(final_ave_error_all / (len(arti_labels_list))) / num_envs)
result_list["succeed once rate"] = np.sum(succeed_once / len(arti_labels_list)) / num_envs
result_list["success rate"] = np.sum(succeed / len(arti_labels_list)) / num_envs
result_list["final average angle differnence succeed once"] = np.sum(final_ave_error_once / (succeed_once + 1e-6)) / num_envs
result_list["final average angle differnence succeed"] = np.sum(final_ave_error / (succeed + 1e-6)) / num_envs
result_list["final average angle differnence all"] = np.sum(final_ave_error_all / (len(arti_labels_list))) / num_envs
if test:
    file_name = f"/ours_fixed_{obj_name}_test.txt"
else:
    file_name = f"/ours_fixed_{obj_name}_train.txt"
with open(direct_save + file_name, 'w') as f:
    for item in result_list:
        f.write("%s: " % item)
        f.write("%s\n" % result_list[item])

