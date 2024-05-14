from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import floating_evaluation as mano
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.floating_evaluation import NormalSampler
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
ref_r = [0.09566994, 0.00638343, 0.0061863]
ref_l = [-0.09566994, 0.00638343, 0.0061863]
path_mean_r = os.path.join("./../rsc/mano_double/right_pose_mean.txt")
path_mean_l = os.path.join("./../rsc/mano_double/left_pose_mean.txt")
pose_mean_r = np.loadtxt(path_mean_r)
pose_mean_l = np.loadtxt(path_mean_l)

weight_path_articulate_l = '/../../../general_two/2023-05-15-10-18-54/full_350_l.pt'
weight_path_articulate_r = '/../../../general_two/2023-05-15-10-18-54/full_350_r.pt'


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

if args.grasp:
    exp = "grasp"
    num_envs = 30
    num_repeats = 30
else:
    exp = "arti"
    num_envs = 5
    num_repeats = 5


pos_bias = np.loadtxt("./data_all/pos_bias.txt")
angle_bias = np.loadtxt("./data_all/angle_bias.txt")
print("pos_bias: ", pos_bias)
print("angle_bias: ", angle_bias)

start_offset = 0
pre_grasp_steps = 100
trail_steps = 200
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

if exp == "arti":
    label_list = arti_labels_list
else:
    label_list = grasp_labels_list

shuffle_label = []
for env_id in range(num_envs):
    shuffle_label.append(label_list[0])
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
ob_dim_grasp_l = env.num_obs_l
ob_dim_critic = ob_dim_grasp
ob_dim_critic_l = ob_dim_grasp_l
act_dim = env.num_acts

# Training
grasp_steps = pre_grasp_steps

if exp == "grasp":
    trans_steps = 300
else:
    trans_steps = 0
n_steps_r = grasp_steps + trail_steps + trans_steps
n_steps_l = trail_steps + grasp_steps + trans_steps
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

final_ave_error = np.zeros((num_envs, 1), 'float32')
succeed_arti = np.zeros((num_envs, 1), 'float32')
average_sim_distance = np.zeros((num_envs, 1), 'float32')
final_ave_error_all = np.zeros((num_envs, 1), 'float32')
average_sim_distance_all = np.zeros((num_envs, 1), 'float32')
final_ave_error_once = np.zeros((num_envs, 1), 'float32')
succeed_arti_once = np.zeros((num_envs, 1), 'float32')
average_sim_distance_once = np.zeros((num_envs, 1), 'float32')


final_pos_error = np.zeros((num_envs, 1), 'float32')
final_angle_error = np.zeros((num_envs, 1), 'float32')
final_pos_error_all = np.zeros((num_envs, 1), 'float32')
final_angle_error_all = np.zeros((num_envs, 1), 'float32')
succeed_grasp = np.zeros((num_envs, 1), 'float32')

if record:
    trans_r = np.zeros((n_steps_r, 3))
    rot_r = np.zeros((n_steps_r, 3))
    pose_r = np.zeros((n_steps_r, 45))
    trans_l = np.zeros((n_steps_r, 3))
    rot_l = np.zeros((n_steps_r, 3))
    pose_l = np.zeros((n_steps_r, 45))
    trans_obj = np.zeros((n_steps_r, 3))
    rot_obj = np.zeros((n_steps_r, 3))
    angle_obj = np.zeros((n_steps_r, 1))

for update in range(len(label_list)):
    start = time.time()
    reward_angle_sum = np.zeros((num_envs, 1), 'float32')
    sim_distance = np.zeros((num_envs, 1), 'float32')
    opened = np.zeros((num_envs, 1), 'float32')
    grasped = np.zeros((num_envs, 1), 'float32')

    shuffle_label = []
    for env_id in range(num_envs):
        shuffle_label.append(label_list[update])
    processed_data, obj_list, left_kind_list, right_kind_list = label_gen_final.pose_gen(shuffle_label, num_repeats,
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

    # if args.random:
    random_noise_angle = final_obj_angle2.copy()
    if exp == "arti":
        for i in range(num_envs):
            random_noise_angle[i] = 0.5 + i * 1 / (num_envs - 1)

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

    obj_pos_goal = obj_pose_reset2.copy()
    if exp == "grasp":
        obj_pos_goal[:, 2] += 0.3
        obj_pos_goal[:, :3] += pos_bias
        obj_euler = rotations.quat2euler(obj_pose_reset2[:, 3:7])
        obj_euler[:, 0] += angle_bias[:, 0]
        obj_euler[:, 1] += angle_bias[:, 1]
        obj_euler[:, 2] += angle_bias[:, 2]
        obj_pos_goal[:, 3:7] = rotations.euler2quat(obj_euler)
    env.set_obj_goal(random_noise_angle, obj_pos_goal)

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

    synthesis_right_kinds = right_kinds.copy()
    synthesis_left_kinds = left_kinds.copy()
    for i in range(num_envs):
        if synthesis_right_kinds[i] == 9:
            synthesis_right_kinds[i] = 10
        if synthesis_left_kinds[i] == 9:
            synthesis_left_kinds[i] = 10

    env.control_switch_all(left_kinds, right_kinds)

    for step in range(n_steps_r):

        if step == grasp_steps + trail_steps and exp == "grasp":
            print("start synthesis")
            env.control_switch_all(synthesis_left_kinds, synthesis_right_kinds)

        obs_r, obs_l = env.observe()
        obs_r = obs_r.astype('float32')
        obs_l = obs_l.astype('float32')

        global_state = env.get_global_state().astype('float32')

        if record:
            trans_r[step, :] = global_state[0, 6:9] - ref_r
            axis, angle = rotations.quat2axisangle(global_state[0, 9:13])
            rot_r[step, :] = axis * angle
            # pose_r[step, :] = global_state[0, 13:58] - pose_mean_r
            temp_pose = global_state[0, 13:58].reshape(15, 3)
            for j in range(15):
                pose_r[step, 3*j:3*j+3] = rotations.euler2axisangle(temp_pose[j].reshape(1,3))
            pose_r[step, :] = pose_r[step, :] - pose_mean_r

            trans_l[step, :] = global_state[0, 58:61] - ref_l
            axis, angle = rotations.quat2axisangle(global_state[0, 61:65])
            rot_l[step, :] = axis * angle
            # pose_l[step, :] = global_state[0, 65:110] - pose_mean_l
            temp_pose = global_state[0, 65:110].reshape(15, 3)
            for j in range(15):
                pose_l[step, 3*j:3*j+3] = rotations.euler2axisangle(temp_pose[j].reshape(1,3))
            pose_l[step, :] = pose_l[step, :] - pose_mean_l

            trans_obj[step, :] = global_state[0, 110:113]
            axis, angle = rotations.quat2axisangle(global_state[0, 113:117])
            rot_obj[step, :] = axis * angle
            angle_obj[step, :] = global_state[0, 117:118]

        action_r = actor_r.architecture.architecture(torch.from_numpy(obs_r.astype('float32')).to(device))
        action_r = action_r.cpu().detach().numpy()

        action_l = actor_l.architecture.architecture(torch.from_numpy(obs_l.astype('float32')).to(device))
        action_l = action_l.cpu().detach().numpy()

        frame_start = time.time()

        _, reward_l, dones = env.step2(action_r.astype('float32'), action_l.astype('float32'))

        reward_angle_sum += np.abs(global_state[:, 4].reshape(-1, 1))
        sim_distance += np.abs(global_state[:, 2]).reshape(-1, 1)

        if exp == "grasp":
            current_grasped_state = (-global_state[:, 5] > 0.1).reshape(-1, 1)
            grasped += current_grasped_state - current_grasped_state * grasped
        elif exp == "arti":
            current_opened_state = (np.abs(global_state[:, 3] - obj_pose_reset2[:, -1]) > 0.3).reshape(-1, 1)
            opened += current_opened_state - current_opened_state * opened

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    if record:
        data = {}
        data['right_hand'] = {}
        data['left_hand'] = {}
        data['laptop'] = {}
        data['right_hand']['trans'] = np.float32(trans_r)
        data['right_hand']['rot'] = np.float32(rot_r)
        data['right_hand']['pose'] = np.float32(pose_r)
        data['left_hand']['trans'] = np.float32(trans_l)
        data['left_hand']['rot'] = np.float32(rot_l)
        data['left_hand']['pose'] = np.float32(pose_l)
        data['laptop']['trans'] = np.float32(trans_obj)
        data['laptop']['rot'] = np.float32(rot_obj)
        data['laptop']['angle'] = np.float32(angle_obj)
        np.save("./data_all/ours.npy", data)
        print("recorded")

    print("end")

    if exp == "arti":
        opened -= (global_state[:, 3] > 2).reshape(-1, 1) * opened
        opened -= (np.abs(global_state[:, 2]) > 0.4).reshape(-1, 1) * opened
        print("succeed once: ", opened.transpose())
        print("succeed: ", (opened * current_opened_state).transpose())
        print("average angle differnence: ", (reward_angle_sum / n_steps_r).transpose())
        print("average simulated distance: ", (sim_distance / n_steps_r).transpose())
        print("final angle differnence: ", (np.abs(global_state[:, 4]).reshape(-1, 1)).transpose())
        succeed_arti_once += opened
        succeed_arti += opened * current_opened_state
        final_ave_error_once += np.abs(global_state[:, 4]).reshape(-1, 1) * opened
        final_ave_error += np.abs(global_state[:, 4]).reshape(-1, 1) * opened * current_opened_state
        final_ave_error_all += np.abs(global_state[:, 4]).reshape(-1, 1)
        average_sim_distance_once += (sim_distance / n_steps_r) * opened
        average_sim_distance += (sim_distance / n_steps_r) * opened * current_opened_state
        average_sim_distance_all += sim_distance / n_steps_r

    else:
        print("success: ", (grasped * current_grasped_state).transpose())
        print("final pos error: ", np.abs(global_state[:, 0].reshape(-1, 1)).transpose())
        print("final angle error: ", np.abs(global_state[:, 1].reshape(-1, 1)).transpose())
        final_pos_error_all += np.abs(global_state[:, 0].reshape(-1, 1))
        final_angle_error_all += np.abs(global_state[:, 1]).reshape(-1, 1)
        succeed_grasp += grasped * current_grasped_state
        final_pos_error += np.abs(global_state[:, 0]).reshape(-1, 1) * grasped * current_grasped_state
        final_angle_error += np.abs(global_state[:, 1]).reshape(-1, 1) * grasped * current_grasped_state

print(" ")
result_list = {}
direct_save = "./data_all/evaluation_result"
if not op.exists(direct_save):
    os.system(f"mkdir -p {direct_save}")
if exp == "arti":
    print("succeed once rate: ", (succeed_arti_once / len(arti_labels_list)).transpose())
    print("success rate: ", (succeed_arti / (len(arti_labels_list))).transpose())
    print("average final angle differnence once: ", (final_ave_error_once / (succeed_arti_once + 1e-6)).transpose())
    print("average final angle differnence: ", (final_ave_error / (succeed_arti + 1e-6)).transpose())
    print("average final angle differnence all: ", (final_ave_error_all / (len(arti_labels_list))).transpose())
    print("average simulated distance once: ", (average_sim_distance_once / (succeed_arti_once + 1e-6)).transpose())
    print("average simulated distance: ", (average_sim_distance / (succeed_arti + 1e-6)).transpose())
    print("average simulated distance all: ", (average_sim_distance_all / (len(arti_labels_list))).transpose())
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print("succeed once rate: ", np.sum(succeed_arti_once / len(arti_labels_list)) / num_envs)
    print("success rate: ", np.sum(succeed_arti / (len(arti_labels_list)) / num_envs))
    print("average final angle differnence once: ",
          np.sum(final_ave_error_once / (succeed_arti_once + 1e-6)) / num_envs)
    print("average final angle differnence: ", np.sum(final_ave_error / (succeed_arti + 1e-6)) / num_envs)
    print("average final angle differnence all: ", np.sum(final_ave_error_all / (len(arti_labels_list))) / num_envs)
    print("average simulated distance once: ",
          np.sum(average_sim_distance_once / (succeed_arti_once + 1e-6)) / num_envs)
    print("average simulated distance: ", np.sum(average_sim_distance / (succeed_arti + 1e-6)) / num_envs)
    print("average simulated distance all: ", np.sum(average_sim_distance_all / (len(arti_labels_list))) / num_envs)

    result_list['succeed_once_rate'] = np.sum(succeed_arti_once / len(arti_labels_list)) / num_envs
    result_list['success_rate'] = np.sum(succeed_arti / (len(arti_labels_list)) / num_envs)
    result_list['average_final_angle_differnence_once'] = np.sum(
        final_ave_error_once / (succeed_arti_once + 1e-6)) / num_envs
    result_list['average_final_angle_differnence'] = np.sum(final_ave_error / (succeed_arti + 1e-6)) / num_envs
    result_list['average_final_angle_differnence_all'] = np.sum(
        final_ave_error_all / (len(arti_labels_list))) / num_envs
    result_list['average_simulated_distance_once'] = np.sum(
        average_sim_distance_once / (succeed_arti_once + 1e-6)) / num_envs
    result_list['average_simulated_distance'] = np.sum(average_sim_distance / (succeed_arti + 1e-6)) / num_envs
    result_list['average_simulated_distance_all'] = np.sum(
        average_sim_distance_all / (len(arti_labels_list))) / num_envs
    if test:
        file_name = f"/ours_floating_arti_{obj_name}_test.txt"
    else:
        file_name = f"/ours_floating_arti_{obj_name}_train.txt"
    with open(direct_save + file_name, 'w') as f:
        for item in result_list:
            f.write("%s: " % item)
            f.write("%s\n" % result_list[item])
else:
    print("success rate: ", (succeed_grasp / len(grasp_labels_list)).transpose())
    print("average final pos error: ", (final_pos_error / (succeed_grasp + 1e-6)).transpose())
    print("average final pos error all: ", (final_pos_error_all / len(grasp_labels_list)).transpose())
    print("average final angle error: ", (final_angle_error / (succeed_grasp + 1e-6)).transpose())
    print("average final angle error all: ", (final_angle_error_all / len(grasp_labels_list)).transpose())
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print("success rate: ", np.sum(succeed_grasp / len(grasp_labels_list))/num_envs)
    print("average final pos error: ", np.sum(final_pos_error / (succeed_grasp + 1e-6))/num_envs)
    print("average final pos error all: ", np.sum(final_pos_error_all / len(grasp_labels_list))/num_envs)
    print("average final angle error: ", np.sum(final_angle_error / (succeed_grasp + 1e-6))/num_envs)
    print("average final angle error all: ", np.sum(final_angle_error_all / len(grasp_labels_list))/num_envs)

    result_list['success_rate'] = np.sum(succeed_grasp / len(grasp_labels_list))/num_envs
    result_list['average_final_pos_error'] = np.sum(final_pos_error / (succeed_grasp + 1e-6))/num_envs
    result_list['average_final_pos_error_all'] = np.sum(final_pos_error_all / len(grasp_labels_list))/num_envs
    result_list['average_final_angle_error'] = np.sum(final_angle_error / (succeed_grasp + 1e-6))/num_envs
    result_list['average_final_angle_error_all'] = np.sum(final_angle_error_all / len(grasp_labels_list))/num_envs
    if test:
        file_name = f"/ours_floating_grasp_{obj_name}_test.txt"
    else:
        file_name = f"/ours_floating_grasp_{obj_name}_train.txt"
    with open(direct_save + file_name, 'w') as f:
        for item in result_list:
            f.write("%s: " % item)
            f.write("%s\n" % result_list[item])
