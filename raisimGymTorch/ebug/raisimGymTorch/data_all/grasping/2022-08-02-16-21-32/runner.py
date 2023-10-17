from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import articulated_hira as mano
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse
from raisimGymTorch.helper import rotations
# from raisimGymTorch.helper import utils
import joblib


grasp_frame_dict = {
                    "ketchup": 485, 
                    "mixer": 80, 
                    "laptop": 110, 
                    "waffleiron": 52, #s3
                    "phone": 102,
                    "box": 60,
                    "notebook": 46, #s8, use 03
                    "portamaker": 264, #s7
                   }

init_frame_dict =  {
                    "ketchup": 290, 
                    "mixer": 40, 
                    "laptop": 100, 
                    "waffleiron": 28,
                    "phone": 88,
                    "box": 36,
                    "notebook": 30,
                    "portamaker": 226,
                   }

contact_dict =     {
                    "ketchup": np.array([[0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1]]), 
                    "mixer": np.array([[1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1]]), 
                    "phone": np.array([[0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1]]),
                    "box": np.array([[1,0,0,1,0,1,1,0,0,1,0,1,1,1,0,1]])
                   }

data_version = "chiral_220223"

seq_name = 'portamaker_use_01'
obj_name = seq_name.split('_')[0]

grasp_frame = grasp_frame_dict[obj_name] # or 498, ketchup


init_frame = init_frame_dict[obj_name] # ketchup

y_adjust_l = 0
z_adjust_l = 0

ref_frame = 0
height_desk = 0.5
xpos_desk = 0.3

#final_contacts_l = np.array([[0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1]]) #ketchup 498
#final_contacts_l = contact_dict[obj_name]

#dim_not_to_mirror = np.array([1,2]+[3*n for n in range(1,17)])
#dim_not_to_mirror_pose = dim_not_to_mirror[2:] - 3
#dim_to_mirror = np.array([3*n for n in range(21)])


MANO_TO_CONTACT = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 6,
    9: 7,
    10:8,
    11:9,
    12:9,
    13:10,
    14:11,
    15:12,
    16:12,
    17:13,
    18:14,
    19:15,
    20:15,
}



# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg_reg.yaml')
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default="grasping")
parser.add_argument('-w', '--weight', type=str, default='2022-06-13-17-21-55/full_3000_l.pt')
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-pr','--prior', action="store_true")
parser.add_argument('-o', '--obj_id',type=int, default=1)
parser.add_argument('-t','--test', action="store_true")
parser.add_argument('-mc','--mesh_collision', action="store_true")
parser.add_argument('-ao','--all_objects', action="store_true")
parser.add_argument('-ev','--evaluate', action="store_true")
parser.add_argument('-to','--test_object_set', type=int, default=-1)
parser.add_argument('-ac','--all_contact', action="store_true")
parser.add_argument('-seed','--seed', type=int, default=1)
parser.add_argument('-itr','--num_iterations', type=int, default=3001)
parser.add_argument('-nr','--num_repeats', type=int, default=200)
parser.add_argument('-debug','--debug', action="store_true")
parser.add_argument('-lr','--log_rewards', action="store_true")

args = parser.parse_args()
mode = args.mode
weight_path = args.weight

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
cfg = YAML().load(open(task_path+'/cfgs/' + args.cfg, 'r'))

if args.seed != 1:
    cfg['seed']=args.seed

num_envs = cfg['environment']['num_envs']
start_offset = 0
pre_grasp_steps = 60
trail_steps = 135
reward_clip = -2.0
torch.set_default_dtype(torch.double)

num_repeats= args.num_repeats
activations = nn.LeakyReLU
output_activation = nn.Tanh

output_act = False
inference_flag = False
all_obj_train = True if args.all_objects else False
all_train = False
test_inference = args.test
train_obj_id = args.obj_id

mano_params = np.load(
    home_path + f"/rsc/mano_double/{seq_name}.params.mano.npy", allow_pickle=True
).item()
obj_params = np.load(
    home_path + f"/rsc/arctic/{obj_name}/{seq_name}.gt.object.npy", allow_pickle=True
)

path_ee_r = os.path.join(home_path + f"/rsc/mano_double/ee/{obj_name}/{grasp_frame:05d}_r.txt")
path_ee_l = os.path.join(home_path + f"/rsc/mano_double/ee/{obj_name}/{grasp_frame:05d}_l.txt")
path_height = os.path.join(home_path + f"/rsc/mano_double/ee/{obj_name}/{init_frame:05d}_min.txt")
path_contact_l = os.path.join(home_path + f"/rsc/mano_double/contact/{obj_name}/{grasp_frame:05d}.txt")
path_mean_r = os.path.join(home_path + f"/rsc/mano_double/right_pose_mean.txt")
path_mean_l = os.path.join(home_path + f"/rsc/mano_double/left_pose_mean.txt")
path_ref_r = os.path.join(home_path + f"/rsc/mano_double/ee/{obj_name}/{ref_frame:05d}_r.txt")
path_ref_l = os.path.join(home_path + f"/rsc/mano_double/ee/{obj_name}/{ref_frame:05d}_l.txt")
ref_r = np.loadtxt(path_ref_r)
ref_l = np.loadtxt(path_ref_l)
height_ref = np.loadtxt(path_height)[2]


### Load data labels for all objects


# data directly from ARCTIC, mano, final
pose_mean_r = np.loadtxt(path_mean_r)
pose_mean_l = np.loadtxt(path_mean_l)

mano_rot_final_r = mano_params["right"]["rot"][grasp_frame] # 3
mano_pose_final_r = mano_params["right"]["pose"][grasp_frame] + pose_mean_r # 45
mano_trans_final_r = mano_params["right"]["trans"][grasp_frame] + ref_r[0] # 3
mano_rot_final_l = mano_params["left"]["rot"][grasp_frame]
mano_pose_final_l = mano_params["left"]["pose"][grasp_frame] + pose_mean_l
mano_trans_final_l = mano_params["left"]["trans"][grasp_frame] + ref_l[0]

mano_rot_final_r = rotations.axisangle2euler(mano_rot_final_r.reshape(-1,3)).reshape(-1)
mano_pose_final_r = rotations.axisangle2euler(mano_pose_final_r.reshape(-1,3)).reshape(-1)
mano_rot_final_l = rotations.axisangle2euler(mano_rot_final_l.reshape(-1,3)).reshape(-1)
mano_pose_final_l = rotations.axisangle2euler(mano_pose_final_l.reshape(-1,3)).reshape(-1)

# data directly from ARCTIC, object, final
gt_obj_angles_final = np.array([obj_params[:, 0][grasp_frame]])  # radian 1
gt_axis_angle_final = obj_params[:, 1:4][grasp_frame] # 3
gt_transl_final = obj_params[:, 4:][grasp_frame] / 1000# 3
gt_quat_final = rotations.axisangle2quat(gt_axis_angle_final)
#gt_obj_eular = np.array([-gt_obj_angles, 0, 0])
#gt_obj_quat = euler2quat(gt_obj_eular)
#gt_quat_t = quat_mul(gt_obj_eular, gt_quat_b)

# mirror the left hand to the right pose
final_qpos_r = np.concatenate((mano_trans_final_r, mano_rot_final_r, mano_pose_final_r))
final_qpos_l = np.concatenate((mano_trans_final_l, mano_rot_final_l, mano_pose_final_l))
#final_qpos_l[:, ~dim_not_to_mirror] = -final_qpos_l[:, ~dim_not_to_mirror]
final_obj_angle = gt_obj_angles_final
final_obj_pos = np.concatenate((gt_transl_final, gt_quat_final, gt_obj_angles_final))
final_pose_r = np.concatenate((mano_rot_final_r, mano_pose_final_r))
final_pose_l = np.concatenate((mano_rot_final_l, mano_pose_final_l))
#print(final_pose_l[39:])
#final_pose_l[:, ~dim_not_to_mirror_pose] = -final_pose_l[:, ~dim_not_to_mirror_pose]

final_ee_r = np.loadtxt(path_ee_r)
final_ee_l = np.loadtxt(path_ee_l)

final_contacts_r = np.array([[1,1,1,1,0,1,1,0,0,1,0,0,1,0,1,1]])
final_contacts_l = np.loadtxt(path_contact_l)
if args.debug:
    print(f"final_contacts_l: {final_contacts_l}")
#final_ee_l[:, dim_to_mirror] = - final_ee_l[:, dim_to_mirror]

# data directly from ARCTIC, mano, init
mano_rot_init_r = mano_params["right"]["rot"][init_frame] # 3
mano_pose_init_r = mano_params["right"]["pose"][init_frame] # 3
mano_trans_init_r = mano_params["right"]["trans"][init_frame] + ref_r[0] # 45
mano_rot_init_l = mano_params["left"]["rot"][init_frame] # 3
mano_pose_init_l = mano_params["left"]["pose"][init_frame] # 3
mano_trans_init_l = mano_params["left"]["trans"][init_frame] + ref_l[0] # 45

# get the true pose parameters
mano_pose_init_r += pose_mean_r
mano_pose_init_l += pose_mean_l

# transform axis angle to Euler for orientation
mano_rot_init_r = rotations.axisangle2euler(mano_rot_init_r.reshape(-1,3)).reshape(-1)
mano_pose_init_r = rotations.axisangle2euler(mano_pose_init_r.reshape(-1,3)).reshape(-1)
mano_rot_init_l = rotations.axisangle2euler(mano_rot_init_l.reshape(-1,3)).reshape(-1)
mano_pose_init_l = rotations.axisangle2euler(mano_pose_init_l.reshape(-1,3)).reshape(-1)


# data directly from ARCTIC, object, init
#gt_obj_angles_init = np.array([0])  # radian 1
gt_obj_angles_init = np.array([obj_params[:, 0][init_frame]]) # radian 1
gt_axis_angle_init = obj_params[:, 1:4][init_frame] # 3
gt_transl_init = obj_params[:, 4:][init_frame] / 1000 # 3
gt_quat_init = rotations.axisangle2quat(gt_axis_angle_init)

mano_trans_init_r[2] = mano_trans_init_r[2] - height_ref + height_desk
mano_trans_init_r[0] += xpos_desk
mano_trans_init_l[2] = mano_trans_init_l[2] - height_ref + height_desk + z_adjust_l
mano_trans_init_l[1] += y_adjust_l
mano_trans_init_l[0] += xpos_desk
gt_transl_init[2] = gt_transl_init[2] - height_ref + height_desk
gt_transl_init[0] += xpos_desk   

obj_pose_reset = np.concatenate((gt_transl_init, gt_quat_init, gt_obj_angles_init))
qpos_reset_r = np.concatenate((mano_trans_init_r, mano_rot_init_r, mano_pose_init_r))
qpos_reset_l = np.concatenate((mano_trans_init_l, mano_rot_init_l, mano_pose_init_l))

# put final object base pose and hand root pose to init pose
qpos_fake = np.zeros_like(final_qpos_l)
qpos_fake[6:] = final_qpos_l[6:]
qpos_fake[:3] = final_qpos_l[:3] + obj_pose_reset[:3] - final_obj_pos[:3] 
rot_mat_i = rotations.quat2mat(obj_pose_reset[3:7])
rot_mat_f = rotations.quat2mat(final_obj_pos[3:7])
rot_mat_h = rotations.euler2mat(final_qpos_l[3:6])
qpos_fake[3:6] = rotations.mat2euler(np.matmul(np.matmul(rot_mat_i, rot_mat_f.T), rot_mat_h))

obj_fake = np.zeros_like(obj_pose_reset)
obj_fake[:7] = obj_pose_reset[:7]
obj_fake[7] = final_obj_pos[7]

#qpos_reset = np.repeat(dict_labels[train_obj_id]['final_qpos'],num_repeats,0)
final_qpos_r = np.repeat(final_qpos_r.reshape(1,-1), num_repeats, 0)
final_qpos_l = np.repeat(final_qpos_l.reshape(1,-1), num_repeats, 0)
final_obj_angle = np.repeat(final_obj_angle.reshape(1,-1), num_repeats, 0)
final_obj_pos = np.repeat(final_obj_pos.reshape(1,-1), num_repeats, 0)
final_pose_r = np.repeat(final_pose_r.reshape(1,-1), num_repeats, 0)
final_pose_l = np.repeat(final_pose_l.reshape(1,-1), num_repeats, 0)
final_ee_r = np.repeat(final_ee_r.reshape(1,-1), num_repeats, 0)
final_ee_l = np.repeat(final_ee_l.reshape(1,-1), num_repeats, 0)
final_contacts_r = np.repeat(final_contacts_r.reshape(1,-1), num_repeats, 0)
final_contacts_l = np.repeat(final_contacts_l.reshape(1,-1), num_repeats, 0)

obj_pose_reset = np.repeat(obj_pose_reset.reshape(1,-1), num_repeats, 0) #(200, 8)
qpos_reset_r = np.repeat(qpos_reset_r.reshape(1,-1), num_repeats, 0) #(200, 51)
qpos_reset_l = np.repeat(qpos_reset_l.reshape(1,-1), num_repeats, 0) #(200, 51)

qpos_fake = np.repeat(qpos_fake.reshape(1,-1), num_repeats, 0)
obj_fake = np.repeat(obj_fake.reshape(1,-1), num_repeats, 0)

final_qpos_r = np.float64(final_qpos_r)
final_qpos_l = np.float64(final_qpos_l)
final_obj_pos = np.float64(final_obj_pos)
final_obj_angle = np.float64(final_obj_angle)
final_pose_r = np.float64(final_pose_r)
final_pose_l = np.float64(final_pose_l)
final_ee_r = np.float64(final_ee_r)
final_ee_l = np.float64(final_ee_l)
final_contacts_r = np.float64(final_contacts_r)
final_contacts_l = np.float64(final_contacts_l)

obj_pose_reset = np.float64(obj_pose_reset)
qpos_reset_r = np.float64(qpos_reset_r)
qpos_reset_l = np.float64(qpos_reset_l)

qpos_fake = np.float64(qpos_fake)
obj_fake = np.float64(obj_fake)

num_envs = final_qpos_r.shape[0]
cfg['environment']['hand_model_r'] = "rhand_mano_meshcoll.urdf" if args.mesh_collision else "rhand_mano.urdf"
cfg['environment']['hand_model_l'] = "lhand_mano_meshcoll.urdf" if args.mesh_collision else "lhand_mano.urdf"
cfg['environment']['num_envs'] = 1 if args.evaluate else num_envs
if args.debug:
    cfg['environment']['num_envs'] = 1
cfg["testing"] = True if test_inference else False
print('num envs', num_envs)

# Environment definition
env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
#env.load_object(obj_idx_stacked,obj_w_stacked, obj_dim_stacked, obj_type_stacked)
obj_path = os.path.join(f"{obj_name}/{obj_name}_fixed_base.urdf")
env.load_articulated(obj_path)

# Setting dimensions from environments
n_act = final_qpos_r[0].shape[0]
ob_dim_l = env.num_obs_r
act_dim = env.num_acts


# Training
grasp_steps = pre_grasp_steps
n_steps_l = grasp_steps + trail_steps
total_steps_l = n_steps_l * env.num_envs

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

actor_l = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], output_activation, activations, ob_dim_l, act_dim, output_act),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),
                         device)

critic_l = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], output_activation, activations, ob_dim_l, 1, output_act),
                           device)

if mode == 'retrain':
    test_dir=True
else:
    test_dir=False

saver = ConfigurationSaver(log_dir = exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp", task_path + "/runner.py"], test_dir=test_dir)
#tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

#time.sleep(1200)

ppo_l = PPO.PPO(actor=actor_l,
                critic=critic_l,
                num_envs=num_envs,
                num_transitions_per_env=n_steps_l,
                num_learning_epochs=4,
                gamma=0.996,
                lam=0.95,
                num_mini_batches=4,
                device=device,
                log_dir=saver.data_dir,
                shuffle_batch=False
                )

#time.sleep(1200)

if mode == 'retrain' or args.evaluate:
    load_param(saver.data_dir.split('eval')[0]+"/../"+weight_path, env, actor_l, critic_l, ppo_l.optimizer, saver.data_dir, args.cfg)

if args.debug:
    qpos_reset_l = final_qpos_l
    obj_pose_reset = final_obj_pos

env.reset_state(qpos_reset_r, 
                qpos_reset_l, 
                np.zeros((num_envs,51),'float64'), 
                np.zeros((num_envs,51),'float64'), 
                obj_pose_reset,
                )
#time.sleep(1200)
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
env.reset_state(qpos_reset_r, 
                qpos_reset_l, 
                np.zeros((num_envs,51),'float64'), 
                np.zeros((num_envs,51),'float64'), 
                obj_pose_reset
                )
if args.debug:
    time.sleep(1200)


for update in range(args.num_iterations):
    start = time.time()

    reward_ll_sum_r = 0
    reward_ll_sum_l = 0
    done_sum = 0
    average_dones = 0.

    contact_reward_l = 0
    impulse_reward_l = 0
    pos_reward_l = 0
    pose_reward_l = 0
    falling_reward_l = 0
    rel_obj_reward_l = 0
    body_vel_reward_l = 0
    body_qvel_reward_l = 0


    ### Evaluate trained model visually (note always the first environment gets visualized)
    
    

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor_l.architecture.state_dict(),
            'actor_distribution_state_dict': actor_l.distribution.state_dict(),
            'critic_architecture_state_dict': critic_l.architecture.state_dict(),
            'optimizer_state_dict': ppo_l.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'_l.pt')

        env.save_scaling(saver.data_dir, str(update))

    ### Add some noise to initial hand position
    random_noise_pos = np.random.uniform([-0.02, -0.02, 0.01],[0.02, 0.02, 0.01], (num_envs,3)).copy()
    random_noise_qpos = np.random.uniform(-0.05,0.05, (num_envs,48)).copy()
    qpos_noisy_reset_r = qpos_reset_r.copy()
    qpos_noisy_reset_l = qpos_reset_l.copy()
    qpos_noisy_reset_r[:,:3] += random_noise_pos[:,:3]
    qpos_noisy_reset_r[:,3:] += random_noise_qpos[:,:]
    qpos_noisy_reset_l[:,:3] += random_noise_pos[:,:3]
    qpos_noisy_reset_l[:,3:] += random_noise_qpos[:,:]


    env.reset_state(qpos_noisy_reset_r, 
                    qpos_noisy_reset_l, 
                    np.zeros((num_envs,51),'float64'), 
                    np.zeros((num_envs,51),'float64'), 
                    obj_pose_reset
                    )

    for step in range(n_steps_l):
        _, obs_l = env.observe()
        obs_l = obs_l[:,:-3].astype('float64')
        if np.isnan(obs_l).any():
            np.savetxt(saver.data_dir + "/nan_obs.txt", obs_l)

        action_l = ppo_l.observe(obs_l)
        action_r = np.zeros_like(action_l)
        
        _, reward_l, dones = env.step(action_r.astype('float64'), action_l.astype('float64'))
        reward_l.clip(min=reward_clip)

        ppo_l.step(value_obs=obs_l, rews=reward_l, dones=dones)
        done_sum = done_sum + np.sum(dones)
        reward_ll_sum_l = reward_ll_sum_l + np.sum(reward_l)
        if args.log_rewards:
            contact_reward_l = contact_reward_l + env.get_reward_info()[0]["contact_reward"]
            impulse_reward_l = impulse_reward_l + env.get_reward_info()[0]["impulse_reward"]
            pos_reward_l = pos_reward_l + env.get_reward_info()[0]["pos_reward"]
            pose_reward_l = pose_reward_l + env.get_reward_info()[0]["pose_reward"]
            falling_reward_l = falling_reward_l + env.get_reward_info()[0]["falling_reward"]
            rel_obj_reward_l = rel_obj_reward_l + env.get_reward_info()[0]["rel_obj_reward_"]
            body_vel_reward_l = body_vel_reward_l + env.get_reward_info()[0]["body_vel_reward_"]
            body_qvel_reward_l = body_qvel_reward_l + env.get_reward_info()[0]["body_qvel_reward_"]

    _, obs_l = env.observe()
    obs_l = obs_l[:,:-3].astype('float64')

    # update policy
    ppo_l.update(actor_obs=obs_l, value_obs=obs_l, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_ll_sum_l / total_steps_l
    if args.log_rewards:
        contact_reward_l = contact_reward_l / n_steps_l
        impulse_reward_l = impulse_reward_l / n_steps_l
        pos_reward_l = pos_reward_l / n_steps_l
        pose_reward_l = pose_reward_l / n_steps_l
        falling_reward_l = falling_reward_l / n_steps_l
        rel_obj_reward_l = rel_obj_reward_l / n_steps_l
        body_vel_reward_l = body_vel_reward_l / n_steps_l
        body_qvel_reward_l = body_qvel_reward_l / n_steps_l
    average_dones = done_sum / total_steps_l

    avg_rewards.append(average_ll_performance)
    if args.log_rewards:
        contact_rewards.append(contact_reward_l)
        impulse_rewards.append(impulse_reward_l)
        pos_rewards.append(pos_reward_l)
        pose_rewards.append(pose_reward_l)
        falling_rewards.append(falling_reward_l)
        rel_obj_rewards.append(rel_obj_reward_l)
        body_vel_rewards.append(body_vel_reward_l)
        body_qvel_rewards.append(body_qvel_reward_l)

    actor_l.distribution.enforce_minimum_std((torch.ones(act_dim)*0.2).to(device))

    end = time.time()
    #print(env.get_reward_info())
    mean_file_name = saver.data_dir + "/rewards.txt"
    contact_file_name = saver.data_dir + "/contact_rewards.txt"
    impulse_file_name = saver.data_dir + "/impulse_rewards.txt"
    pos_file_name = saver.data_dir + "/pos_rewards.txt"
    pose_file_name = saver.data_dir + "/pose_rewards.txt"
    falling_file_name = saver.data_dir + "/falling_rewards.txt"
    rel_obj_file_name = saver.data_dir + "/rel_obj_rewards.txt"
    body_vel_file_name = saver.data_dir + "/body_vel_rewards.txt"
    body_qvel_file_name = saver.data_dir + "/body_qvel_rewards.txt"

    np.savetxt(mean_file_name, avg_rewards)
    np.savetxt(contact_file_name, contact_rewards)
    np.savetxt(impulse_file_name, impulse_rewards)
    np.savetxt(pos_file_name, pos_rewards)
    np.savetxt(pose_file_name, pose_rewards)
    np.savetxt(falling_file_name, falling_rewards)
    np.savetxt(rel_obj_file_name, rel_obj_rewards)
    np.savetxt(body_vel_file_name, body_vel_rewards)
    np.savetxt(body_qvel_file_name, body_qvel_rewards)

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps_l / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps_l / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('std: ')
    print(np.exp(actor_l.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')

    if np.isnan(np.isnan(np.exp(actor_l.distribution.std.cpu().detach().numpy())).any()):
        print('resetting env')
        env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
        
