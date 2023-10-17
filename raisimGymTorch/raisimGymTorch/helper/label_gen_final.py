import os
import os.path as op
import math
import numpy as np
import torch
from raisimGymTorch.helper import rotations
import random

from raisimGymTorch.helper.all_frame_dict import test_frame, test_frame_new, test_frame_final, extra_dict, recon_dict


# Function of this script:
# Use heuristics to generate a sequence of frame numbers
# as training/testing set candidate.
# The frame numbers must come along with the motion sequence name, subject number
# and label kind (for 2lift / for left articulate / for right articulate)
# save the files to "grasp_label".
# Also write a function to generate initial frame

task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../.."
hand_path = home_path + "/rsc/mano_double"
obj_path = home_path + "/rsc/arctic"

path_mean_r = os.path.join(home_path + f"/rsc/mano_double/right_pose_mean.txt")
path_mean_l = os.path.join(home_path + f"/rsc/mano_double/left_pose_mean.txt")
pose_mean_r = np.loadtxt(path_mean_r)
pose_mean_l = np.loadtxt(path_mean_l)

eps_angle = 1e-2

adjust_2lift = {"s1_waffleiron_use_01": [0.73, 0.08, 0.601],
				"s1_waffleiron_grab_01": [0.63, 0.05, 0.601],
				"s2_waffleiron_use_01": [0.73, 0.08, 0.601],
				"s2_waffleiron_use_02": [0.63, 0.05, 0.601],
				"s2_waffleiron_grab_01": [0.63, 0.05, 0.601],
				"s9_waffleiron_use_01": [0.69, 0.10, 0.601],
				"s9_waffleiron_use_04": [0.73, 0.08, 0.601],
			   }

class Labels:
	def __init__(self):
		# self.kind = ""
		self.seq_num = 0
		self.seq_name = ""
		self.init_frame_r = []
		self.init_frame_l = []
		self.grasp_frame_r = []
		self.grasp_frame_l = []
		self.kind = []

def generate_recon_test_set_online(obj_name):
	label_path = hand_path + "/grasp_label/final_labels/s5_recon"
	original_seqs = {}
	original_seqs[f"{obj_name}_list"] = np.load(label_path + f"/{obj_name}.npy", allow_pickle=True)


	for labels_item in original_seqs:
		arti_labels_list = []
		grasp_labels_list = []
		obj_name = labels_item.split('_')[0]
		for item in original_seqs[labels_item]:
			grasp_list_len = len(item.grasp_frame_r)
			for frame in range(grasp_list_len):
				label_item = {}
				label_item["seq_num"] = item.seq_num
				label_item["seq_name"] = item.seq_name
				label_item["obj_name"] = item.seq_name.split('_')[0]
				label_item["init_frame"] = item.init_frame_l[frame]
				label_item["grasp_frame"] = item.grasp_frame_r[frame]
				label_item["kind"] = item.kind[frame]
				if label_item["kind"] == "right_arti":
					arti_labels_list.append(label_item)
				elif label_item["kind"] == "right_grasp" or label_item["kind"] == "two_grasp" or label_item[
					"kind"] == "left_grasp":
					grasp_labels_list.append(label_item)

		test_labels = np.concatenate((arti_labels_list, grasp_labels_list), axis=0)


		test_dic = label_path + "/../test_recon"
		if not op.exists(test_dic):
			os.system(f"mkdir -p {test_dic}")
		np.save(f"{test_dic}/{obj_name}.npy", test_labels)

		print("obj name: ", obj_name)
		print("test num: ", len(test_labels))
		print(" ")


def generate_new_train_test_set():
	label_path = hand_path + "/grasp_label/final_labels/final"
	original_seqs = {}
	original_seqs["notebook_list"] = np.load(label_path + f"/notebook.npy", allow_pickle=True)
	original_seqs["box_list"] = np.load(label_path + f"/box.npy", allow_pickle=True)
	original_seqs["laptop_list"] = np.load(label_path + f"/laptop.npy", allow_pickle=True)
	original_seqs["waffleiron_list"] = np.load(label_path + f"/waffleiron.npy", allow_pickle=True)
	original_seqs["mixer_list"] = np.load(label_path + f"/mixer.npy", allow_pickle=True)
	original_seqs["microwave_list"] = np.load(label_path + f"/microwave.npy", allow_pickle=True)
	original_seqs["ketchup_list"] = np.load(label_path + f"/ketchup.npy", allow_pickle=True)
	original_seqs["capsulemachine_list"] = np.load(label_path + f"/capsulemachine.npy", allow_pickle=True)
	original_seqs["espressomachine_list"] = np.load(label_path + f"/espressomachine.npy", allow_pickle=True)
	original_seqs["phone_list"] = np.load(label_path + f"/phone.npy", allow_pickle=True)

	for labels_item in original_seqs:
		arti_labels_list = []
		obj_name = labels_item.split('_')[0]
		for item in original_seqs[labels_item]:
			grasp_list_len = len(item.grasp_frame_r)
			for frame in range(grasp_list_len):
				label_item = {}
				label_item["seq_num"] = item.seq_num
				label_item["seq_name"] = item.seq_name
				label_item["obj_name"] = item.seq_name.split('_')[0]
				label_item["init_frame"] = item.init_frame_l[frame] - 1
				label_item["grasp_frame"] = item.grasp_frame_r[frame] - 1
				label_item["kind"] = item.kind[frame]
				if label_item["kind"] == "right_arti_on_hand":
					label_arti_copy = label_item.copy()
					label_arti_copy["init_frame"] = 20
					label_arti_copy["kind"] = "right_arti"
					arti_labels_list.append(label_arti_copy)

		arti_labels_num = len(arti_labels_list)
		print(obj_name)
		print("new arti_labels_num: ", arti_labels_num)

		test_arti_num = int(arti_labels_num * 0.5 + 0.5)
		arti_labels_list = np.array(arti_labels_list, dtype=object)
		np.random.shuffle(arti_labels_list)
		test_arti_labels = arti_labels_list[:test_arti_num]
		train_arti_labels = arti_labels_list[test_arti_num:]
		print("new test_arti_num: ", len(test_arti_labels))
		print("new train_arti_num: ", len(train_arti_labels))

		ori_train_arti = []
		ori_train_grasp = []
		train_dic = label_path + "/../train_set"
		train_list = np.load(train_dic + f"/{obj_name}.npy", allow_pickle=True)
		for item in train_list:
			if item["kind"] == "right_arti":
				ori_train_arti.append(item)
			else:
				ori_train_grasp.append(item)

		new_train_set = np.concatenate((ori_train_arti, train_arti_labels, ori_train_grasp), axis=0)
		print("train")
		print("arti num: ", len(new_train_set) - len(ori_train_grasp))
		print("grasp num: ", len(ori_train_grasp))

		ori_test_arti = []
		ori_test_grasp = []
		test_dic = label_path + "/../test_set"
		test_list = np.load(test_dic + f"/{obj_name}.npy", allow_pickle=True)
		for item in test_list:
			if item["kind"] == "right_arti":
				ori_test_arti.append(item)
			else:
				ori_test_grasp.append(item)

		new_test_set = np.concatenate((ori_test_arti, test_arti_labels, ori_test_grasp), axis=0)
		print("test")
		print("arti num: ", len(new_test_set) - len(ori_test_grasp))
		print("grasp num: ", len(ori_test_grasp))

		new_train_dic = label_path + "/../new_train_set"
		if not op.exists(new_train_dic):
			os.system(f"mkdir -p {new_train_dic}")
		np.save(f"{new_train_dic}/{obj_name}.npy", new_train_set)

		new_test_dic = label_path + "/../new_test_set"
		if not op.exists(new_test_dic):
			os.system(f"mkdir -p {new_test_dic}")
		np.save(f"{new_test_dic}/{obj_name}.npy", new_test_set)


		recon_arti = []
		recon_grasp = []
		test_recon_dic = label_path + "/../test_recon"
		test_recon_list = np.load(test_recon_dic + f"/{obj_name}.npy", allow_pickle=True)
		for item in test_recon_list:
			if item["kind"] == "right_arti":
				recon_arti.append(item)
			else:
				recon_grasp.append(item)
		new_test_set_with_s5 = np.concatenate((ori_test_arti, recon_arti, test_arti_labels, ori_test_grasp, recon_grasp), axis=0)
		print("test with s5")
		print("arti num: ", len(new_test_set_with_s5) - len(ori_test_grasp) - len(recon_grasp))
		print("grasp num: ", len(ori_test_grasp) + len(recon_grasp))
		print(" ")
		new_test_dic_with_s5 = label_path + "/../new_test_set_with_s5"
		if not op.exists(new_test_dic_with_s5):
			os.system(f"mkdir -p {new_test_dic_with_s5}")
		np.save(f"{new_test_dic_with_s5}/{obj_name}.npy", new_test_set_with_s5)


def get_special_labels(): # special means left hand grasp while right hand articulate
	label_path = hand_path + "/grasp_label/final_labels/final"
	original_seqs = {}
	original_seqs["mixer_list"] = np.load(label_path + f"/mixer.npy", allow_pickle=True)
	original_seqs["ketchup_list"] = np.load(label_path + f"/ketchup.npy", allow_pickle=True)
	original_seqs["phone_list"] = np.load(label_path + f"/phone.npy", allow_pickle=True)

	for labels_item in original_seqs:
		special_labels_list = []
		obj_name = labels_item.split('_')[0]
		for item in original_seqs[labels_item]:
			grasp_list_len = len(item.grasp_frame_r)
			for frame in range(grasp_list_len):
				label_item = {}
				label_item["seq_num"] = item.seq_num
				label_item["seq_name"] = item.seq_name
				label_item["obj_name"] = item.seq_name.split('_')[0]
				label_item["init_frame"] = item.init_frame_l[frame] - 1
				label_item["grasp_frame"] = item.grasp_frame_r[frame] - 1
				label_item["kind"] = item.kind[frame]
				if label_item["kind"] == "right_special":
					special_labels_list.append(label_item)

		special_dict = label_path + "/../special_label"
		if not op.exists(special_dict):
			os.system(f"mkdir -p {special_dict}")
		np.save(f"{special_dict}/{obj_name}.npy", special_labels_list)
		print("obj name: ", obj_name)
		print("special num: ", len(special_labels_list))

# final evaluation on floating object
def label_eval_floating(obj_name, test):
	if test:
		label_path = hand_path + "/grasp_label/final_labels/test_set"
	else:
		label_path = hand_path + "/grasp_label/final_labels/train_set"

	original_seqs = np.load(label_path + f"/{obj_name}.npy", allow_pickle=True)

	arti_labels_list = []
	grasp_labels_list = []

	for label_item in original_seqs:
		if label_item["kind"] == "right_arti":
			if label_item["obj_name"] != "mixer" and label_item["obj_name"] != "ketchup" and label_item[
				"obj_name"] != "phone":
				arti_labels_list.append(label_item)
		elif label_item["kind"] == "right_grasp" or label_item["kind"] == "two_grasp" or label_item[
			"kind"] == "left_grasp":
			grasp_labels_list.append(label_item)

	print(obj_name)
	print(len(arti_labels_list))
	print(len(grasp_labels_list))
	print(" ")

	return arti_labels_list, grasp_labels_list

# final evaluation on composed task
def label_eval_compose(obj_name, test):
	if test:
		label_path = hand_path + "/grasp_label/final_labels/test_set"
	else:
		label_path = hand_path + "/grasp_label/final_labels/train_set"

	original_seqs = np.load(label_path + f"/{obj_name}.npy", allow_pickle=True)

	arti_labels_list = []
	grasp_labels_list = []

	if obj_name != "mixer" and obj_name != "ketchup" and obj_name != "phone":
		for label_item in original_seqs:
			if label_item["kind"] == "right_arti":
				arti_labels_list.append(label_item)
			elif label_item["kind"] == "right_grasp" or label_item["kind"] == "two_grasp" or label_item[
				"kind"] == "left_grasp":
				grasp_labels_list.append(label_item)
	else:
		for label_item in original_seqs:
			if label_item["kind"] == "right_arti":
				arti_labels_list.append(label_item)
			elif label_item["kind"] == "left_grasp":
				grasp_labels_list.append(label_item)

	print(obj_name)
	print(len(arti_labels_list))
	print(len(grasp_labels_list))
	print(" ")

	return arti_labels_list, grasp_labels_list

# final evaluation on fixed object
def label_eval_fixed(obj_name, test):
	if test:
		label_path = hand_path + "/grasp_label/final_labels/test_set"
	else:
		label_path = hand_path + "/grasp_label/final_labels/train_set"

	original_seqs = np.load(label_path + f"/{obj_name}.npy", allow_pickle=True)

	arti_labels_list = []
	grasp_labels_list = []

	for label_item in original_seqs:
		if label_item["kind"] == "right_arti":
			arti_labels_list.append(label_item)

	print(obj_name)
	print(len(arti_labels_list))
	print(len(grasp_labels_list))
	print(" ")

	return arti_labels_list


def label_train_r(num_repeats, eval, test, obj_cmd=None):
	obj_lists = ["notebook", "box", "laptop", "waffleiron", "mixer", "ketchup", "phone", "capsulemachine", "espressomachine", "microwave"]
	shuffle_label = []
	if test:
		label_path = hand_path + "/grasp_label/final_labels/test_set"
	else:
		label_path = hand_path + "/grasp_label/final_labels/train_set"
	original_seqs = {}
	for obj in obj_lists:
		original_seqs[f"{obj}_list"] = np.load(label_path + f"/{obj}.npy", allow_pickle=True)

	original_labels_arti = {}
	original_labels_grasp = {}

	for labels_item in original_seqs:
		arti_labels_list = []
		grasp_labels_list = []
		obj_name = labels_item.split('_')[0]
		for label_item in original_seqs[labels_item]:
			if label_item["kind"] == "right_arti":
				arti_labels_list.append(label_item)
			elif label_item["kind"] == "right_grasp" or label_item["kind"] == "two_grasp":
				grasp_labels_list.append(label_item)
		original_labels_arti[f"{obj_name}_list"] = arti_labels_list
		original_labels_grasp[f"{obj_name}_list"] = grasp_labels_list

		print(obj_name)
		print("arti label num: ", len(arti_labels_list))
		print("grasp label num: ", len(grasp_labels_list))
		print(" ")

	if eval:
		if obj_cmd is not None:
			obj_name = obj_cmd
		else:
			rand_flag = random.random()
			# rand_flag = 0.0
			if rand_flag < 0.1:
				obj_name = "box"
			elif rand_flag < 0.2:
				obj_name = "laptop"
			elif rand_flag < 0.3:
				obj_name = "notebook"
			elif rand_flag < 0.4:
				obj_name = "waffleiron"
			elif rand_flag < 0.5:
				obj_name = "mixer"
			elif rand_flag < 0.6:
				obj_name = "ketchup"
			elif rand_flag < 0.7:
				obj_name = "microwave"
			elif rand_flag < 0.8:
				obj_name = "capsulemachine"
			elif rand_flag < 0.9:
				obj_name = "phone"
			else:
				obj_name = "espressomachine"

		kind_flag = random.random()
		if kind_flag < 0.5:
			labels_len = len(original_labels_arti[f"{obj_name}_list"])
			temp_label = original_labels_arti[f"{obj_name}_list"][random.randint(0, labels_len - 1)]
		else:
			labels_len = len(original_labels_grasp[f"{obj_name}_list"])
			temp_label = original_labels_grasp[f"{obj_name}_list"][random.randint(0, labels_len - 1)]
		shuffle_label.append(temp_label)
	else:
		per_obj_cat = int(num_repeats * 0.05)
		print("per_obj_cat: ", per_obj_cat)
		for obj in obj_lists:
			grasp_len = len(original_labels_grasp[f"{obj}_list"])
			arti_len = len(original_labels_arti[f"{obj}_list"])
			grasp_itr_num = int(per_obj_cat / grasp_len)
			arti_itr_num = int(per_obj_cat / arti_len)
			for i in range(grasp_itr_num):
				shuffle_label.extend(original_labels_grasp[f"{obj}_list"])
			shuffled_grasp = original_labels_grasp[f"{obj}_list"].copy()
			random.shuffle(shuffled_grasp)
			shuffle_label.extend(shuffled_grasp[:per_obj_cat - grasp_itr_num * grasp_len])
			for i in range(arti_itr_num):
				shuffle_label.extend(original_labels_arti[f"{obj}_list"])
			shuffled_arti = original_labels_arti[f"{obj}_list"].copy()
			random.shuffle(shuffled_arti)
			shuffle_label.extend(shuffled_arti[:per_obj_cat - arti_itr_num * arti_len])

	random.shuffle(shuffle_label)

	return original_labels_arti, original_labels_grasp, shuffle_label

def label_train_l(num_repeats, eval, test, obj_cmd=None):
	obj_lists = ["notebook", "box", "laptop", "waffleiron", "mixer", "ketchup", "phone", "capsulemachine",
				 "espressomachine", "microwave"]
	shuffle_label = []
	if test:
		label_path = hand_path + "/grasp_label/final_labels/test_set"
	else:
		label_path = hand_path + "/grasp_label/final_labels/train_set"
	original_seqs = {}
	for obj in obj_lists:
		original_seqs[f"{obj}_list"] = np.load(label_path + f"/{obj}.npy", allow_pickle=True)

	original_labels_arti = {}
	original_labels_grasp = {}

	for labels_item in original_seqs:
		grasp_labels_list = []
		obj_name = labels_item.split('_')[0]
		for label_item in original_seqs[labels_item]:
			if label_item["kind"] == "left_grasp" or label_item["kind"] == "two_grasp":
				grasp_labels_list.append(label_item)
		original_labels_grasp[f"{obj_name}_list"] = grasp_labels_list

		print(obj_name)
		print("grasp label num: ", len(grasp_labels_list))
		print(" ")

	if eval:
		if obj_cmd is not None:
			obj_name = obj_cmd
		else:
			rand_flag = random.random()
			rand_flag = 0.55
			if rand_flag < 0.1:
				obj_name = "box"
			elif rand_flag < 0.2:
				obj_name = "laptop"
			elif rand_flag < 0.3:
				obj_name = "notebook"
			elif rand_flag < 0.4:
				obj_name = "waffleiron"
			elif rand_flag < 0.5:
				obj_name = "mixer"
			elif rand_flag < 0.6:
				obj_name = "ketchup"
			elif rand_flag < 0.7:
				obj_name = "microwave"
			elif rand_flag < 0.8:
				obj_name = "capsulemachine"
			elif rand_flag < 0.9:
				obj_name = "phone"
			else:
				obj_name = "espressomachine"

		labels_len = len(original_labels_grasp[f"{obj_name}_list"])
		temp_label = original_labels_grasp[f"{obj_name}_list"][random.randint(0, labels_len - 1)]
		shuffle_label.append(temp_label)
	else:
		per_obj_cat = int(num_repeats * 0.1)
		print("per_obj_cat: ", per_obj_cat)
		for obj in obj_lists:
			grasp_len = len(original_labels_grasp[f"{obj}_list"])
			grasp_itr_num = int(per_obj_cat / grasp_len)
			for i in range(grasp_itr_num):
				shuffle_label.extend(original_labels_grasp[f"{obj}_list"])
			shuffled_grasp = original_labels_grasp[f"{obj}_list"].copy()
			random.shuffle(shuffled_grasp)
			shuffle_label.extend(shuffled_grasp[:per_obj_cat - grasp_itr_num * grasp_len])

	random.shuffle(shuffle_label)

	return original_labels_arti, original_labels_grasp, shuffle_label


def label_train(num_repeats, eval, test):
	obj_lists = ["notebook", "box", "laptop", "waffleiron", "mixer", "ketchup", "phone", "capsulemachine",
				 "espressomachine", "microwave"]
	shuffle_label = []
	if test:
		label_path = hand_path + "/grasp_label/final_labels/test_set"
	else:
		label_path = hand_path + "/grasp_label/final_labels/train_set"
	original_seqs = {}
	for obj in obj_lists:
		original_seqs[f"{obj}_list"] = np.load(label_path + f"/{obj}.npy", allow_pickle=True)

	original_labels_arti = {}
	original_labels_grasp = {}

	for labels_item in original_seqs:
		arti_labels_list = []
		grasp_labels_list = []
		obj_name = labels_item.split('_')[0]
		for label_item in original_seqs[labels_item]:
			if label_item["kind"] == "right_arti":
				arti_labels_list.append(label_item)
			elif label_item["kind"] == "right_grasp" or label_item["kind"] == "two_grasp" or label_item["kind"] == "left_grasp":
				grasp_labels_list.append(label_item)
		original_labels_arti[f"{obj_name}_list"] = arti_labels_list
		original_labels_grasp[f"{obj_name}_list"] = grasp_labels_list

		print(obj_name)
		print("arti label num: ", len(arti_labels_list))
		print("grasp label num: ", len(grasp_labels_list))
		print(" ")

	if eval:
		rand_flag = random.random()
		if rand_flag < 0.1:
			obj_name = "box"
		elif rand_flag < 0.2:
			obj_name = "laptop"
		elif rand_flag < 0.3:
			obj_name = "notebook"
		elif rand_flag < 0.4:
			obj_name = "waffleiron"
		elif rand_flag < 0.5:
			obj_name = "mixer"
		elif rand_flag < 0.6:
			obj_name = "ketchup"
		elif rand_flag < 0.7:
			obj_name = "microwave"
		elif rand_flag < 0.8:
			obj_name = "capsulemachine"
		elif rand_flag < 0.9:
			obj_name = "phone"
		else:
			obj_name = "espressomachine"

		kind_flag = random.random()
		if kind_flag < 0.5:
			labels_len = len(original_labels_arti[f"{obj_name}_list"])
			temp_label = original_labels_arti[f"{obj_name}_list"][random.randint(0, labels_len - 1)]
		else:
			labels_len = len(original_labels_grasp[f"{obj_name}_list"])
			temp_label = original_labels_grasp[f"{obj_name}_list"][random.randint(0, labels_len - 1)]
		shuffle_label.append(temp_label)
	else:
		per_obj_cat = int(num_repeats * 0.05)
		print("per_obj_cat: ", per_obj_cat)
		for obj in obj_lists:
			grasp_len = len(original_labels_grasp[f"{obj}_list"])
			arti_len = len(original_labels_arti[f"{obj}_list"])
			grasp_itr_num = int(per_obj_cat / grasp_len)
			arti_itr_num = int(per_obj_cat / arti_len)
			for i in range(grasp_itr_num):
				shuffle_label.extend(original_labels_grasp[f"{obj}_list"])
			shuffled_grasp = original_labels_grasp[f"{obj}_list"].copy()
			random.shuffle(shuffled_grasp)
			shuffle_label.extend(shuffled_grasp[:per_obj_cat - grasp_itr_num * grasp_len])
			for i in range(arti_itr_num):
				shuffle_label.extend(original_labels_arti[f"{obj}_list"])
			shuffled_arti = original_labels_arti[f"{obj}_list"].copy()
			random.shuffle(shuffled_arti)
			shuffle_label.extend(shuffled_arti[:per_obj_cat - arti_itr_num * arti_len])


	random.shuffle(shuffle_label)

	return original_labels_arti, original_labels_grasp, shuffle_label

def pose_gen(labels, num_repeats, use_init_frame=False, recon_flag = False, pre_contact = False):
	if recon_flag:
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!This is recon result!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	# initialize arrays
	stage_dim = np.zeros((num_repeats, 3))
	stage_pos = np.zeros((num_repeats, 3))
	obj_pose_reset = np.zeros((num_repeats, 8))
	qpos_reset_r = np.zeros((num_repeats, 51))
	qpos_reset_l = np.zeros((num_repeats, 51))
	final_obj_angle = np.zeros((num_repeats, 1))
	final_obj_pos = np.zeros((num_repeats, 8))
	final_obj_pos_r = np.zeros((num_repeats, 8))
	final_ee_r = np.zeros((num_repeats, 63))
	final_ee_l = np.zeros((num_repeats, 63))
	final_pose_r = np.zeros((num_repeats, 48))
	final_pose_l = np.zeros((num_repeats, 48))
	final_qpos_r = np.zeros((num_repeats, 51))
	final_qpos_l = np.zeros((num_repeats, 51))
	final_contacts_r = np.zeros((num_repeats, 16))
	final_contacts_l = np.zeros((num_repeats, 16))
	final_obj_euler = np.zeros((num_repeats, 3))

	obj_name_list = []
	left_kind_list = []
	right_kind_list = []

	for i in range(num_repeats):
		# load raw data, for articulate, the two hands are trained individually, for one training, these two can be the same
		seq_num = labels[i]["seq_num"]
		seq_name = labels[i]["seq_name"]
		obj_name = labels[i]["seq_name"].split('_')[0]
		init_frame = labels[i]["init_frame"]
		grasp_frame = labels[i]["grasp_frame"]
		kind = labels[i]["kind"]

		obj_name_list.append(obj_name)

		if i == 0:
			print("seq_num: ", seq_num)
			print("seq_name: ", seq_name)
			print("init_frame: ", init_frame)
			print("grasp_frame: ", grasp_frame)
			print("kind: ", kind)

		stage_dim[i,0] = 1.0
		stage_dim[i,1] = 1.0
		stage_dim[i,2] = 0.1
		stage_pos[i,0] = 0.75
		stage_pos[i,1] = 0.0
		stage_pos[i,2] = 0.551

		if kind == "two_arti":
			left_kind = "arti"
			right_kind = "arti"
			left_idx = 8
			right_idx = 8
		elif kind == "right_arti":
			left_kind = "no"
			right_kind = "arti"
			left_idx = 7
			right_idx = 8
		elif kind == "left_arti":
			left_kind = "arti"
			right_kind = "no"
			left_idx = 8
			right_idx = 7
		elif kind == "two_grasp":
			left_kind = "grasp"
			right_kind = "grasp"
			left_idx = 9
			right_idx = 9
		elif kind == "right_grasp":
			left_kind = "no"
			right_kind = "grasp"
			left_idx = 7
			right_idx = 9
		elif kind == "left_grasp":
			left_kind = "grasp"
			right_kind = "no"
			left_idx = 9
			right_idx = 7
		elif kind == "left_special":
			left_kind = "arti"
			right_kind = "grasp"
			left_idx = 8
			right_idx = 9
		elif kind == "right_special":
			left_kind = "grasp"
			right_kind = "arti"
			left_idx = 9
			right_idx = 8
		else:
			left_kind = ""
			right_kind = ""
			left_idx = 7
			right_idx = 7

		left_kind_list.append(left_idx)
		right_kind_list.append(right_idx)

		if recon_flag:
			mano_params = np.load(
				home_path + f"/rsc/mano_double/s{seq_num}_recon/{seq_name}.mano.npy", allow_pickle=True
			).item()
			obj_params = np.load(
				home_path + f"/rsc/arctic/{obj_name}/s{seq_num}_recon/{seq_name}.object.npy", allow_pickle=True
			)
			obj_params_gt = np.load(
				home_path + f"/rsc/arctic/{obj_name}/s{seq_num}/{seq_name}.object.npy", allow_pickle=True
			)
			final_ee_r[i, :] = np.load(hand_path + f"/ee_recon/s{seq_num}/{seq_name}_r.npy")[grasp_frame, :, :].reshape(1, -1)
			final_ee_l[i, :] = np.load(hand_path + f"/ee_recon/s{seq_num}/{seq_name}_l.npy")[grasp_frame, :, :].reshape(1, -1)
			height_ref = np.load(hand_path + f"/ee_recon/s{seq_num}/{seq_name}_min.npy")[init_frame, :][2]
			height_ref_gt = np.load(hand_path + f"/ee/s{seq_num}/{seq_name}_min.npy")[init_frame, :][2]
		else:
			mano_params = np.load(
				home_path + f"/rsc/mano_double/s{seq_num}/{seq_name}.mano.npy", allow_pickle=True
			).item()
			obj_params = np.load(
				home_path + f"/rsc/arctic/{obj_name}/s{seq_num}/{seq_name}.object.npy", allow_pickle=True
			)
			final_ee_r[i, :] = np.load(hand_path + f"/ee/s{seq_num}/{seq_name}_r.npy")[grasp_frame, :, :].reshape(1, -1)
			final_ee_l[i, :] = np.load(hand_path + f"/ee/s{seq_num}/{seq_name}_l.npy")[grasp_frame, :, :].reshape(1, -1)
			height_ref = np.load(hand_path + f"/ee/s{seq_num}/{seq_name}_min.npy")[init_frame, :][2]
		ref_r = [0.09566994, 0.00638343, 0.0061863]
		ref_l = [-0.09566994, 0.00638343, 0.0061863]

		# final mano
		mano_rot_final_r = mano_params["right"]["rot"][grasp_frame]  # 3
		mano_pose_final_r = mano_params["right"]["pose"][grasp_frame] + pose_mean_r  # 45
		mano_trans_final_r = mano_params["right"]["trans"][grasp_frame] + ref_r  # 3
		mano_rot_final_l = mano_params["left"]["rot"][grasp_frame]  # 3
		mano_pose_final_l = mano_params["left"]["pose"][grasp_frame] + pose_mean_l  # 45
		mano_trans_final_l = mano_params["left"]["trans"][grasp_frame] + ref_l  # 3
		mano_rot_final_r = rotations.axisangle2euler(mano_rot_final_r.reshape(-1, 3)).reshape(-1)
		mano_pose_final_r = rotations.axisangle2euler(mano_pose_final_r.reshape(-1, 3)).reshape(-1)
		mano_rot_final_l = rotations.axisangle2euler(mano_rot_final_l.reshape(-1, 3)).reshape(-1)
		mano_pose_final_l = rotations.axisangle2euler(mano_pose_final_l.reshape(-1, 3)).reshape(-1)

		# final obj
		if left_kind == "arti" or right_kind == "arti":
			gt_obj_angles_final = np.array([obj_params[:, 0][grasp_frame]])  # radian 1
			gt_obj_angles_final_r = np.array([obj_params[:, 0][grasp_frame]])  # radian 1
		else:
			if obj_name == "ketchup" or obj_name == "phone":
				gt_obj_angles_final = np.array([obj_params[:, 0][init_frame]])
				gt_obj_angles_final_r = np.array([obj_params[:, 0][init_frame]])
				if obj_params[:, 0][init_frame] > 2.48:
					gt_obj_angles_final[:] = 2.48
					gt_obj_angles_final_r[:] = 2.48
			else:
				gt_obj_angles_final = np.array([0])
				gt_obj_angles_final_r = np.array([0])
		gt_axis_angle_final = obj_params[:, 1:4][grasp_frame]  # 3
		gt_axis_angle_final_r = obj_params[:, 1:4][grasp_frame]  # 3
		gt_transl_final = obj_params[:, 4:][grasp_frame] / 1000  # 3
		gt_transl_final_r = obj_params[:, 4:][grasp_frame] / 1000  # 3
		gt_quat_final = rotations.axisangle2quat(gt_axis_angle_final)
		gt_euler_final = rotations.axisangle2euler(gt_axis_angle_final.reshape(1, -1)).reshape(-1)  # 3
		gt_quat_final_r = rotations.axisangle2quat(gt_axis_angle_final_r)
		final_obj_angle[i, :] = gt_obj_angles_final

		# init obj
		if recon_flag:
			if obj_name == "ketchup" or obj_name == "phone":
				gt_obj_angles_init = np.array([obj_params_gt[:, 0][init_frame]])
				if obj_params_gt[:, 0][init_frame] > 2.48:
					gt_obj_angles_init[:] = 2.48
			else:
				gt_obj_angles_init = np.array([0])  # radian 1
			gt_axis_angle_init = obj_params_gt[:, 1:4][init_frame]  # 3
			gt_transl_init = obj_params_gt[:, 4:][init_frame] / 1000  # 3
			gt_quat_init = rotations.axisangle2quat(gt_axis_angle_init)
		else:
			if obj_name == "ketchup" or obj_name == "phone":
				gt_obj_angles_init = np.array([obj_params[:, 0][init_frame]])
				if obj_params[:, 0][init_frame] > 2.48:
					gt_obj_angles_init[:] = 2.48
			else:
				gt_obj_angles_init = np.array([0])  # radian 1
			gt_axis_angle_init = obj_params[:, 1:4][init_frame]  # 3
			gt_transl_init = obj_params[:, 4:][init_frame] / 1000  # 3
			gt_quat_init = rotations.axisangle2quat(gt_axis_angle_init)

		# init mano
		if use_init_frame:
			mano_rot_init_r = mano_params["right"]["rot"][init_frame]  # 3
			mano_pose_init_r = mano_params["right"]["pose"][init_frame] + pose_mean_r  # 3
			mano_trans_init_r = mano_params["right"]["trans"][init_frame] + ref_r  # 45
			mano_rot_init_l = mano_params["left"]["rot"][init_frame]  # 3
			mano_pose_init_l = mano_params["left"]["pose"][init_frame] + pose_mean_l  # 3
			mano_trans_init_l = mano_params["left"]["trans"][init_frame] + ref_l  # 45

			mano_rot_init_r = rotations.axisangle2euler(mano_rot_init_r.reshape(-1, 3)).reshape(-1)
			mano_pose_init_r = rotations.axisangle2euler(mano_pose_init_r.reshape(-1, 3)).reshape(-1)
			mano_rot_init_l = rotations.axisangle2euler(mano_rot_init_l.reshape(-1, 3)).reshape(-1)
			mano_pose_init_l = rotations.axisangle2euler(mano_pose_init_l.reshape(-1, 3)).reshape(-1)
		else:
			gt_matrix_final = rotations.quat2mat(gt_quat_final)
			gt_matrix_init = rotations.quat2mat(gt_quat_init)

			rel_angle = gt_obj_angles_final - gt_obj_angles_init
			gt_rel_rot = rotations.euler2mat([0, 0, rel_angle])

			mano_pose_init_r = mano_pose_final_r.copy() * 0.3
			mano_pose_init_l = mano_pose_final_l.copy() * 0.3
			mano_rel_trans_r = mano_trans_final_r - gt_transl_final
			mano_rel_trans_l = mano_trans_final_l - gt_transl_final
			mano_rel_trans_r_unit = mano_rel_trans_r / np.linalg.norm(mano_rel_trans_r)
			mano_rel_trans_l_unit = mano_rel_trans_l / np.linalg.norm(mano_rel_trans_l)

			if (obj_name == "ketchup" or obj_name == "capsulemachine" or obj_name == "mixer") and right_kind == "grasp" or left_kind == "grasp":
				mano_rel_trans_r_unit[2] = 0.
				mano_rel_trans_l_unit[2] = 0.
			else:
				mano_rel_trans_r_unit[2] *= 0.5
				mano_rel_trans_l_unit[2] *= 0.5

			if right_kind == "no":
				mano_rel_trans_r += 0.3 * mano_rel_trans_r_unit
			else:
				if obj_name == "ketchup":
					mano_rel_trans_r += 0.15 * mano_rel_trans_r_unit
				else:
					mano_rel_trans_r += 0.1 * mano_rel_trans_r_unit
			if left_kind == "no":
				mano_rel_trans_l += 0.3 * mano_rel_trans_l_unit
			else:
				if obj_name == "ketchup":
					mano_rel_trans_l += 0.15 * mano_rel_trans_l_unit
				else:
					mano_rel_trans_l += 0.1 * mano_rel_trans_l_unit

			mano_rel_trans_r = np.matmul(gt_matrix_final.T, mano_rel_trans_r)
			mano_rel_trans_l = np.matmul(gt_matrix_final.T, mano_rel_trans_l)

			if right_kind == "arti":
				mano_rel_trans_r = np.matmul(gt_rel_rot, mano_rel_trans_r)
			if left_kind == "arti":
				mano_rel_trans_l = np.matmul(gt_rel_rot, mano_rel_trans_l)

			mano_rel_trans_r = np.matmul(gt_matrix_init, mano_rel_trans_r)
			mano_rel_trans_l = np.matmul(gt_matrix_init, mano_rel_trans_l)
			mano_trans_init_r = gt_transl_init + mano_rel_trans_r
			mano_trans_init_l = gt_transl_init + mano_rel_trans_l

			mano_rot_matrix_final_r = rotations.euler2mat(mano_rot_final_r)
			mano_rot_matrix_final_l = rotations.euler2mat(mano_rot_final_l)

			mano_rot_matrix_temp_r = np.matmul(gt_matrix_final.T, mano_rot_matrix_final_r)
			mano_rot_matrix_temp_l = np.matmul(gt_matrix_final.T, mano_rot_matrix_final_l)

			if right_kind == "arti":
				mano_rot_matrix_temp_r = np.matmul(gt_rel_rot, mano_rot_matrix_temp_r)
			if left_kind == "arti":
				mano_rot_matrix_temp_l = np.matmul(gt_rel_rot, mano_rot_matrix_temp_l)

			mano_rot_matrix_init_r = np.matmul(gt_matrix_init, mano_rot_matrix_temp_r)
			mano_rot_matrix_init_l = np.matmul(gt_matrix_init, mano_rot_matrix_temp_l)

			mano_rot_init_r = rotations.mat2euler(mano_rot_matrix_init_r)
			mano_rot_init_l = rotations.mat2euler(mano_rot_matrix_init_l)

		if right_kind == "grasp":
			if recon_flag:
				final_contacts_r[i, :] = np.logical_or(
					np.load(hand_path + f"/contact_recon/s{seq_num}/{seq_name}_a_r.npy")[grasp_frame, 0, :],
					np.load(hand_path + f"/contact_recon/s{seq_num}/{seq_name}_a_r.npy")[grasp_frame, 1, :])
			elif pre_contact:
				final_contacts_r[i, :] = np.logical_or(
					np.load(hand_path + f"/contact_pre/s{seq_num}/{seq_name}_a_r.npy")[grasp_frame, 0, :],
					np.load(hand_path + f"/contact_pre/s{seq_num}/{seq_name}_a_r.npy")[grasp_frame, 1, :])
			else:
				final_contacts_r[i, :] = np.logical_or(
					np.load(hand_path + f"/contact/s{seq_num}/{seq_name}_a_r.npy")[grasp_frame, 0, :],
					np.load(hand_path + f"/contact/s{seq_num}/{seq_name}_a_r.npy")[grasp_frame, 1, :])
		elif right_kind == "arti":
			if recon_flag:
				final_contacts_r[i, :] = np.load(hand_path + f"/contact_recon/s{seq_num}/{seq_name}_a_r.npy")[grasp_frame, 0, :]
			elif pre_contact:
				final_contacts_r[i, :] = np.load(hand_path + f"/contact_pre/s{seq_num}/{seq_name}_a_r.npy")[grasp_frame, 0, :]
			else:
				final_contacts_r[i, :] = np.load(hand_path + f"/contact/s{seq_num}/{seq_name}_a_r.npy")[grasp_frame, 0, :]

		elif right_kind == "no":
			final_contacts_r[i, :] = 0
			mano_pose_final_r = mano_pose_init_r.copy()
			mano_trans_final_r = mano_trans_init_r.copy()
			mano_rot_final_r = mano_rot_init_r.copy()

		if left_kind == "grasp":
			if recon_flag:
				final_contacts_l[i, :] = np.logical_or(
					np.load(hand_path + f"/contact_recon/s{seq_num}/{seq_name}_a_l.npy")[grasp_frame, 0, :],
					np.load(hand_path + f"/contact_recon/s{seq_num}/{seq_name}_a_l.npy")[grasp_frame, 1, :])
			elif pre_contact:
				final_contacts_l[i, :] = np.logical_or(
					np.load(hand_path + f"/contact_pre/s{seq_num}/{seq_name}_a_l.npy")[grasp_frame, 0, :],
					np.load(hand_path + f"/contact_pre/s{seq_num}/{seq_name}_a_l.npy")[grasp_frame, 1, :])
			else:
				final_contacts_l[i, :] = np.logical_or(
					np.load(hand_path + f"/contact/s{seq_num}/{seq_name}_a_l.npy")[grasp_frame, 0, :],
					np.load(hand_path + f"/contact/s{seq_num}/{seq_name}_a_l.npy")[grasp_frame, 1, :])
			# final_contacts_l[i, :] = np.load(hand_path + f"/contact/s{seq_num}/{seq_name}_a_l.npy")[grasp_frame, 2, :]
		elif left_kind == "arti":
			if recon_flag:
				final_contacts_l[i, :] = np.load(hand_path + f"/contact_recon/s{seq_num}/{seq_name}_a_l.npy")[grasp_frame, 0, :]
			elif pre_contact:
				final_contacts_l[i, :] = np.load(hand_path + f"/contact_pre/s{seq_num}/{seq_name}_a_l.npy")[grasp_frame, 0, :]
			else:
				final_contacts_l[i, :] = np.load(hand_path + f"/contact/s{seq_num}/{seq_name}_a_l.npy")[grasp_frame, 0, :]
		elif left_kind == "no":
			mano_pose_final_l = mano_pose_init_l.copy()
			mano_trans_final_l = mano_trans_init_l.copy()
			mano_rot_final_l = mano_rot_init_l.copy()
			final_contacts_l[i, :] = 0

		if obj_name == "phone":
			height_desk = 0.651
		else:
			height_desk = 0.601
		xpos_desk = 0.7
		ypos_desk = 0.05

		mano_trans_final_r[2] += - height_ref + height_desk
		mano_trans_final_r[1] += ypos_desk
		mano_trans_final_r[0] += xpos_desk
		mano_trans_final_l[2] += - height_ref + height_desk
		mano_trans_final_l[1] += ypos_desk
		mano_trans_final_l[0] += xpos_desk

		gt_transl_final[2] += - height_ref + height_desk
		gt_transl_final[1] += ypos_desk
		gt_transl_final[0] += xpos_desk

		for j in range(21):
			final_ee_l[i, j * 3 + 2] += - height_ref + height_desk
			final_ee_l[i, j * 3 + 1] += ypos_desk
			final_ee_l[i, j * 3 + 0] += xpos_desk
			final_ee_r[i, j * 3 + 2] += - height_ref + height_desk
			final_ee_r[i, j * 3 + 1] += ypos_desk
			final_ee_r[i, j * 3 + 0] += xpos_desk

		if recon_flag:
			mano_trans_init_r[2] += - height_ref_gt + height_desk
			mano_trans_init_r[1] += ypos_desk
			mano_trans_init_r[0] += xpos_desk
			mano_trans_init_l[2] += - height_ref_gt + height_desk
			mano_trans_init_l[1] += ypos_desk
			mano_trans_init_l[0] += xpos_desk
			gt_transl_init[2] += - height_ref_gt + height_desk
			gt_transl_init[1] += ypos_desk
			gt_transl_init[0] += xpos_desk
		else:
			mano_trans_init_r[2] += - height_ref + height_desk
			mano_trans_init_r[1] += ypos_desk
			mano_trans_init_r[0] += xpos_desk
			mano_trans_init_l[2] += - height_ref + height_desk
			mano_trans_init_l[1] += ypos_desk
			mano_trans_init_l[0] += xpos_desk
			gt_transl_init[2] += - height_ref + height_desk
			gt_transl_init[1] += ypos_desk
			gt_transl_init[0] += xpos_desk

		final_qpos_r[i, :] = np.concatenate((mano_trans_final_r, mano_rot_final_r, mano_pose_final_r))
		final_qpos_l[i, :] = np.concatenate((mano_trans_final_l, mano_rot_final_l, mano_pose_final_l))

		final_obj_pos[i, :] = np.concatenate((gt_transl_final, gt_quat_final, gt_obj_angles_final))
		final_obj_pos_r[i, :] = np.concatenate((gt_transl_final_r, gt_quat_final_r, gt_obj_angles_final_r))
		final_pose_l[i, :] = np.concatenate((mano_rot_final_l, mano_pose_final_l))
		final_pose_r[i, :] = np.concatenate((mano_rot_final_r, mano_pose_final_r))

		obj_pose_reset[i, :] = np.concatenate((gt_transl_init, gt_quat_init, gt_obj_angles_init))
		qpos_reset_r[i, :] = np.concatenate((mano_trans_init_r, mano_rot_init_r, mano_pose_init_r))
		qpos_reset_l[i, :] = np.concatenate((mano_trans_init_l, mano_rot_init_l, mano_pose_init_l))

		final_obj_euler[i, :] = gt_euler_final

	stage_dim = np.float32(stage_dim)
	stage_pos = np.float32(stage_pos)
	obj_pose_reset = np.float32(obj_pose_reset)
	qpos_reset_r = np.float32(qpos_reset_r)
	qpos_reset_l = np.float32(qpos_reset_l)
	final_obj_angle = final_obj_angle.astype("float32")
	final_obj_pos = np.float32(final_obj_pos)
	final_obj_pos_r = np.float32(final_obj_pos_r)
	final_ee_r = np.float32(final_ee_r)
	final_ee_l = np.float32(final_ee_l)
	final_pose_r = np.float32(final_pose_r)
	final_pose_l = np.float32(final_pose_l)
	final_qpos_r = np.float32(final_qpos_r)
	final_qpos_l = np.float32(final_qpos_l)
	final_contacts_r = np.float32(final_contacts_r)
	final_contacts_l = np.float32(final_contacts_l)
	final_obj_euler = np.float32(final_obj_euler)

	return (stage_dim,
			stage_pos,
			obj_pose_reset,
			qpos_reset_r,
			qpos_reset_l,
			final_obj_angle,
			final_obj_pos,
			final_obj_pos_r,
			final_ee_r,
			final_ee_l,
			final_pose_r,
			final_pose_l,
			final_qpos_r,
			final_qpos_l,
			final_contacts_r,
			final_contacts_l,
			final_obj_euler), obj_name_list, left_kind_list, right_kind_list

def label_gen_final(seq_num, seq_name):
	labels = Labels()
	labels.seq_num = seq_num
	labels.seq_name = seq_name

	labels.init_frame_l = test_frame_final[f"seq{seq_num}/{seq_name}/init_frame"]
	labels.grasp_frame_l = test_frame_final[f"seq{seq_num}/{seq_name}/grasp_frame"]
	labels.init_frame_r = test_frame_final[f"seq{seq_num}/{seq_name}/init_frame"]
	labels.grasp_frame_r = test_frame_final[f"seq{seq_num}/{seq_name}/grasp_frame"]
	labels.kind = test_frame_final[f"seq{seq_num}/{seq_name}/kind"]

	return labels


if __name__ == '__main__':
	generate_new_train_test_set()
	direct_save = hand_path + "/grasp_label/final_labels/final/"
	if not op.exists(direct_save):
		os.system(f"mkdir -p {direct_save}")
	obj_name = "waffleiron"
	file_save = []
	for seq_num in range(1,11):
		if seq_num != 3 and seq_num != 5:
			direct = hand_path + f"/contact/s{seq_num}/"
			for root, d, file in os.walk(direct):
				for f in file:
					if f.endswith("_l.npy"):
						seq_name = f.replace("_a_l.npy","")
						obj_name_file = seq_name.split('_')[0]
						data_cat = seq_name.split('_')[1]
						if obj_name_file == obj_name:
							labels = label_gen_final(seq_num, seq_name) # all labels in frame
							file_save.append(labels)
							print(f"Finish articulate s{seq_num}/{seq_name}")
							print(f"Init frame: {labels.init_frame_l}")
							print(f"Grasp frame: {labels.grasp_frame_l}")
							print(f"kind: {labels.kind}")

	np.save(direct_save + f"{obj_name}.npy", file_save)
