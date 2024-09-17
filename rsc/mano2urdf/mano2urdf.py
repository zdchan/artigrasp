import argparse
import os

import numpy as np

from export_meshes import export_body_part_meshes
from mano_helpers import get_mano_data, get_mano_joint_names, to_joint_dict
from urdf import export_mano2urdf

dst_test = './rsc/mano/'
dst_working = '/local/home/luzheng/raisim_grasp_arctic/rsc/mano_double'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lh', '--is_rhand', help='if the model is right hand or not', default=True, action='store_false')
    parser.add_argument('-op', '--out-path', help='output location of urdf and meshes', type=str,
                        default=dst_working)

    args = parser.parse_args()
    is_rhand = args.is_rhand
    out_path = args.out_path

    model_path = 'MANO_RIGHT.pkl' if is_rhand else 'MANO_LEFT.pkl'
    model_path = os.path.join('../mano_v1_2/models', model_path)

    hand_name = 'rhand_mano' if is_rhand else 'lhand_mano'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    lbs_weight_matrix, verts, joints_dict = get_mano_data(model_path=model_path, is_rhand=is_rhand)

    joint_names = get_mano_joint_names(is_rhand)

    joint_mesh_data = export_body_part_meshes(out_path=out_path,
                                              lbs_weight_matrix=lbs_weight_matrix,
                                              vertices=verts,
                                              joint_names=joint_names,
                                              decimation_factor_obj=0.5,
                                              is_rhand=is_rhand)

    for idx, name in enumerate(joint_names):



    #pose_min_limit = np.loadtxt('./data/mano_mean_min_limit.txt').reshape((-1, 3))
    #pose_max_limit = np.loadtxt('./data/mano_mean_max_limit.txt').reshape((-1, 3))

    #pose_min_limit_dict = to_joint_dict(pose_min_limit)
    #pose_max_limit_dict = to_joint_dict(pose_max_limit)

    export_mano2urdf(hand_name=hand_name,
                     out_path=out_path,
                     joints_dict=joints_dict,
                     joint_mesh_data=joint_mesh_data,
                     is_rhand=is_rhand,
                     #pose_limits=[pose_min_limit_dict, pose_max_limit_dict]
                     )
