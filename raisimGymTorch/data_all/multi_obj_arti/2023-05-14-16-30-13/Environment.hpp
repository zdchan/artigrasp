//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include "raisim/World.hpp"
#include <vector>
#include "raisim/math.hpp"
#include <math.h>

namespace raisim {
    class ENVIRONMENT : public RaisimGymEnv {
 
    public:

        explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

            /// create world
            world_ = std::make_unique<raisim::World>();
            world_->addGround();
            world_->setERP(0.0);

            world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);

            /// add mano
            std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();
            resourceDir_ = resourceDir;
            mano_r_ = world_->addArticulatedSystem(resourceDir+"/mano_double/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
            mano_r_->setName("mano_r");

            /// add table
            box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
            box->setPosition(1.25, 0, 0.25);
            box->setAppearance("0.0 0.0 0.0 0.0");

            /// set PD control mode
            mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

            /// get actuation dimensions
            gcDim_ = mano_r_->getGeneralizedCoordinateDim();
            gvDim_ = mano_r_->getDOF();
            nJoints_ = gcDim_-3;

            gc_r_.setZero(gcDim_);
            gv_r_.setZero(gvDim_);
            gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);

            gc_l_.setZero(gcDim_);
            gv_l_.setZero(gvDim_);
            gc_set_l_.setZero(gcDim_); gv_set_l_.setZero(gvDim_);

            /// initialize all variables
            pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
            pTarget_l_.setZero(gcDim_); vTarget_l_.setZero(gvDim_); pTarget6_l_.setZero(6);
            final_pose_r_.setZero(nJoints_), final_obj_pos_b_.setZero(7), final_ee_pos_r_.setZero(num_bodyparts*3), final_vertex_normals_r_.setZero(num_contacts*3), contact_body_idx_r_.setZero(num_contacts), final_contact_array_r_.setZero(num_contacts);
            final_pose_l_.setZero(nJoints_), final_obj_pos_t_.setZero(7), final_ee_pos_l_.setZero(num_bodyparts*3), final_vertex_normals_l_.setZero(num_contacts*3), contact_body_idx_l_.setZero(num_contacts), final_contact_array_l_.setZero(num_contacts);
            rel_pose_r_.setZero(nJoints_), rel_obj_pos_r_.setZero(3), rel_objpalm_pos_r_.setZero(3), rel_body_pos_r_.setZero(num_bodyparts*3), rel_contact_pos_r_.setZero(num_contacts*3), rel_obj_pose_r_.setZero(3), contacts_r_.setZero(num_contacts), rel_contacts_r_.setZero(num_contacts), impulses_r_.setZero(num_contacts);
            rel_pose_l_.setZero(nJoints_), rel_obj_pos_l_.setZero(3), rel_objpalm_pos_l_.setZero(3), rel_body_pos_l_.setZero(num_bodyparts*3), rel_contact_pos_l_.setZero(num_contacts*3), rel_obj_pose_l_.setZero(3), contacts_l_.setZero(num_contacts), rel_contacts_l_.setZero(num_contacts), impulses_l_.setZero(num_contacts);
            actionDim_ = gcDim_;
            actionMean_r_.setZero(actionDim_);  actionStd_r_.setOnes(actionDim_);
            actionMean_l_.setZero(actionDim_);  actionStd_l_.setOnes(actionDim_);
            joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);
            Position_r.setZero(); Position_l.setZero(); Obj_Position_b.setZero(); Obj_Position_t.setZero(); Rel_fpos.setZero();
            obj_quat_b.setZero(); obj_quat_b[0] = 1.0;
            obj_quat_t.setZero(); obj_quat_t[0] = 1.0;
            Obj_linvel_b.setZero(); Obj_qvel_b.setZero();
            Obj_linvel_t.setZero(); Obj_qvel_t.setZero();
            rel_obj_vel_b.setZero(); rel_obj_qvel_b.setZero();
            rel_obj_vel_t.setZero(); rel_obj_qvel_t.setZero();
            bodyLinearVel_r_.setZero(); bodyAngularVel_r_.setZero();
            bodyLinearVel_l_.setZero(); bodyAngularVel_l_.setZero();
            init_or_r_.setZero(); init_or_l_.setZero(); init_rot_r_.setZero(); init_rot_l_.setZero();
            init_root_r_.setZero(); init_root_l_.setZero(); init_obj_.setZero(); init_obj_rot_.setZero(); init_obj_or_.setZero();
            obj_pose_r_.setZero(); obj_pose_l_.setZero(); obj_pos_init_.setZero(8); obj_pos_init_b_.setZero(7); obj_pos_init_t_.setZero(7);
            palm_world_pos_r_.setZero(); palm_world_pos_l_.setZero(); Fpos_world_r.setZero(); Fpos_world_l.setZero();
            final_obj_angle_.setZero(1); rel_obj_angle_.setZero(1); obj_angle_.setZero(1); obj_avel_.setZero(1);
            final_obj_pos_b_[3] = 1.0; final_obj_pos_t_[3] = 1.0;

            /// initialize 3D positions weights for fingertips higher than for other fingerparts
            finger_weights_.setOnes(num_bodyparts*3);
            finger_weights_.segment(16*3,3) *= 4;
            finger_weights_.segment(17*3,3) *= 4;
            finger_weights_.segment(18*3,3) *= 4;
            finger_weights_.segment(19*3,3) *= 4;
            finger_weights_.segment(20*3,3) *= 4;
            finger_weights_ /= finger_weights_.sum();
            finger_weights_ *= num_bodyparts*3;

            /// set PD gains
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.head(3).setConstant(50);
            jointDgain.head(3).setConstant(0.1);
            jointPgain.tail(nJoints_).setConstant(50.0);
            jointDgain.tail(nJoints_).setConstant(0.2);

            mano_r_->setPdGains(jointPgain, jointDgain);
            mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
            mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_r_ =  276;
            obDim_l_ = 273;
            obDouble_r_.setZero(obDim_r_);
            obDouble_l_.setZero(obDim_l_);

            root_guided =  cfg["root_guided"].As<bool>();

            float finger_action_std = cfg["finger_action_std"].As<float>();
            float rot_action_std = cfg["rot_action_std"].As<float>();

            /// retrieve joint limits from model
            joint_limits_ = mano_r_->getJointLimits();

            for(int i=0; i < int(gcDim_); i++){
                actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                actionMean_l_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                joint_limit_low[i] = joint_limits_[i][0];
                joint_limit_high[i] = joint_limits_[i][1];
            }

            /// set actuation parameters
            if (root_guided){
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.001);
                actionStd_r_.segment(3,3).setConstant(rot_action_std);
                actionStd_l_.setConstant(finger_action_std);
                actionStd_l_.head(3).setConstant(0.001);
                actionStd_l_.segment(3,3).setConstant(rot_action_std);
            }
            else{
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.01);
                actionStd_r_.segment(3,3).setConstant(0.01);
                actionStd_l_.setConstant(finger_action_std);
                actionStd_l_.head(3).setConstant(0.01);
                actionStd_l_.segment(3,3).setConstant(0.01);
            }

            world_->setMaterialPairProp("object", "object", 0.8, 0.0, 0.0);
            world_->setMaterialPairProp("object", "finger", 0.8, 0.0, 0.0);
            world_->setMaterialPairProp("finger", "finger", 0.8, 0.0, 0.0);

            /// Initialize reward
            rewards_r_.initializeFromConfigurationFile(cfg["reward"]);
            rewards_l_.initializeFromConfigurationFile(cfg["reward"]);

            /// start visualization server
            if (visualizable_) {
                if(server_) server_->lockVisualizationServerMutex();
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();

                /// Create table
                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                table_top->setPosition(1.25, 0, 0.475);
                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                leg1->setPosition(0.2625,0.4675,0.2375);
                leg2->setPosition(2.2275,0.4875,0.2375);
                leg3->setPosition(0.2625,-0.4675,0.2375);
                leg4->setPosition(2.2275,-0.4875,0.2375);

                /// initialize spheres for target 3D positions
                for(int i = 0; i < num_bodyparts; i++){
                    spheres[2*i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                    spheres[2*i+1] = server_->addVisualSphere(body_parts_l_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                }
                if(server_) server_->unlockVisualizationServerMutex();
            }
        }

        void init() final { }

        /// This function loads the object into the environment
        void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {

            /// Set standard properties
            raisim::Mat<3, 3> inertia;
            inertia.setIdentity();
            const raisim::Vec<3> com = {0, 0, 0};
            obj_weight_ = obj_weight[0];

            /// if obj is a primitive cylinder
            if (obj_type[0] == 0)
            {
                cylinder = static_cast<raisim::Cylinder*>(world_->addCylinder(obj_dim[0],obj_dim[1],obj_weight[0],"object", raisim::COLLISION(2)));
                cylinder->setCom(com);
                cylinder->setInertia(inertia);
                cylinder->setPosition(obj_pos_init_[0],obj_pos_init_[1],obj_pos_init_[2]);
                obj_idx_ = cylinder->getIndexInWorld();
                cylinder->setOrientation(1,0,0,0);
                cylinder->setVelocity(0,0,0,0,0,0);
                cylinder_mesh = true;

            }
            /// if obj is a primitive box
            else if (obj_type[0] == 1)
            {
                box_obj = static_cast<raisim::Box*>(world_->addBox(obj_dim[0],obj_dim[1],obj_dim[2], obj_weight[0],"object", raisim::COLLISION(2)));
                box_obj->setCom(com);
                box_obj->setInertia(inertia);
                box_obj->setPosition(obj_pos_init_[0],obj_pos_init_[1],obj_pos_init_[2]);
                obj_idx_ = box_obj->getIndexInWorld();
                box_obj->setOrientation(1,0,0,0);
                box_obj->setVelocity(0,0,0,0,0,0);
                box_obj_mesh = true;
            }
            /// if obj is a mesh
            else
            {
                std::string obj_name;
                /// if mesh is a processed and aligned mesh
                if (obj_type[0] == 3)
                    obj_name = resourceDir_ + "/meshes_simplified/" + ycb_objects_[obj_idx[0]] + "/mesh_aligned.obj";
                /// if mesh is a decimated mesh
                else
                    obj_name = resourceDir_ + "/meshes_simplified/" + ycb_objects_[obj_idx[0]] + "/textured_meshlab_quart.obj";
                obj_mesh_1 =  static_cast<raisim::Mesh*>(world_->addMesh(obj_name, obj_weight[0], inertia, com, 1.0,"",raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(63)));
                obj_idx_ = obj_mesh_1->getIndexInWorld();
                obj_mesh_1->setPosition(obj_pos_init_[0],obj_pos_init_[1],obj_pos_init_[2]);
                obj_mesh_1->setOrientation(1,0,0,0);
                obj_mesh_1->setVelocity(0,0,0,0,0,0);
            }

            if (visualizable_)
            {

                std::string obj_name_target = resourceDir_ + "/meshes_simplified/" + ycb_objects_[obj_idx[0]] + "/textured_meshlab.obj";
                obj_mesh_2 = static_cast<raisim::Mesh*>(world_->addMesh(obj_name_target, obj_weight[0], inertia, com, 1.0,"",raisim::COLLISION(10),raisim::COLLISION(10)));
                obj_mesh_2->setPosition(obj_pos_init_[0],obj_pos_init_[1],obj_pos_init_[2]);
                obj_mesh_2->setOrientation(1,0,0,0);
                obj_mesh_2->setVelocity(0,0,0,0,0,0);
                obj_mesh_2->setAppearance("0 1 0 0.5");
            }

        }

        void load_articulated(const std::string& obj_model){
            /// Set standard properties
            raisim::Mat<3, 3> inertia;
            inertia.setIdentity();
            const raisim::Vec<3> com = {0, 0, 0};

            arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/arctic/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
            //arcticVisual = server_->addVisualArticulatedSystem("arcticVisual", resourceDir_+"/arctic/"+obj_model, 1, 0, 0, 1);
            //arctic = &arcticVisual->obj;
            arctic->setName("arctic");
            gcDim_obj = arctic->getGeneralizedCoordinateDim();
            gvDim_obj = arctic->getDOF();
            arctic->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_obj));
            arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
            auto top_id = arctic->getBodyIdx("top");
            obj_weight_ = arctic->getMass(top_id);
            //std::cout << "obj index" << arctic->getIndexInWorld() << std::endl;
            //std::cout << "hand index" << mano_l_->getIndexInWorld() << std::endl;
        }

        /// Resets the object and hand to its initial pose
        void reset() final {

            if (first_reset_)
            {
                first_reset_=false;
            }
            else{
                Eigen::VectorXd obj_goal_angle;
                obj_goal_angle.setZero(1);
                obj_goal_angle[0] = obj_pos_init_[7];
                /// all settings to initial state configuration
                actionMean_r_.setZero();
                actionMean_l_.setZero();
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                gvDim_obj = arctic->getDOF();
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(obj_goal_angle, Eigen::VectorXd::Zero(gvDim_obj));
                //obj_pos_init: reset pose of object

                box->clearExternalForcesAndTorques();
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                auto top_id = arctic->getBodyIdx("top");
                arctic->getAngularVelocity(top_id, Obj_qvel_t);
                updateObservation();
                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);


//                std::cout << " box pos" << box->getPosition() << std::endl;
            }

        }

        /// Resets the state to a user defined input
        // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
        // init_state_l in right-hand coord
        void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                         const Eigen::Ref<EigenVec>& init_state_l,
                         const Eigen::Ref<EigenVec>& init_vel_r,
                         const Eigen::Ref<EigenVec>& init_vel_l,
                         const Eigen::Ref<EigenVec>& obj_pose) final {
            //std::cout << "reset state: " << std::endl;
            /// reset gains (only required in case for inference)
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.head(3).setConstant(50);
            jointDgain.head(3).setConstant(0.1);
            jointPgain.tail(nJoints_).setConstant(50.0);
            jointDgain.tail(nJoints_).setConstant(0.2);
            mano_r_->setPdGains(jointPgain, jointDgain);

            Eigen::VectorXd gen_force;
            gen_force.setZero(gcDim_);
            mano_r_->setGeneralizedForce(gen_force);


            /// reset box position (only required in case for inference)
            box->setPosition(1.25, 0, 0.25);
            box->setOrientation(1,0,0,0);
            box->setVelocity(0,0,0,0,0,0);

            /// set initial hand pose (45 DoF) and velocity (45 DoF)
            gc_set_r_.head(6).setZero();
            gc_set_r_.tail(nJoints_-3) = init_state_r.tail(nJoints_-3).cast<double>(); //.cast<double>();
            gv_set_r_ = init_vel_r.cast<double>(); //.cast<double>();
            gc_set_l_.head(6).setZero();
            gc_set_l_.tail(nJoints_-3) = init_state_l.tail(nJoints_-3).cast<double>(); //.cast<double>();
            gv_set_l_ = init_vel_l.cast<double>(); //.cast<double>();
            //std::cout << "reset coord: \n" << gc_set_l_ << std::endl;
            mano_r_->setState(gc_set_r_, gv_set_r_);

            /// set initial root position in global frame as origin in new coordinate frame
            init_root_r_  = init_state_r.head(3);
            init_root_l_  = init_state_l.head(3);

            init_obj_ = obj_pose.head(3).cast<double>();

            /// set initial root orientation in global frame as origin in new coordinate frame
            raisim::Vec<4> quat;
            raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
            raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
            raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse
            raisim::eulerToQuat(init_state_l.segment(3,3), quat); // initial base ori, in quat
            raisim::quatToRotMat(quat, init_rot_l_); // ..., in matrix

            raisim::transpose(init_rot_l_, init_or_l_); // ..., inverse

            int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
            int arcticVelDim = arctic->getDOF();
            Eigen::VectorXd arcticCoord, arcticVel;
            arcticCoord.setZero(arcticCoordDim);
            arcticVel.setZero(arcticVelDim);
            arcticCoord[0] = obj_pose[7];

            raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
            raisim::transpose(init_obj_rot_, init_obj_or_);
            arctic->setBasePos(init_obj_);
            arctic->setBaseOrientation(init_obj_rot_);
            arctic->setState(arcticCoord, arcticVel);
            mano_r_->setBasePos(init_root_r_);
            mano_r_->setBaseOrientation(init_rot_r_);
            mano_r_->setState(gc_set_r_, gv_set_r_);


            /// set initial object pose
            obj_pos_init_  = obj_pose.cast<double>(); // 8 dof

            set_gc_for_arctic(obj_pos_init_b_, obj_pos_init_t_, obj_pos_init_);


            /// Set action mean to initial pose (first 6DoF since start at 0)

            actionMean_r_.setZero();
            actionMean_r_.tail(nJoints_-3) = gc_set_r_.tail(nJoints_-3);
            actionMean_l_.setZero();
            actionMean_l_.tail(nJoints_-3) = gc_set_l_.tail(nJoints_-3);

            motion_synthesis = false;
            root_guiding_counter_r_ = 0;
            root_guiding_counter_l_ = 0;
            obj_table_contact_ = 0;


            gen_force.setZero(gcDim_);
            mano_r_->setGeneralizedForce(gen_force);
            updateObservation();
        }

        /// This function is used to set user specified goals that the policy is conditioned on
        /// Obj_pos: Object goal state in global frame (7), 3 translation + 4 quaternion rotation
        /// ee_pos: Hand 3D joint position goal state in global frame (63DoF)
        /// goal_pose: Hand goal pose (48DoF), 3DoF global euler rotation + 45DoF local joint angles
        /// contact_pos: Deprecated (63 DoF)
        /// goal_contacts: Hand parts that should be in contact (16 hand parts)

        //obj_goal_pos: 8dof, [trans(3), rot(4, quat), joint angle(1))

        //states in the free base: final_obj_pos_b_, final_obj_pos_t_,
        void set_goals(const Eigen::Ref<EigenVec>& obj_goal_angle,
                       const Eigen::Ref<EigenVec>& obj_goal_pos,
                       const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                       const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                       const Eigen::Ref<EigenVec>& goal_pose_r,
                       const Eigen::Ref<EigenVec>& goal_pose_l,
                       const Eigen::Ref<EigenVec>& goal_qpos_r,
                       const Eigen::Ref<EigenVec>& goal_qpos_l,
                       const Eigen::Ref<EigenVec>& goal_contacts_r,
                       const Eigen::Ref<EigenVec>& goal_contacts_l)
        final {
            //std::cout << "set goal:" << std::endl;
            raisim::Vec<4> quat_goal_hand_w, quat_goal_hand_r, quat_obj_init_t, quat_obj_init_b;
            raisim::Vec<3> euler_goal_pose;
            raisim::Mat<3,3> rotm_goal_hand_r, rotm_goal_hand_l, mat_temp, mat_rot_left, rotm_goal_b, rotm_goal_tran_b;

            final_obj_angle_[0] = obj_goal_angle[0];
//            Eigen::VectorXd final_obj_b_test, final_obj_t_test, final_obj_test;
//            //test code for motion synthesis, goal angle 0.7
//            final_obj_b_test.setZero(7); final_obj_t_test.setZero(7); final_obj_test.setZero(8);
//            final_obj_test = obj_goal_pos.cast<double>();
//            final_obj_test[7] = 0.7;
//            set_gc_for_arctic(final_obj_b_test, final_obj_t_test, final_obj_test);


//            Eigen::VectorXd final_qpos_r = goal_qpos_r;
            Eigen::VectorXd final_qpos_r = goal_qpos_r.cast<double>();
            Eigen::VectorXd obj_goal_pos_cast;
            obj_goal_pos_cast = obj_goal_pos.cast<double>();
//            if(right_kind_idx == 9){
//                obj_goal_pos_cast[7] = 0;
//            }
            set_gc_for_arctic(final_obj_pos_b_, final_obj_pos_t_, obj_goal_pos_cast);
            raisim::quatToRotMat(final_obj_pos_t_.tail(4), Obj_ori_test);
            /// set final object pose

            /// convert object and handpose pose to rotation matrix format
            raisim::quatToRotMat(final_obj_pos_b_.tail(4), Obj_orientation_temp_b); // Obj_orientation_temp: orientation of object goal pose, in matrix
            raisim::quatToRotMat(final_obj_pos_t_.tail(4), Obj_orientation_temp_t);

            quat_obj_init_b = final_obj_pos_b_.tail(4).cast<double>(); //quat_obj_init: orientation of object goal pose, in quaternion
            quat_obj_init_t = final_obj_pos_t_.tail(4).cast<double>();

            raisim::transpose(Obj_orientation_temp_b, Obj_orientation_b);
            raisim::transpose(Obj_orientation_temp_t, Obj_orientation_t); // Obj_orientation: relative orientation of world coordinate to object

            raisim::eulerToQuat(goal_pose_r.head(3), quat_goal_hand_w);
            raisim::quatToRotMat(quat_goal_hand_w, root_pose_world_r_);

            /// Compute and set object relative goal hand pose
            raisim::quatInvQuatMul(quat_obj_init_t, quat_goal_hand_w, quat_goal_hand_r);  //修改
            raisim::quatToRotMat(quat_goal_hand_r, rotm_goal_hand_r);
            raisim::RotmatToEuler(rotm_goal_hand_r, euler_goal_pose);

            final_pose_r_ = goal_pose_r.cast<double>();
            final_pose_r_.head(3) = euler_goal_pose.e().cast<double>(); // change the orientation of hand final pose to the object frame

            raisim::eulerToQuat(goal_pose_l.head(3), quat_goal_hand_w);
            raisim::quatToRotMat(quat_goal_hand_w, root_pose_world_l_);

            /// Compute and set object relative goal hand pose
            raisim::quatToRotMat(quat_obj_init_b, rotm_goal_b);   //修改
            raisim::transpose(rotm_goal_b, rotm_goal_tran_b);
            raisim::matmul(rotm_goal_tran_b, root_pose_world_l_, rotm_goal_hand_l);
            //raisim::quatInvQuatMul(quat_obj_init_t, quat_goal_hand_w, quat_goal_hand_r);
            //raisim::quatToRotMat(quat_goal_hand_r, rotm_goal_hand_r);
            //raisim::matmul(Right2Left, rotm_goal_hand_r, mat_temp);
            //raisim::matmul(mat_temp, Right2Left, mat_rot_left);
            raisim::RotmatToEuler(rotm_goal_hand_l, euler_goal_pose);

            final_pose_l_ = goal_pose_l.cast<double>();
            final_pose_l_.head(3) = euler_goal_pose.e().cast<double>();
            //std::cout << "goal state: \n" << final_pose_l_ << std::endl;

            mano_r_->setBasePos(goal_qpos_r.head(3));
            mano_r_->setBaseOrientation(root_pose_world_r_);
            final_qpos_r.head(6).setZero();
            mano_r_->setGeneralizedCoordinate(final_qpos_r);

            raisim::Vec<3> ee_goal_exact_r;
            /// Compute and convert hand 3D joint positions into object relative frame
            for(int i = 0; i < num_bodyparts; i++){
                Position_l[0] = ee_goal_pos_l[i*3] - final_obj_pos_b_[0];
                Position_l[1] = ee_goal_pos_l[i*3+1] - final_obj_pos_b_[1];
                Position_l[2] = ee_goal_pos_l[i*3+2] - final_obj_pos_b_[2];

                raisim::matvecmul(Obj_orientation_b, Position_l, Rel_fpos);

                final_ee_pos_l_[i*3] = Rel_fpos[0];
                final_ee_pos_l_[i*3+1] = Rel_fpos[1];
                final_ee_pos_l_[i*3+2] =  Rel_fpos[2]; // change the ee pose to object frame
            }
            for(int i = 0; i < num_bodyparts; i++){
                //Position_l[0] = ee_goal_pos_l[i*3] - final_obj_pos_t_[0];
                //Position_l[1] = ee_goal_pos_l[i*3+1] - final_obj_pos_t_[1];
                //Position_l[2] = ee_goal_pos_l[i*3+2] - final_obj_pos_t_[2];
                mano_r_->getFramePosition(body_parts_r_[i], ee_goal_exact_r);
                Position_r[0] = ee_goal_exact_r[0] - final_obj_pos_t_[0];
                Position_r[1] = ee_goal_exact_r[1] - final_obj_pos_t_[1];
                Position_r[2] = ee_goal_exact_r[2] - final_obj_pos_t_[2];

                raisim::matvecmul(Obj_orientation_t, Position_r, Rel_fpos);

                final_ee_pos_r_[i*3] = Rel_fpos[0];
                final_ee_pos_r_[i*3+1] = Rel_fpos[1];
                final_ee_pos_r_[i*3+2] = Rel_fpos[2];
            }

            /// Intialize and set goal contact array
            num_active_contacts_r_ = float(goal_contacts_r.sum());
            num_active_contacts_l_ = float(goal_contacts_l.sum());
            final_contact_array_r_ = goal_contacts_r.cast<double>();
            final_contact_array_l_ = goal_contacts_l.cast<double>();

            for(int i = 0; i < num_contacts ;i++){
                contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
            }
            //make a map from body index to pre-defined name list

            /// Compute contact weight
            k_contact_r_ = 1.0 / num_active_contacts_r_; // parameter about number of contact body
            k_contact_l_ = 1.0 / num_active_contacts_l_;

            // change final object pose of b and t to the true pose in fixed base
            Eigen::VectorXd obj_goal_pos_true;
            obj_goal_pos_true = obj_pos_init_.cast<double>();
            obj_goal_pos_true[7] = obj_goal_angle[0];
            set_gc_for_arctic(final_obj_pos_b_, final_obj_pos_t_, obj_goal_pos_true);

        }

        /// This function takes an environment step given an action (51DoF) input
        // action_l in left-hand coord
        float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {
            //std::cout << "step in: " << std::endl;
            raisim::Vec<4> obj_orientation_quat, quat_final_pose, quat_world;
            raisim::Mat<3, 3> rot, rot_trans, rot_world, rot_goal, rotmat_final_obj_pos, rotmat_final_obj_pos_trans;
            raisim::Vec<3> obj_pos_raisim_b, obj_pos_raisim_t, euler_goal_world, final_obj_pose_mat, hand_pos_world, hand_pose, act_pos_r, act_pos_l, act_or_pose_r, act_or_pose_l;
            raisim::transpose(Obj_orientation_temp_b, Obj_orientation_b);
            raisim::transpose(Obj_orientation_temp_t, Obj_orientation_t);
            obj_pos_raisim_b[0] = final_obj_pos_b_[0] - Obj_Position_b[0];
            obj_pos_raisim_b[1] = final_obj_pos_b_[1] - Obj_Position_b[1];
            obj_pos_raisim_b[2] = final_obj_pos_b_[2] - Obj_Position_b[2]; // distance from current position to final position, in world coord
            obj_pos_raisim_t[0] = final_obj_pos_t_[0] - Obj_Position_t[0];
            obj_pos_raisim_t[1] = final_obj_pos_t_[1] - Obj_Position_t[1];
            obj_pos_raisim_t[2] = final_obj_pos_t_[2] - Obj_Position_t[2];

            if (motion_synthesis)
            {
                raisim::Mat<3,3> rotm_goal_hand_r, rotm_goal_hand_w, rotm_goal_hand_h;
                raisim::Vec<4> final_pose_quat_r;
                raisim::Vec<3> act_pose_r;

                raisim::eulerToQuat(final_pose_r_.head(3), final_pose_quat_r);
                raisim::quatToRotMat(final_pose_quat_r, rotm_goal_hand_r);
                raisim::matmul(Obj_ori_test, rotm_goal_hand_r, rotm_goal_hand_w);
                raisim::matmul(init_or_r_, rotm_goal_hand_w, rotm_goal_hand_h);
                raisim::RotmatToEuler(rotm_goal_hand_h, act_pose_r);

                //actionMean_l_.segment(3,3) = rel_obj_pose_l_ * std::min(1.0,(0.0005*root_guiding_counter_l_));
                actionMean_r_.segment(3,3) = act_pose_r.e();
                //root_guiding_counter_l_ += 1;

            }

            /// The following applies the wrist guidance technique (compare with paper)
            else if (root_guided){
                /// Retrieve current object pose
                raisim::Mat<3,3> rotm_goal_hand_r, rotm_goal_hand_w, rotm_goal_hand_h;
                raisim::Vec<4> final_pose_quat_r;
                raisim::Vec<3> act_pose_r;
                auto bottom_id = arctic->getBodyIdx("bottom");
                auto top_id = arctic->getBodyIdx("top");
                arctic->getPosition(bottom_id, Obj_Position_b); //Obj_Position: object base position at this time
                arctic->getPosition(top_id, Obj_Position_t);
                arctic->getOrientation(bottom_id, Obj_orientation_temp_b); //Obj_orientation_temp: object bottom orientation at this time, in matrix
                arctic->getOrientation(top_id, Obj_orientation_temp_t);
                raisim::rotMatToQuat(Obj_orientation_temp_b, obj_quat_b); //obj_quat: object bottom orientation at this time, in quat
                raisim::rotMatToQuat(Obj_orientation_temp_t, obj_quat_t);

                /// Convert final root hand translation back from (current) object into world frame
//                raisim::matvecmul(Obj_orientation_temp_b, final_ee_pos_r_.head(3), Fpos_world_r);
//                raisim::vecadd(Obj_Position_b, Fpos_world_r);
                raisim::matvecmul(Obj_orientation_temp_t, final_ee_pos_r_.head(3), Fpos_world_r);
                raisim::vecadd(Obj_Position_t, Fpos_world_r);
                raisim::matvecmul(Obj_orientation_temp_b, final_ee_pos_l_.head(3), Fpos_world_l);
                raisim::vecadd(Obj_Position_b, Fpos_world_l);

                raisim::vecsub(Fpos_world_r, init_root_r_, act_pos_r); // compute distance of current root to initial root in world frame
                raisim::matvecmul(init_or_r_, act_pos_r, act_or_pose_r); // rotate the world coordinate into hand's origin frame (from the start of the episode)

                raisim::vecsub(Fpos_world_l, init_root_l_, act_pos_l); // compute distance of current root to initial root in world frame
                raisim::matvecmul(init_or_l_, act_pos_l, act_or_pose_l); // rotate the world coordinate into hand's origin frame (from the start of the episode)

                actionMean_r_.head(3) = act_or_pose_r.e();
                actionMean_l_.head(3) = act_or_pose_l.e();
                //actionMean_l_.segment(3,3) = act_pose_l.e();
                //mano_l_->setGeneralizedCoordinate(actionMean_l_);
                //actionMean_r_.head(3) = act_pos_r.e();
                //actionMean_l_.head(3) = act_pos_l.e();
            }

            /// Compute position target for actuators
            pTarget_r_ = action_r.cast<double>();
            if (!root_guided) {
                raisim::Vec<3> wrist_act, temp;
                raisim::matvecmul(init_obj_rot_, action_r.head(3), temp);
                raisim::matvecmul(init_or_r_, temp, wrist_act);
                pTarget_r_.head(3) = wrist_act.e();
            }
            pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
            pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (48DoF)

            pTarget_l_ = action_l.cast<double>();
            if (!root_guided) {
                raisim::Vec<3> wrist_act, temp;
                raisim::matvecmul(init_obj_rot_, action_l.head(3), temp);
                raisim::matvecmul(init_or_l_, temp, wrist_act);
                pTarget_l_.head(3) = wrist_act.e();
            }
            pTarget_l_ = pTarget_l_.cwiseProduct(actionStd_l_); //residual action * scaling
            pTarget_l_ += actionMean_l_; //add wrist bias (first 3DOF) and last pose (48DoF)

            /// Clip targets to limits
            Eigen::VectorXd pTarget_clipped_r, pTarget_clipped_l;
            pTarget_clipped_r.setZero(gcDim_);
            pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);
            pTarget_clipped_l.setZero(gcDim_);
            pTarget_clipped_l = pTarget_l_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

            /// Set PD targets (velocity zero)
            mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);

            /// Apply N control steps
            for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();
                if(server_) server_->unlockVisualizationServerMutex();
            }

            /// update observation and set new mean to the latest pose
            updateObservation();
            actionMean_r_ = gc_r_;
            actionMean_l_ = gc_l_;

            /// Compute general reward terms
            pose_reward_r_ = -(rel_pose_r_).norm();
            root_pos_reward_r_ = -rel_body_pos_r_.head(3).norm();
            root_pose_reward_r_ = -rel_pose_r_.head(3).squaredNorm();
            pos_reward_r_ = -rel_body_pos_r_.cwiseProduct(finger_weights_).squaredNorm();
            obj_reward_r_ = -rel_obj_pos_r_.norm();
            contact_pos_reward_r_ =  rel_contact_pos_r_.squaredNorm();

            pose_reward_l_ = -(rel_pose_l_).norm();
            root_pos_reward_l_ = -rel_body_pos_l_.head(3).norm();
            root_pose_reward_l_ = -rel_pose_l_.head(3).squaredNorm();
            pos_reward_l_ = -rel_body_pos_l_.cwiseProduct(finger_weights_).squaredNorm();
            obj_reward_l_ = -rel_obj_pos_l_.norm();
            contact_pos_reward_l_ =  rel_contact_pos_l_.squaredNorm();

            //obj_angle_reward_ = 8*(rel_obj_angle_.norm() < 0.6)
            //                    + 2*(rel_obj_angle_.norm() < 0.4)
            //                    + 10*(rel_obj_angle_.norm() < 0.2)
            //                    +
            //                    - rel_obj_angle_.norm();

            obj_angle_reward_ = - rel_obj_angle_.norm();

            /// Compute regularization rewards
            rel_obj_reward_r_ = rel_obj_vel_t.squaredNorm();
            body_vel_reward_r_ = bodyLinearVel_r_.squaredNorm();
            body_qvel_reward_r_ = bodyAngularVel_r_.squaredNorm();
            contact_reward_r_ = k_contact_r_*(rel_contacts_r_.sum());
            impulse_reward_r_ = ((final_contact_array_r_.cwiseProduct(impulses_r_)).sum());

            rel_obj_reward_l_ = rel_obj_vel_b.squaredNorm();
            body_vel_reward_l_ = bodyLinearVel_l_.squaredNorm();
            body_qvel_reward_l_ = bodyAngularVel_l_.squaredNorm();
            contact_reward_l_ = k_contact_l_*(rel_contacts_l_.sum());
            impulse_reward_l_ = ((final_contact_array_l_.cwiseProduct(impulses_l_)).sum());

            obj_avel_reward_ = obj_avel_.squaredNorm();

            if(isnan(impulse_reward_r_))
                impulse_reward_r_ = 0.0;
            if(isnan(impulse_reward_l_))
                impulse_reward_l_ = 0.0;

            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            right_hand_torque.setZero(gcDim_);
            right_wrist_torque.setZero(6);
            right_hand_torque = mano_r_->getGeneralizedForce().e();
            right_wrist_torque = right_hand_torque.head(6);
//            std::cout<<right_hand_torque.squaredNorm()<<std::endl;

            rewards_r_.record("pos_reward", std::max(-10.0, pos_reward_r_));
            rewards_r_.record("root_pos_reward_", std::max(-10.0, root_pos_reward_r_));
            rewards_r_.record("root_pose_reward_", std::max(-10.0, root_pose_reward_r_));
            rewards_r_.record("pose_reward", std::max(-10.0, pose_reward_r_));
            rewards_r_.record("contact_pos_reward", std::max(-10.0, contact_pos_reward_r_));
            rewards_r_.record("contact_reward", std::max(-10.0, contact_reward_r_));
            rewards_r_.record("obj_reward", std::max(-10.0, obj_reward_r_));
            rewards_r_.record("obj_pose_reward_", std::max(-10.0, obj_pose_reward_r_));
            rewards_r_.record("impulse_reward", std::min(impulse_reward_r_, obj_weight_*5));
            rewards_r_.record("rel_obj_reward_", std::max(0.0, rel_obj_reward_r_));
            rewards_r_.record("body_vel_reward_", std::max(0.0,body_vel_reward_r_));
            rewards_r_.record("body_qvel_reward_", std::max(0.0,body_qvel_reward_r_));
            rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));
            rewards_r_.record("obj_angle_reward_", std::max(-10.0, obj_angle_reward_));
            rewards_r_.record("obj_avel_reward_", std::max(-10.0, obj_avel_reward_));

            rewards_l_.record("pos_reward", std::max(-10.0, pos_reward_l_));
            rewards_l_.record("root_pos_reward_", std::max(-10.0, root_pos_reward_l_));
            rewards_l_.record("root_pose_reward_", std::max(-10.0, root_pose_reward_l_));
            rewards_l_.record("pose_reward", std::max(-10.0, pose_reward_l_));
            rewards_l_.record("contact_pos_reward", std::max(-10.0, contact_pos_reward_l_));
            rewards_l_.record("contact_reward", std::max(-10.0, contact_reward_l_));
            rewards_l_.record("obj_reward", std::max(-10.0, obj_reward_l_));
            rewards_l_.record("obj_pose_reward_", std::max(-10.0,obj_pose_reward_l_));
            rewards_l_.record("impulse_reward", std::min(impulse_reward_l_, obj_weight_*5));
            rewards_l_.record("rel_obj_reward_", std::max(0.0, rel_obj_reward_l_));
            rewards_l_.record("body_vel_reward_", std::max(0.0,body_vel_reward_l_));
            rewards_l_.record("body_qvel_reward_", std::max(0.0,body_qvel_reward_l_));
            rewards_l_.record("obj_angle_reward_", std::max(-10.0, obj_angle_reward_));
            rewards_l_.record("obj_avel_reward_", std::max(-10.0, obj_avel_reward_));
            //std::map<std::string, float> reward_map = rewards_l_.getStdMap();
            //for(auto& item: reward_map){
            //    std::cout << item.first << ": " << item.second << std::endl;
            //}

            rewards_sum_[0] = rewards_r_.sum();
            rewards_sum_[1] = rewards_l_.sum();

            return rewards_sum_;
        }

        /// This function computes and updates the observation/state space
        void updateObservation() {
            // update observation
            //std::cout << "observe: " << std::endl;
            raisim::Vec<4> quat, quat_hand, quat_obj_init;
            raisim::Vec<3> body_vel, obj_frame_diff_r, obj_frame_diff_l, obj_frame_diff_w, obj_frame_diff_h_r, obj_frame_diff_h_l, euler_hand_r, euler_hand_l, sphere_pos, norm_pos, rel_wbody_root, final_obj_euler, euler_obj, rel_rbody_root, rel_body_table_r, rel_body_table_l, rel_obj_init_b, rel_obj_init_t, rel_objpalm_r, rel_objpalm_l, rel_obj_pose_r3;
            raisim::Mat<3,3> rot, rot_mult_r, rot_mult_l, body_orientation_transpose_r, body_orientation_transpose_l, palm_world_pose_mat_r, palm_world_pose_mat_l, palm_world_pose_mat_trans_r, palm_world_pose_mat_trans_l, obj_pose_wrist_mat_r, obj_pose_wrist_mat_l, rel_pose_mat, final_obj_rotmat_temp, diff_obj_pose_mat, final_obj_wrist, obj_wrist, obj_wrist_trans, final_obj_pose_mat, mat_temp, mat_rot_left;

            contacts_r_.setZero();
            rel_contacts_r_.setZero();
            impulses_r_.setZero();
            contacts_l_.setZero();
            rel_contacts_l_.setZero();
            impulses_l_.setZero();

            /// Get updated hand state
            mano_r_->getState(gc_r_, gv_r_);
            //std::cout << "generalized coord: \n" << gc_l_ << std::endl;

            /// Get updated object pose
            auto bottom_id = arctic->getBodyIdx("bottom");
            auto top_id = arctic->getBodyIdx("top");
            arctic->getPosition(bottom_id, Obj_Position_b); //Obj_Position: object base position at this time
            arctic->getPosition(top_id, Obj_Position_t);
            arctic->getOrientation(bottom_id, Obj_orientation_temp_b); //Obj_orientation_temp: object bottom orientation at this time, in matrix
            arctic->getOrientation(top_id, Obj_orientation_temp_t);
            raisim::rotMatToQuat(Obj_orientation_temp_b, obj_quat_b); //obj_quat: object bottom orientation at this time, in quat
            raisim::rotMatToQuat(Obj_orientation_temp_t, obj_quat_t);
            arctic->getAngularVelocity(bottom_id, Obj_qvel_b);
            arctic->getAngularVelocity(top_id, Obj_qvel_t);
            arctic->getVelocity(bottom_id, Obj_linvel_b);
            arctic->getVelocity(top_id, Obj_linvel_t);
            obj_angle_ = arctic->getGeneralizedCoordinate().e();
            obj_avel_ = arctic->getGeneralizedVelocity().e();
            rel_obj_angle_ = final_obj_angle_ - obj_angle_;

            raisim::transpose(Obj_orientation_temp_b, Obj_orientation_b);
            raisim::transpose(Obj_orientation_temp_t, Obj_orientation_t);

            /// compute relative hand pose to final pose
            rel_pose_r_ = final_pose_r_ - gc_r_.tail(gcDim_-3);
            rel_pose_l_ = final_pose_l_ - gc_l_.tail(gcDim_-3);

            /// compute object pose in wrist frame
            mano_r_->getFrameOrientation(body_parts_r_[0], palm_world_pose_mat_r);
            raisim::transpose(palm_world_pose_mat_r,palm_world_pose_mat_trans_r);

            raisim::matmul(palm_world_pose_mat_trans_r, Obj_orientation_temp_t, obj_pose_wrist_mat_r);
            raisim::RotmatToEuler(obj_pose_wrist_mat_r, obj_pose_r_); // obj_pose_: object pose in wrist frame

//            //Calculate wrist pose in object frame
//            raisim::Vec<3> wrist_temp;
//            raisim::matvecmul(init_rot_r_, gc_r_.head(3), wrist_temp);
//            raisim::vecadd(init_root_r_, wrist_temp);
//            raisim::vecsub(init_obj_, wrist_temp);
//            raisim::matvecmul(init_obj_or_, wrist_temp, wrist_pos_obj_r_); //需要修改
//
//            mano_r_->getFrameOrientation(body_parts_r_[0], Body_orientation_r);
//            raisim::Mat<3,3> wrist_pose_obj_r_mat;
//            raisim::matmul(init_obj_or_, Body_orientation_r, wrist_pose_obj_r_mat);
//            raisim::RotmatToEuler(wrist_pose_obj_r_mat, wrist_pose_obj_r_); //需要修改
            wrist_pos_obj_r_.setZero();
            wrist_pose_obj_r_.setZero();

            /// iterate over all hand parts to compute relative distances, poses, etc.
            for(int i = 0; i < num_bodyparts ; i++){
                mano_r_->getFramePosition(body_parts_r_[i], Position_r); // Position: body position of a body in mano
                mano_r_->getFrameOrientation(body_parts_r_[i], Body_orientation_r); // Body_orientation:
                /// for the hand root, compute relevant features
                if (i==0)
                {
                    raisim::transpose(Body_orientation_r, body_orientation_transpose_r);
                    rel_objpalm_r[0] = Position_r[0]-Obj_Position_t[0];
                    rel_objpalm_r[1] = Position_r[1]-Obj_Position_t[1];
                    rel_objpalm_r[2] = Position_r[2]-Obj_Position_t[2];

                    rel_objpalm_pos_r_ = Body_orientation_r.e().transpose()*rel_objpalm_r.e();
                    //rel_objpalm_pos_l_[0] = -rel_objpalm_pos_l_[0];

                    rel_body_table_r[0] = 0.0;
                    rel_body_table_r[1] = 0.0;
                    rel_body_table_r[2] = Position_r[2]-0.5;
                    rel_body_table_pos_r_ = Body_orientation_r.e().transpose()*rel_body_table_r.e(); // z-distance to the table in wrist coordinates
                    rel_body_table_l[0] = 0.0;
                    rel_body_table_l[1] = 0.0;
                    rel_body_table_l[2] = Position_l[2]-0.5;
                    rel_body_table_pos_l_ = Body_orientation_l.e().transpose()*rel_body_table_l.e();
                    //rel_body_table_pos_l_[0] = -rel_body_table_pos_l_[0];

                    rel_obj_init_b[0] = obj_pos_init_b_[0] - Obj_Position_b[0];
                    rel_obj_init_b[1] = obj_pos_init_b_[1] - Obj_Position_b[1];
                    rel_obj_init_b[2] = obj_pos_init_b_[2] - Obj_Position_b[2];
                    rel_obj_init_t[0] = obj_pos_init_t_[0] - Obj_Position_t[0];
                    rel_obj_init_t[1] = obj_pos_init_t_[1] - Obj_Position_t[1];
                    rel_obj_init_t[2] = obj_pos_init_t_[2] - Obj_Position_t[2];

                    rel_obj_pos_r_ = Body_orientation_r.e().transpose()*rel_obj_init_t.e(); // object displacement from initial position in wrist coordinates
                    rel_obj_pos_l_ = Body_orientation_l.e().transpose()*rel_obj_init_b.e();
                    //rel_obj_pos_l_[0] = -rel_obj_pos_l_[0];

                    raisim::matmul(Obj_orientation_t, Body_orientation_r, rot_mult_r); // current global wirst pose in object relative frame
                    raisim::RotmatToEuler(rot_mult_r, euler_hand_r);
                    raisim::matmul(Obj_orientation_b, Body_orientation_l, rot_mult_l);
                    //raisim::matmul(Right2Left, rot_mult_l, mat_temp);
                    //raisim::matmul(mat_temp, Right2Left, mat_rot_left);
                    raisim::RotmatToEuler(rot_mult_l, euler_hand_l);

                    rel_pose_r_.head(3) = final_pose_r_.head(3) - euler_hand_r.e(); // difference between target and current global wrist pose
                    rel_pose_l_.head(3) = final_pose_l_.head(3) - euler_hand_l.e();
                    //std::cout << "present root pose: \n" << euler_hand_l.e() << std::endl;
                    //std::cout << "rel_pose_l_.head(3): \n" << rel_pose_l_.head(3) << std::endl;

                    bodyLinearVel_r_ =  gv_r_.segment(0, 3);
                    bodyAngularVel_r_ = gv_r_.segment(3, 3);
                    bodyLinearVel_l_ =  gv_l_.segment(0, 3);
                    bodyAngularVel_l_ = gv_l_.segment(3, 3);

                    rel_obj_vel_t = Body_orientation_r.e().transpose() * Obj_linvel_t.e(); // object velocity in wrist frame
                    rel_obj_qvel_t = Body_orientation_r.e().transpose() * Obj_qvel_t.e();
                    rel_obj_vel_b = Body_orientation_l.e().transpose() * Obj_linvel_b.e();
                    //rel_obj_vel_t[0] = -rel_obj_vel_t[0];
                    rel_obj_qvel_b = Body_orientation_l.e().transpose() * Obj_qvel_b.e();
                    //rel_obj_qvel_t[1] = -rel_obj_qvel_t[1];
                    //rel_obj_qvel_t[2] = -rel_obj_qvel_t[2];

                    mano_r_->getFramePosition(body_parts_r_[0], palm_world_pos_r_);

//                    raisim::quatToRotMat(final_obj_pos_t_.segment(3,4), final_obj_rotmat_temp);
//                    raisim::matmul(Obj_orientation_t, final_obj_rotmat_temp, rel_pose_mat);   // deprecated

//                    raisim::quatToEuler(final_obj_pos_t_.segment(3,4), final_obj_euler); // final_obj_euler:
//                    rel_obj_pose_r_[0] = final_obj_euler.e()[0] - obj_pose_r_.e()[0];
//                    rel_obj_pose_r_[1] = final_obj_euler.e()[1] - obj_pose_r_.e()[1];
//                    rel_obj_pose_r_[2] = final_obj_euler.e()[2] - obj_pose_r_.e()[2];   // deprecated
//
                    raisim::quatToRotMat(final_obj_pos_t_.segment(3,4), final_obj_pose_mat); // final_obj_pose_mat: final object orientation in matrix
                    raisim::matmul(init_or_r_, final_obj_pose_mat, final_obj_wrist); // object final pose in wrist frame
                    raisim::matmul(init_or_r_, Obj_orientation_temp_t, obj_wrist); // object pose in wrist frame
                    raisim::transpose(obj_wrist, obj_wrist_trans);
                    raisim::matmul(final_obj_wrist, obj_wrist_trans, diff_obj_pose_mat);
                    raisim::RotmatToEuler(diff_obj_pose_mat, rel_obj_pose_r3);
                    rel_obj_pose_r_ = rel_obj_pose_r3.e(); // relative position of final orientation to current orientation, in wrist frame

                    raisim::quatToRotMat(final_obj_pos_b_.segment(3,4), final_obj_pose_mat); // final_obj_pose_mat: final object orientation in matrix
                    raisim::matmul(init_or_l_, final_obj_pose_mat, final_obj_wrist); // object final pose in wrist frame
                    raisim::matmul(init_or_l_, Obj_orientation_temp_b, obj_wrist); // object pose in wrist frame
                    raisim::transpose(obj_wrist, obj_wrist_trans);
                    raisim::matmul(final_obj_wrist, obj_wrist_trans, diff_obj_pose_mat);
                    raisim::RotmatToEuler(diff_obj_pose_mat, rel_obj_pose_r3);
                    rel_obj_pose_l_ = rel_obj_pose_r3.e(); // relative position of final orientation to current orientation, in wrist frame

                }

                /// Compute relative 3D position features for all hand joints
                Position_r[0] = Position_r[0] - Obj_Position_t[0];
                Position_r[1] = Position_r[1] - Obj_Position_t[1];
                Position_r[2] = Position_r[2] - Obj_Position_t[2]; // relative pose from body position to object position
                raisim::matvecmul(Obj_orientation_t, Position_r, Rel_fpos); // compute current relative pose in object coordinates

                obj_frame_diff_r[0] = final_ee_pos_r_[i * 3]- Rel_fpos[0];
                obj_frame_diff_r[1] = final_ee_pos_r_[i * 3 + 1] - Rel_fpos[1];
                obj_frame_diff_r[2] = final_ee_pos_r_[i * 3 + 2] - Rel_fpos[2]; // distance between target 3D positions and current 3D positions in object frame

                Position_l[0] = Position_l[0] - Obj_Position_b[0];
                Position_l[1] = Position_l[1] - Obj_Position_b[1];
                Position_l[2] = Position_l[2] - Obj_Position_b[2];
                raisim::matvecmul(Obj_orientation_b, Position_l, Rel_fpos);

                obj_frame_diff_l[0] = final_ee_pos_l_[i * 3] - Rel_fpos[0];
                obj_frame_diff_l[1] = final_ee_pos_l_[i * 3 + 1] - Rel_fpos[1];
                obj_frame_diff_l[2] = final_ee_pos_l_[i * 3 + 2] - Rel_fpos[2]; // distance between target 3D positions and current 3D positions in object frame

                //if(i == 0) {
                //    std::cout << "obj_frame_diff_l: \n" << obj_frame_diff_l << std::endl;
                //}
                //std::cout << "obj_frame_diff_l_[" << i << "] \n" << obj_frame_diff_l << std::endl;

                raisim::matvecmul(Obj_orientation_temp_t, obj_frame_diff_r, obj_frame_diff_w); // convert distances to world frame
                raisim::matvecmul(body_orientation_transpose_r, obj_frame_diff_w, obj_frame_diff_h_r); // convert distances to wrist frame
                raisim::matvecmul(Obj_orientation_temp_b, obj_frame_diff_l, obj_frame_diff_w); // convert distances to world frame
                raisim::matvecmul(body_orientation_transpose_l, obj_frame_diff_w, obj_frame_diff_h_l); // convert distances to wrist frame

                rel_body_pos_r_[i * 3] = obj_frame_diff_h_r[0];
                rel_body_pos_r_[i * 3 + 1] = obj_frame_diff_h_r[1];
                rel_body_pos_r_[i * 3 + 2] = obj_frame_diff_h_r[2];
                rel_body_pos_l_[i * 3] = obj_frame_diff_h_l[0];
                rel_body_pos_l_[i * 3 + 1] = obj_frame_diff_h_l[1];
                rel_body_pos_l_[i * 3 + 2] = obj_frame_diff_h_l[2];

                /// visualization of 3D joint position goals on the object
                if (visualizable_)
                {
                    raisim::matvecmul(Obj_orientation_temp_t, {final_ee_pos_r_[i * 3], final_ee_pos_r_[i * 3 + 1], final_ee_pos_r_[i * 3 + 2]}, sphere_pos);
                    vecadd(Obj_Position_t, sphere_pos);
                    spheres[2*i]->setPosition(sphere_pos.e());
                    raisim::matvecmul(Obj_orientation_temp_b, {final_ee_pos_l_[i * 3], final_ee_pos_l_[i * 3 + 1], final_ee_pos_l_[i * 3 + 2]}, sphere_pos);
                    vecadd(Obj_Position_b, sphere_pos);
                    spheres[2*i+1]->setPosition(sphere_pos.e());
                    if(i >= 16) {
                        spheres[2*i+1]->setColor(1, 0, 0, 1);
                    }
                    if(i == 0) {
                        spheres[2*i+1]->setColor(0, 0, 1, 1);
                    }
                }

            }

            /// compute current contacts of hand parts and the contact force
            auto& contact_list_obj = arctic->getContacts();

            for(auto& contact: mano_r_->getContacts()) {
                if (contact.skip() || contact.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                if (contact.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                if(right_kind_idx == 8){
                    if (contact_list_obj[contact.getPairContactIndexInPairObject()].getlocalBodyIndex() != top_id) continue;
                }
                contacts_r_[contactMapping_r_[contact.getlocalBodyIndex()]] = 1;
                impulses_r_[contactMapping_r_[contact.getlocalBodyIndex()]] = contact.getImpulse().norm();
            }

            /// compute relative target contact vector, i.e., which goal contacts are currently in contact
            rel_contacts_r_ = final_contact_array_r_.cwiseProduct(contacts_r_);
            rel_contacts_l_ = final_contact_array_l_.cwiseProduct(contacts_l_);

            //mirror observation for left hand
            //gc_l_obs_ = gc_l_;
            //gv_l_obs_ = gv_l_;
            //gc_l_obs_[0] = -gc_l_obs_[0];
            //gv_l_obs_[0] = -gv_l_obs_[0];
            //for(int i = 1; i < num_joint; i++) {
            //    gc_l_obs_[3*i+1] = -gc_l_obs_[3*i+1];
            //    gv_l_obs_[3*i+1] = -gv_l_obs_[3*i+1];
            //    gc_l_obs_[3*i+2] = -gc_l_obs_[3*i+2];
            //    gv_l_obs_[3*i+2] = -gv_l_obs_[3*i+2];
            //}
            //bodyLinearVel_l_obs_ = gv_l_obs_.segment(0,3);
            //bodyAngularVel_l_obs_ = gv_l_obs_.segment(3,3);
            //std::cout << "rel_body_pos_l_: \n" << rel_body_pos_l_ << std::endl;
            /// add all features to observation

//            std::cout<<"no bug here"<<std::endl;
////
//            raisim::Vec<3> obj_top_vel, obj_bottom_vel, obj_top_avel, obj_bottom_avel;
//            raisim::Vec<3> body_vel_r_obj, body_avel_r_obj, body_vel_l_obj, body_avel_l_obj;
            raisim::Vec<3> body_vel_r_w, body_avel_r_w, body_vel_l_w, body_avel_l_w;
            raisim::Vec<3> body_vel_r_o, body_avel_r_o, body_vel_l_o, body_avel_l_o;
//            auto left_id = mano_r_->getBodyIdx("left_index_fixed");
//            auto right_id = mano_l_->getBodyIdx("right_index_fixed");
//
//            mano_r_->getVelocity(right_id, body_vel_r_);
//            mano_r_->getAngularVelocity(right_id, body_avel_r_);
//            mano_l_->getVelocity(left_id, body_vel_l_);
//            mano_l_->getAngularVelocity(left_id, body_avel_l_);
//

            raisim::matvecmul(init_rot_r_, bodyLinearVel_r_, body_vel_r_w);
            raisim::matvecmul(init_rot_r_, bodyAngularVel_r_, body_avel_r_w);
            raisim::matvecmul(init_rot_l_, bodyLinearVel_l_, body_vel_l_w);
            raisim::matvecmul(init_rot_l_, bodyAngularVel_l_, body_avel_l_w);


            raisim::matvecmul(Obj_orientation_t, body_vel_r_w, body_vel_r_o);
            raisim::matvecmul(Obj_orientation_t, body_avel_r_w, body_avel_r_o);
            raisim::matvecmul(Obj_orientation_b, body_vel_l_w, body_vel_l_o);
            raisim::matvecmul(Obj_orientation_b, body_avel_l_w, body_avel_l_o);

            bodyLinearVel_r_obs_ = body_vel_r_o.e();
            bodyAngularVel_r_obs_ = body_avel_r_o.e();
            bodyLinearVel_l_obs_ = body_vel_l_o.e();
            bodyAngularVel_l_obs_ = body_avel_l_o.e();


//            bodyLinearVel_r_obs_ = Obj_orientation_t * body_vel_r_w.e();
//            bodyAngularVel_r_obs_ = Obj_orientation_t * body_avel_r_w.e();
//            bodyLinearVel_l_obs_ = Obj_orientation_b * body_vel_l_w.e();
//            bodyAngularVel_l_obs_ = Obj_orientation_b * body_avel_l_w.e();
//            std::cout<<"no bug here"<<std::endl;
//
//            arctic->getVelocity(top_id, obj_top_vel);
//            arctic->getAngularVelocity(top_id, obj_top_avel);
//            arctic->getVelocity(bottom_id, obj_bottom_vel);
//            arctic->getAngularVelocity(bottom_id, obj_bottom_avel);
//
//            raisim::matvecmul(Obj_orientation_t, body_vel_r_-obj_top_vel, body_vel_r_obj);
//            raisim::matvecmul(Obj_orientation_t, body_avel_r_-obj_top_avel, body_avel_r_obj);
//            raisim::matvecmul(Obj_orientation_b, body_vel_l_-obj_bottom_vel, body_vel_l_obj);
//            raisim::matvecmul(Obj_orientation_b, body_avel_l_-obj_bottom_avel, body_avel_l_obj); //vel in obj frame
//
//
//            bodyLinearVel_l_obs_.setZero();
//            bodyAngularVel_l_obs_.setZero();
//            bodyLinearVel_r_obs_.setZero();
//            bodyAngularVel_r_obs_.setZero();

//            bodyLinearVel_l_obs_[0] = body_vel_l_obj[0];
//            bodyLinearVel_l_obs_[1] = body_vel_l_obj[1];
//            bodyLinearVel_l_obs_[2] = body_vel_l_obj[2];
//
//            bodyLinearVel_r_obs_[0] = body_vel_r_obj[0];
//            bodyLinearVel_r_obs_[1] = body_vel_r_obj[1];
//            bodyLinearVel_r_obs_[2] = body_vel_r_obj[2];
//
//            bodyAngularVel_l_obs_[0] = body_avel_l_obj[0];
//            bodyAngularVel_l_obs_[1] = body_avel_l_obj[1];
//            bodyAngularVel_l_obs_[2] = body_avel_l_obj[2];
//
//            bodyAngularVel_r_obs_[0] = body_avel_r_obj[0];
//            bodyAngularVel_r_obs_[1] = body_avel_r_obj[1];
//            bodyAngularVel_r_obs_[2] = body_avel_r_obj[2];
//            temp_vel[0] = bodyLinearVel_r_[0]-obj_top_vel[0];
//            temp_vel[1] = bodyLinearVel_r_[1]-obj_top_vel[1];
//            temp_vel[2] = bodyLinearVel_r_[2]-obj_top_vel[2];
//            raisim::matvecmul(Obj_orientation_t, temp_vel, body_vel_r_obj);
//
//            temp_vel[0] = bodyAngularVel_r_[0]-obj_top_avel[0];
//            temp_vel[1] = bodyAngularVel_r_[1]-obj_top_avel[1];
//            temp_vel[2] = bodyAngularVel_r_[2]-obj_top_avel[2];
//            raisim::matvecmul(Obj_orientation_t, temp_vel, body_avel_r_obj);
//
//            temp_vel[0] = bodyLinearVel_l_[0]-obj_bottom_vel[0];
//            temp_vel[1] = bodyLinearVel_l_[1]-obj_bottom_vel[1];
//            temp_vel[2] = bodyLinearVel_l_[2]-obj_bottom_vel[2];
//            raisim::matvecmul(Obj_orientation_b, temp_vel, body_vel_l_obj);
//
//            temp_vel[0] = bodyAngularVel_l_[0]-obj_bottom_avel[0];
//            temp_vel[1] = bodyAngularVel_l_[1]-obj_bottom_avel[1];
//            temp_vel[2] = bodyAngularVel_l_[2]-obj_bottom_avel[2];
//            raisim::matvecmul(Obj_orientation_b, temp_vel, body_avel_l_obj); //vel in obj frame




            raisim::Mat<3,3> wrist_ori_r;
            mano_r_->getFrameOrientation(body_parts_r_[0], wrist_ori_r);
            Eigen::Vector3d rotation_axis_obj, rotation_axis_w, rotation_axis_h;
            Eigen::Vector3d arm_in_obj, arm_in_w, arm_in_wrist, wrist_in_obj, proj_in_obj;

            wrist_in_obj = Obj_orientation_temp_b.e().transpose()*rel_objpalm_r.e();
            proj_in_obj.setZero();
            proj_in_obj[2] = wrist_in_obj[2];
            arm_in_obj.setZero();
            arm_in_obj[0] = -wrist_in_obj[0];
            arm_in_obj[1] = -wrist_in_obj[1];
            arm_in_obj[2] = 0;
            arm_in_w = Obj_orientation_temp_b.e() * arm_in_obj;
            arm_in_wrist = wrist_ori_r.e().transpose() * arm_in_w;

            rotation_axis_obj.setZero();
            rotation_axis_obj[2] = 1;
            rotation_axis_w = Obj_orientation_temp_b.e() * rotation_axis_obj;
            rotation_axis_h = wrist_ori_r.e().transpose() * rotation_axis_w;

//            std::cout<<"contact info "<<final_contact_array_r_<<std::endl;

            obDouble_r_ << gc_r_.tail(gcDim_ - 6),      // (mirror) 51, generalized coordinate
                    bodyLinearVel_r_obs_,  // (mirror) 3, wrist linear velocity
                    bodyAngularVel_r_obs_, // (mirror) 3, wrist angular velocity
                    gv_r_.tail(gvDim_ - 6), // (mirror) 45, joint anglular velocity
                    rel_body_pos_r_,    //  (x mirror) 63, joint position relative to target position in wrist coord,
                    rel_pose_r_,  // (need to mirror the first 3 dimension) 48, angle between current pose and final pose, wrist pose in object coord
                    rel_objpalm_pos_r_, // (x mirror) 3, relative position between object and wrist in wrist coordinates
                    rel_obj_vel_t,  // (x mirror) 3, object velocity in wrist frame
                    rel_obj_qvel_t, // (yz mirror) 3, object angular velocity in wrist frame
                    final_contact_array_r_, // 16, goal contact array for all parts
                    impulses_r_, // 16, impulse array
                    rel_contacts_r_, // 16, relative target contact vector, i.e., which goal contacts are currently in contact
//                    rel_obj_pos_r_, //(x mirror) 3, object displacement from initial position in wrist coordinates
//                    rel_body_table_pos_r_, // (x mirror) 3, z-distance of hand to the table in wrist coordinates
//                    obj_pose_r_.e(), //(need to mirror)  3, object pose in wrist frame
                    arm_in_wrist,
                    arm_in_wrist.norm(),  //norm of arm
                    0,  // weight of bottom
                    obj_weight_, // weight of top
                    rotation_axis_h, // rotation axis in wrist frame
                    obj_angle_, // current object angle
                    obj_avel_, // object angular velocity
                    rel_obj_angle_; // relative object angle to the goal;
//                    wrist_pos_obj_r_.e(), // wrist position object frame (3)
//                    wrist_pose_obj_r_.e(); // wrist pose in object frame (3)
            obDouble_l_ << gc_l_.tail(gcDim_ - 6),
                    bodyLinearVel_l_obs_,
                    bodyAngularVel_l_obs_,
                    gv_l_.tail(gvDim_ - 6),
                    rel_body_pos_l_,
                    rel_pose_l_,
                    rel_objpalm_pos_l_,
                    rel_obj_vel_b,
                    rel_obj_qvel_b,
                    final_contact_array_l_,
                    impulses_l_,
                    rel_contacts_l_,
                    rel_obj_pos_l_,
                    rel_body_table_pos_l_,
                    obj_pose_l_.e();

//            std::cout<<obDouble_r_<<std::endl;
        }

        /// Set observation in wrapper to current observation
        void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
            ob_r = obDouble_r_.cast<float>();
            ob_l = obDouble_l_.cast<float>();
        }

        /// This function is only relevant for testing
        /// It increases the gain of the root control and lowers the table surface
        void set_rootguidance() final {

            motion_synthesis = true;

            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.head(3).setConstant(500);
            jointDgain.head(3).setConstant(0.1);
            jointPgain.tail(nJoints_).setConstant(50.0);
            jointDgain.tail(nJoints_).setConstant(0.2);

            mano_r_->setPdGains(jointPgain, jointDgain); // set PD gains for manos

        }

        void switch_root_guidance(bool is_on) {
            root_guided = is_on;
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            if(root_guided) {
                jointPgain.head(3).setConstant(50);
                jointDgain.head(3).setConstant(0.1);
                jointPgain.tail(nJoints_).setConstant(50.0);
                jointDgain.tail(nJoints_).setConstant(0.2);
            }
            else {
                jointPgain.head(3).setConstant(500);
                jointDgain.head(3).setConstant(0.1);
                jointPgain.tail(nJoints_).setConstant(50.0);
                jointDgain.tail(nJoints_).setConstant(0.2);
            }
            mano_r_->setPdGains(jointPgain, jointDgain); // set PD gains for manos

        }

        /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
        bool isTerminalState(float& terminalReward) final {

            if(obDouble_r_.hasNaN())
            {return true;}

            return false;
        }

    void set_gc_for_arctic(Eigen::VectorXd& gc_for_b, Eigen::VectorXd& gc_for_t, const Eigen::VectorXd& gc_arctic) {
            raisim::Vec<3> obj_goal_trans_b, obj_goal_trans_t;
            raisim::Mat<3,3> obj_goal_ori_mat_b, obj_goal_ori_mat_t, base_rot;
            raisim::Vec<4> obj_goal_ori_b, obj_goal_ori_t;
            Eigen::VectorXd obj_goal_angle;


            raisim::quatToRotMat(gc_arctic.segment(3,4), base_rot);
            obj_goal_angle.setZero(1);
            obj_goal_angle[0] = gc_arctic[7];

            arctic->setBasePos(gc_arctic.head(3));
            arctic->setBaseOrientation(base_rot);
            arctic->setGeneralizedCoordinate(obj_goal_angle);

            auto bottom_id = arctic->getBodyIdx("bottom");
            auto top_id = arctic->getBodyIdx("top");
            arctic->getPosition(bottom_id, obj_goal_trans_b);
            arctic->getPosition(top_id, obj_goal_trans_t);
            arctic->getOrientation(bottom_id, obj_goal_ori_mat_b);
            arctic->getOrientation(top_id, obj_goal_ori_mat_t);
            raisim::rotMatToQuat(obj_goal_ori_mat_b, obj_goal_ori_b);
            raisim::rotMatToQuat(obj_goal_ori_mat_t, obj_goal_ori_t);
            gc_for_b.head(3) = obj_goal_trans_b.e();
            gc_for_t.head(3) = obj_goal_trans_t.e();
            gc_for_b.segment(3,4) = obj_goal_ori_b.e();
            gc_for_t.segment(3,4) = obj_goal_ori_t.e();

    }

    void control_switch(int left, int right) {
        left_kind_idx = left;
        right_kind_idx = right;
    }


    private:
        int gcDim_, gvDim_, nJoints_;
        int gcDim_obj, gvDim_obj, nJoints_obj;
        bool visualizable_ = false;
        raisim::ArticulatedSystem* mano_;
        Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
        Eigen::VectorXd gc_l_, gv_l_, pTarget_l_, pTarget6_l_, vTarget_l_, gc_set_l_, gv_set_l_;
        Eigen::VectorXd gc_l_obs_, gv_l_obs_;
        Eigen::VectorXd obj_pos_init_, obj_pos_init_b_, obj_pos_init_t_;
        Eigen::VectorXd gen_force_r_, gen_force_l_, final_obj_pos_b_, final_obj_pos_t_, final_pose_r_, final_pose_l_, final_ee_pos_r_, final_ee_pos_l_, final_contact_array_r_, final_contact_array_l_, contact_body_idx_r_, contact_body_idx_l_, final_vertex_normals_r_, final_vertex_normals_l_;
        Eigen::VectorXd final_obj_angle_;
        double terminalRewardCoeff_ = -10.;
        double pose_reward_r_= 0.0, pose_reward_l_= 0.0;
        double pos_reward_r_ = 0.0, pos_reward_l_ = 0.0;
        double contact_reward_r_= 0.0, contact_reward_l_= 0.0;
        double obj_reward_r_ = 0.0, obj_reward_l_ = 0.0;
        double root_reward_r_ = 0.0, root_reward_l_ = 0.0;
        double contact_pos_reward_r_ = 0.0, contact_pos_reward_l_ = 0.0;
        double root_pos_reward_r_ = 0.0, root_pos_reward_l_ = 0.0;
        double root_pose_reward_r_ = 0.0, root_pose_reward_l_ = 0.0;
        double rel_obj_reward_r_ = 0.0, rel_obj_reward_l_ = 0.0;
        double body_vel_reward_r_ = 0.0, body_vel_reward_l_ = 0.0;
        double body_qvel_reward_r_ = 0.0, body_qvel_reward_l_ = 0.0;
        double obj_pose_reward_r_ = 0.0, obj_pose_reward_l_ = 0.0;
        double falling_reward_r = 0.0, falling_reward_l = 0.0;
        double k_obj = 50;
        double k_pose = 0.5;
        double k_ee = 1.0;
        double k_contact_r_ = 1.0, k_contact_l_ = 1.0;
        double ray_length = 0.05;
        double num_active_contacts_r_;
        double num_active_contacts_l_;
        double impulse_reward_r_ = 0.0, impulse_reward_l_ = 0.0;
        double obj_angle_reward_ = 0.0, obj_avel_reward_ = 0.0;
        double obj_weight_;
        int left_kind_idx = 0;
        int right_kind_idx = 0;
        Eigen::VectorXd joint_limit_high, joint_limit_low, actionMean_r_, actionMean_l_, actionStd_r_, actionStd_l_, obDouble_r_, obDouble_l_, rel_pose_r_, rel_pose_l_, finger_weights_, rel_obj_pos_r_, rel_obj_pos_l_, rel_objpalm_pos_r_, rel_objpalm_pos_l_, rel_body_pos_r_, rel_body_pos_l_, rel_contact_pos_r_, rel_contact_pos_l_, rel_contacts_r_, rel_contacts_l_, contacts_r_, contacts_l_, impulses_r_, impulses_l_, rel_obj_pose_r_, rel_obj_pose_l_;
        Eigen::Vector3d bodyLinearVel_r_, bodyLinearVel_l_, bodyLinearVel_l_obs_, bodyLinearVel_r_obs_, bodyAngularVel_r_, bodyAngularVel_l_, bodyAngularVel_l_obs_, bodyAngularVel_r_obs_, rel_obj_qvel_b, rel_obj_qvel_t, rel_obj_vel_b, rel_obj_vel_t, up_pose_r, up_pose_l, rel_body_table_pos_r_, rel_body_table_pos_l_;
        std::set<size_t> footIndices_;
        raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
        raisim::Cylinder *cylinder;
        raisim::Box *box_obj;
        raisim::Box *box;
        raisim::ArticulatedSystem *arctic, *mano_l_, *mano_r_;
        raisim::ArticulatedSystemVisual *arcticVisual;
        int rf_dim = 6;
        int num_obj = 4;
        int num_contacts = 16;
        int num_joint = 17;
        int num_bodyparts = 21;
        int obj_table_contact_ = 0;
        int root_guiding_counter_r_ = 0, root_guiding_counter_l_ = 0;
        int obj_idx_;
        bool root_guided=false;
        bool cylinder_mesh=false;
        bool box_obj_mesh=false;
        bool first_reset_=true;
        bool no_pose_state = false;
        bool nohierarchy = false;
        bool contact_pruned = false;
        bool motion_synthesis = false;
        float rewards_sum_[2];
        raisim::Vec<3> pose_goal_r, pos_goal_r, up_vec_r, up_gen_vec_r, obj_pose_r_, Position_r, Obj_Position_b, Rel_fpos, Obj_linvel_b, Obj_qvel_b, Fpos_world_r, palm_world_pos_r_, init_root_r_, init_obj_;
        raisim::Vec<3> pose_goal_l, pos_goal_l, up_vec_l, up_gen_vec_l, obj_pose_l_, Position_l, Obj_Position_t,           Obj_linvel_t, Obj_qvel_t, Fpos_world_l, palm_world_pos_l_, init_root_l_;
        raisim::Mat<3,3> Obj_orientation_b, Obj_orientation_t, Obj_orientation_temp_b, Obj_orientation_temp_t, Body_orientation_r, Body_orientation_l, init_or_r_, init_or_l_, root_pose_world_r_, root_pose_world_l_, init_rot_r_, init_rot_l_, init_obj_rot_, init_obj_or_;
        raisim::Mat<3,3> Obj_ori_test;
        raisim::Vec<4> obj_quat_b, obj_quat_t;
        Eigen::VectorXd obj_angle_, obj_avel_, rel_obj_angle_;
        raisim::Vec<3> wrist_pos_obj_r_, wrist_pose_obj_r_;
        std::vector<int> contact_idxs_;
        std::string body_parts_r_[21] = {"right_wrist_0rz",
                                         "right_index1_x", "right_index2_x", "right_index3_x", "right_index_tip",
                                         "right_middle1_x", "right_middle2_x", "right_middle3_x", "right_middle_tip",
                                         "right_pinky1_x", "right_pinky2_x", "right_pinky3_x", "right_pinky_tip",
                                         "right_ring1_x", "right_ring2_x", "right_ring3_x", "right_ring_tip",
                                         "right_thumb1_x", "right_thumb2_x", "right_thumb3_x", "right_thumb_tip",
        };
        std::string body_parts_l_[21] = {"left_wrist_0rz",
                                         "left_index1_x", "left_index2_x", "left_index3_x",
                                         "left_middle1_x", "left_middle2_x", "left_middle3_x",
                                         "left_pinky1_x", "left_pinky2_x", "left_pinky3_x",
                                         "left_ring1_x", "left_ring2_x", "left_ring3_x",
                                         "left_thumb1_x", "left_thumb2_x", "left_thumb3_x",
                                         "left_thumb_tip", "left_index_tip", "left_middle_tip", "left_ring_tip", "left_pinky_tip",
        };

        std::string contact_bodies_r_[16] = {"right_wrist_rz",
                                             "right_index1_z",  "right_index2_z",  "right_index3_z",
                                             "right_middle1_z", "right_middle2_z", "right_middle3_z",
                                             "right_pinky1_z",  "right_pinky2_z",  "right_pinky3_z",
                                             "right_ring1_z",   "right_ring2_z",   "right_ring3_z",
                                             "right_thumb1_z",  "right_thumb2_z",  "right_thumb3_z"
        };

        std::string contact_bodies_l_[16] = {"left_wrist_rz",
                                             "left_index1_z",  "left_index2_z",  "left_index3_z",
                                             "left_middle1_z", "left_middle2_z", "left_middle3_z",
                                             "left_pinky1_z",  "left_pinky2_z",  "left_pinky3_z",
                                             "left_ring1_z",   "left_ring2_z",   "left_ring3_z",
                                             "left_thumb1_z",  "left_thumb2_z",  "left_thumb3_z"
        };

        std::string ycb_objects_[21] = {"002_master_chef_can",
                                        "003_cracker_box",
                                        "004_sugar_box",
                                        "005_tomato_soup_can",
                                        "006_mustard_bottle",
                                        "007_tuna_fish_can",
                                        "008_pudding_box",
                                        "009_gelatin_box",
                                        "010_potted_meat_can",
                                        "011_banana",
                                        "019_pitcher_base",
                                        "021_bleach_cleanser",
                                        "024_bowl",
                                        "025_mug",
                                        "035_power_drill",
                                        "036_wood_block",
                                        "037_scissors",
                                        "040_large_marker",
                                        "051_large_clamp",
                                        "052_extra_large_clamp",
                                        "061_foam_brick"};
        raisim::PolyLine *lines[21];
        raisim::Visuals *spheres[42];
        raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
        std::map<int,int> contactMapping_r_;
        std::map<int,int> contactMapping_l_;
        std::string resourceDir_;
        std::vector<raisim::Vec<2>> joint_limits_;
        raisim::PolyLine *line;
        const double pi_ = 3.14159265358979323846;
        Eigen::MatrixXd Right2Left = Eigen::MatrixXd::Identity(3,3);
    };
}