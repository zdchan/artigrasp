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
            resourceDir_ = resourceDir;
            std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();
            mano_r_ = world_->addArticulatedSystem(resourceDir+"/mano_double/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
            mano_r_->setName("mano_r");

            std::string hand_model_l =  cfg["hand_model_l"].As<std::string>();
            mano_l_ = world_->addArticulatedSystem(resourceDir+"/mano_double/"+hand_model_l,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
            mano_l_->setName("mano_l");

            /// add table
            box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
            box->setPosition(1.25, 0, 0.25);
            box->setAppearance("0.0 0.0 0.0 0.0");

            /// set PD control mode
            mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            mano_l_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

            /// get actuation dimensions
            gcDim_ = mano_l_->getGeneralizedCoordinateDim();
            gvDim_ = mano_l_->getDOF();
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
            pTarget_r_bk.setZero(gcDim_); pTarget_l_bk.setZero(gcDim_);
            freezedTarget_l_.setZero(gcDim_);
            freezedTarget_r_.setZero(gcDim_);
            final_pose_r_.setZero(nJoints_), final_pose_r2_.setZero(nJoints_), final_obj_pos_b_.setZero(7), final_ee_pos_r_.setZero(num_bodyparts*3), final_ee_pos_r2_.setZero(num_bodyparts*3), final_vertex_normals_r_.setZero(num_contacts*3), contact_body_idx_r_.setZero(num_contacts), final_contact_array_r_.setZero(num_contacts), final_contact_array_r2_.setZero(num_contacts);
            final_pose_l_.setZero(nJoints_), final_obj_pos_t_.setZero(7), final_ee_pos_l_.setZero(num_bodyparts*3), final_vertex_normals_l_.setZero(num_contacts*3), contact_body_idx_l_.setZero(num_contacts), final_contact_array_l_.setZero(num_contacts);
            pregrasp_pose_r_.setZero(nJoints_); pregrasp_ee_pos_r_.setZero(3);
            pregrasp_pose_l_.setZero(nJoints_); pregrasp_ee_pos_l_.setZero(3);
            rel_pose_r_.setZero(nJoints_), rel_obj_pos_r_.setZero(3), rel_objpalm_pos_r_.setZero(3), rel_body_pos_r_.setZero(num_bodyparts*3), rel_contact_pos_r_.setZero(num_contacts*3), rel_obj_pose_r_.setZero(3), contacts_r_.setZero(num_contacts), rel_contacts_r_.setZero(num_contacts), impulses_r_.setZero(num_contacts);
            rel_pose_l_.setZero(nJoints_), rel_obj_pos_l_.setZero(3), rel_objpalm_pos_l_.setZero(3), rel_body_pos_l_.setZero(num_bodyparts*3), rel_contact_pos_l_.setZero(num_contacts*3), rel_obj_pose_l_.setZero(3), contacts_l_.setZero(num_contacts), rel_contacts_l_.setZero(num_contacts), impulses_l_.setZero(num_contacts);
            obj_body_pos_r_.setZero(num_bodyparts*3), obj_body_pos_l_.setZero(num_bodyparts*3);
            actionDim_ = gcDim_;
            actionMean_r_.setZero(actionDim_);  actionStd_r_.setOnes(actionDim_);
            actionMean_l_.setZero(actionDim_);  actionStd_l_.setOnes(actionDim_);
            joint_limit_high_r.setZero(actionDim_); joint_limit_low_r.setZero(actionDim_);
            joint_limit_high_l.setZero(actionDim_); joint_limit_low_l.setZero(actionDim_);
            Position_r.setZero(); Position_l.setZero(); Obj_Position_b.setZero(); Obj_Position_t.setZero(); Rel_fpos.setZero();
            obj_quat_b.setZero(); obj_quat_b[0] = 1.0;
            obj_quat_t.setZero(); obj_quat_t[0] = 1.0;
            Obj_linvel_b.setZero(); Obj_qvel_b.setZero();
            Obj_linvel_t.setZero(); Obj_qvel_t.setZero();
            rel_obj_vel_b.setZero(); rel_obj_qvel_b.setZero();
            rel_obj_vel_t.setZero(); rel_obj_qvel_t.setZero();
            bodyLinearVel_r_.setZero(); bodyAngularVel_r_.setZero();
            bodyLinearVel_l_.setZero(); bodyAngularVel_l_.setZero();
            stage_dim.setZero(); stage_pos.setZero();
            init_or_r_.setZero(); init_or_l_.setZero(); init_rot_r_.setZero(); init_rot_l_.setZero();
            init_root_r_.setZero(); init_root_l_.setZero(); init_obj_.setZero(); init_obj_rot_.setZero();
            obj_pose_r_.setZero(); obj_pose_l_.setZero(); obj_pos_init_.setZero(8); obj_pos_init_b_.setZero(7); obj_pos_init_t_.setZero(7);
            palm_world_pos_r_.setZero(); palm_world_pos_l_.setZero(); Fpos_world_r.setZero(); Fpos_world_l.setZero();
            final_obj_angle_.setZero(1); rel_obj_angle_.setZero(1); obj_angle_.setZero(1); obj_avel_.setZero(1);
            rel_obj_goal_pos_b_.setZero(3); rel_obj_goal_pos_t_.setZero(3);
            force_r_.setZero(3); force_l_.setZero(3); torque_r_.setZero(3); torque_l_.setZero(3);
            final_obj_pos_b_[3] = 1.0; final_obj_pos_t_[3] = 1.0;

            Obj_orientation_set_b.setZero();
            Obj_orientation_set_t.setZero();
            obj_pos_set_t.setZero();
            obj_pos_set_b.setZero();
            wrist_l_euler_world.setZero();
            init_rel_objpalm_pos_l.setZero();
            rel_obj_pos_b.setZero();
            initial_gc_l.setZero(gcDim_);
            vel_l.setZero();

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

            mano_l_->setPdGains(jointPgain, jointDgain);
            mano_l_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
            mano_l_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_r_ = 276;
            obDim_l_ = 276;
            gsDim_ = 110;

            obDouble_r_.setZero(obDim_r_);
            obDouble_l_.setZero(obDim_l_);
            global_state_.setZero(gsDim_);

            root_guided =  cfg["root_guided"].As<bool>();

            float finger_action_std = cfg["finger_action_std"].As<float>();
            float rot_action_std = cfg["rot_action_std"].As<float>();

            /// retrieve joint limits from model
            joint_limits_l_ = mano_l_->getJointLimits();
            joint_limits_r_ = mano_r_->getJointLimits();

            // set joint torque limits
            for(int i=0; i < int(gcDim_); i++){
                actionMean_r_[i] = (joint_limits_r_[i][1]+joint_limits_r_[i][0])/2.0;
                actionMean_l_[i] = (joint_limits_l_[i][1]+joint_limits_l_[i][0])/2.0;
                joint_limit_low_r[i] = joint_limits_r_[i][0];
                joint_limit_high_r[i] = joint_limits_r_[i][1];
                joint_limit_low_l[i] = joint_limits_l_[i][0];
                joint_limit_high_l[i] = joint_limits_l_[i][1];
            }

            /// set actuation parameters
            if (root_guided){
                actionStd_l_.setConstant(finger_action_std);
                actionStd_l_.head(3).setConstant(0.001);
                actionStd_l_.segment(3,3).setConstant(rot_action_std);
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.001);
                actionStd_r_.segment(3,3).setConstant(rot_action_std);
            }
            else{
                actionStd_l_.setConstant(finger_action_std);
                actionStd_l_.head(3).setConstant(0.01);
                actionStd_l_.segment(3,3).setConstant(0.01);
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.01);
                actionStd_r_.segment(3,3).setConstant(0.01);
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

        // stage_pos: 3, stage_dim: 3
        void add_stage(const Eigen::Ref<EigenVec>& stage_dim_input,
                       const Eigen::Ref<EigenVec>& stage_pos_input) {
            stage = static_cast<raisim::Box*>(world_->addBox(stage_dim_input[0], stage_dim_input[1], stage_dim_input[2], 10, "", raisim::COLLISION(1)));
            stage->setPosition(stage_pos_input[0], stage_pos_input[1], stage_pos_input[2]);
            stage->setAppearance("blue");
            stage_dim = stage_dim_input.cast<double>();
            stage_pos = stage_pos_input.cast<double>();
        }

        /// This function loads the object into the environment
        void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {  }

        void load_articulated(const std::string& obj_model){
            arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/arctic/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
            arctic->setName("arctic");

            gcDim_obj = arctic->getGeneralizedCoordinateDim();
            gvDim_obj = arctic->getDOF();

            gc_obj_.setZero(gcDim_obj);
            gv_obj_.setZero(gvDim_obj);

            Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
            gen_coord[4] = 1;
            arctic->setGeneralizedCoordinate(gen_coord);

            arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
            auto top_id = arctic->getBodyIdx("top");
            auto bottom_id = arctic->getBodyIdx("bottom");
            obj_weight_t_ = arctic->getMass(top_id);
            obj_weight_b_ = arctic->getMass(bottom_id);
            obj_weight_ = obj_weight_t_ + obj_weight_b_;

            Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
            objPgain.setZero();
            objDgain.setZero();
            arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            arctic->setPdGains(objPgain, objDgain);
            arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));
        }

        /// Resets the object and hand to its initial pose
        void reset() final {

            if (first_reset_)
            {
                first_reset_=false;
            }
            else{
                /// all settings to initial state configuration
                Eigen::VectorXd obj_goal_angle;
                obj_goal_angle.setZero(1);
                obj_goal_angle[0] = obj_pos_init_[7];
                actionMean_r_.setZero();
                actionMean_l_.setZero();

                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                mano_l_->setBasePos(init_root_l_);
                mano_l_->setBaseOrientation(init_rot_l_);
                mano_l_->setState(gc_set_l_, gv_set_l_);

                arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));

                box->clearExternalForcesAndTorques();
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                stage->clearExternalForcesAndTorques();
                stage->setPosition(stage_pos[0], stage_pos[1], stage_pos[2]);
                stage->setOrientation(1,0,0,0);
                stage->setVelocity(0,0,0,0,0,0);

                updateObservation2();

                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_l_->setGeneralizedForce(gen_force);
                mano_r_->setGeneralizedForce(gen_force);

                gen_force.setZero(gcDim_obj);
                arctic->setGeneralizedForce(gen_force);
            }

        }

        void reset_right_hand(const Eigen::Ref<EigenVec>& init_state_r,
                              const Eigen::Ref<EigenVec>& init_state_l,
                              const Eigen::Ref<EigenVec>& obj_pose) final {}

        /// Resets the state to a user defined input
        // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
        void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                         const Eigen::Ref<EigenVec>& init_state_l,
                         const Eigen::Ref<EigenVec>& init_vel_r,
                         const Eigen::Ref<EigenVec>& init_vel_l,
                         const Eigen::Ref<EigenVec>& obj_pose) final {
            /// reset gains (only required in case for inference)
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.head(3).setConstant(50);
            jointDgain.head(3).setConstant(0.1);
            jointPgain.tail(nJoints_).setConstant(50.0);
            jointDgain.tail(nJoints_).setConstant(0.2);
            mano_l_->setPdGains(jointPgain, jointDgain);
            mano_r_->setPdGains(jointPgain, jointDgain);

            grasp_num_l = 0;
            grasp_num_r = 0;

            Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
            objPgain.setZero();
            objDgain.setZero();
            arctic->setPdGains(objPgain, objDgain);

            Eigen::VectorXd gen_force;
            gen_force.setZero(gcDim_);
            mano_l_->setGeneralizedForce(gen_force);
            mano_r_->setGeneralizedForce(gen_force);

            arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

            /// reset box position (only required in case for inference)
            box->setPosition(1.25, 0, 0.25);
            box->setOrientation(1,0,0,0);
            box->setVelocity(0,0,0,0,0,0);

            stage->setPosition(stage_pos[0], stage_pos[1], stage_pos[2]);
            stage->setOrientation(1,0,0,0);
            stage->setVelocity(0,0,0,0,0,0);

            /// set initial hand pose (45 DoF) and velocity (45 DoF)
            gc_set_r_.head(6).setZero();
            gc_set_r_.tail(nJoints_-3) = init_state_r.tail(nJoints_-3).cast<double>(); //.cast<double>();
            gv_set_r_ = init_vel_r.cast<double>(); //.cast<double>();
            gc_set_l_.head(6).setZero();
            gc_set_l_.tail(nJoints_-3) = init_state_l.tail(nJoints_-3).cast<double>(); //.cast<double>();
            gv_set_l_ = init_vel_l.cast<double>(); //.cast<double>();
            mano_r_->setState(gc_set_r_, gv_set_r_);
            mano_l_->setState(gc_set_l_, gv_set_l_);

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
            Eigen::VectorXd arcticVel;
            arcticVel.setZero(arcticVelDim);

            raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
            Eigen::VectorXd obj_pose_cast;
            obj_pose_cast = obj_pose.cast<double>();
            arctic->setState(obj_pose_cast, arcticVel);
            mano_r_->setBasePos(init_root_r_);
            mano_r_->setBaseOrientation(init_rot_r_);
            mano_r_->setState(gc_set_r_, gv_set_r_);
            mano_l_->setBasePos(init_root_l_);
            mano_l_->setBaseOrientation(init_rot_l_);
            mano_l_->setState(gc_set_l_, gv_set_l_);

            /// set initial object pose
            obj_pos_init_ = obj_pose.cast<double>(); // 8 dof

            set_gc_for_arctic(obj_pos_init_b_, obj_pos_init_t_, obj_pos_init_);

            /// Set action mean to initial pose (first 6DoF since start at 0)

            actionMean_r_.setZero();
            actionMean_r_.tail(nJoints_-3) = gc_set_r_.tail(nJoints_-3);
            actionMean_l_.setZero();
            actionMean_l_.tail(nJoints_-3) = gc_set_l_.tail(nJoints_-3);

            motion_synthesis = false;
            root_guiding_counter_r_ = 0;
            root_guiding_counter_l_ = 0;

            gen_force.setZero(gcDim_);
            mano_l_->setGeneralizedForce(gen_force);
            mano_r_->setGeneralizedForce(gen_force);

            auto top_id = arctic->getBodyIdx("top");
            auto bottom_id = arctic->getBodyIdx("bottom");
            obj_weight_t_ = arctic->getMass(top_id);
            obj_weight_b_ = arctic->getMass(bottom_id);
            obj_weight_ = obj_weight_t_ + obj_weight_b_;

            updateObservation2();

            freezedTarget_l_ = gc_set_l_;

            raisim::Vec<3> temp_pos;
            arctic->getPosition(bottom_id, obj_pos_set_b); //Obj_Position: object base position at this time
            arctic->getOrientation(bottom_id, Obj_orientation_set_b); //Obj_orientation_temp: object bottom orientation at this time, in matrix

            mano_l_->getFramePosition(body_parts_l_[0], Position_l); // Position: body position of a body in mano
            mano_l_->getFrameOrientation(body_parts_l_[0], Body_orientation_l); // Body_orientation:

            temp_pos[0] = Position_l[0]-obj_pos_set_b[0];
            temp_pos[1] = Position_l[1]-obj_pos_set_b[1];
            temp_pos[2] = Position_l[2]-obj_pos_set_b[2];

            init_rel_objpalm_pos_l = Body_orientation_l.e().transpose()*temp_pos.e();

            raisim::matmul(Body_orientation_l.e().transpose(), Obj_orientation_set_b, init_obj_ori_wrist_l);

            Eigen::VectorXd gc_l_temp, gv_l_temp;
            mano_l_->getState(gc_l_temp, gv_l_temp);
            initial_gc_l = gc_l_temp;
        }

        void set_goals_r(const Eigen::Ref<EigenVec>& obj_goal_pos_r,
                         const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                         const Eigen::Ref<EigenVec>& goal_pose_r,
                         const Eigen::Ref<EigenVec>& goal_qpos_r)
        {
            raisim::Vec<4> quat_goal_hand_w, quat_goal_hand_r, quat_obj_init_t, quat_obj_init_b;
            raisim::Vec<3> euler_goal_pose;
            raisim::Mat<3,3> rotm_goal_hand_r, mat_temp, mat_rot_left, rotm_goal_t, rotm_goal_tran_t;

            Eigen::VectorXd final_qpos_r = goal_qpos_r.cast<double>();
            Eigen::VectorXd obj_goal_pos_r_cast;
            obj_goal_pos_r_cast = obj_goal_pos_r.cast<double>();
            set_gc_for_arctic(final_obj_pos_b_, final_obj_pos_t_, obj_goal_pos_r_cast);
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
            raisim::quatInvQuatMul(quat_obj_init_t, quat_goal_hand_w, quat_goal_hand_r);
            raisim::quatToRotMat(quat_goal_hand_r, rotm_goal_hand_r);
            raisim::RotmatToEuler(rotm_goal_hand_r, euler_goal_pose);

            final_pose_r_ = goal_pose_r.cast<double>();
            final_pose_r_.head(3) = euler_goal_pose.e().cast<double>(); // change the orientation of hand final pose to the object frame

            mano_r_->setBasePos(goal_qpos_r.head(3));
            mano_r_->setBaseOrientation(root_pose_world_r_);
            final_qpos_r.head(6).setZero();
            mano_r_->setGeneralizedCoordinate(final_qpos_r);

            raisim::Vec<3> ee_goal_exact_r;
            /// Compute and convert hand 3D joint positions into object relative frame
            for(int i = 0; i < num_bodyparts; i++){
                mano_r_->getFramePosition(body_parts_r_[i], ee_goal_exact_r);
                Position_r[0] = ee_goal_exact_r[0] - final_obj_pos_t_[0];
                Position_r[1] = ee_goal_exact_r[1] - final_obj_pos_t_[1];
                Position_r[2] = ee_goal_exact_r[2] - final_obj_pos_t_[2];

                raisim::matvecmul(Obj_orientation_t, Position_r, Rel_fpos);

                final_ee_pos_r_[i*3] = Rel_fpos[0];
                final_ee_pos_r_[i*3+1] = Rel_fpos[1];
                final_ee_pos_r_[i*3+2] = Rel_fpos[2];
            }
        }

        void set_goals_r2(const Eigen::Ref<EigenVec>& obj_goal_pos_r,
                          const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                          const Eigen::Ref<EigenVec>& goal_pose_r,
                          const Eigen::Ref<EigenVec>& goal_qpos_r,
                          const Eigen::Ref<EigenVec>& goal_contacts_r)
        {
            raisim::Vec<4> quat_goal_hand_w, quat_goal_hand_r, quat_obj_init_t, quat_obj_init_b;
            raisim::Vec<3> euler_goal_pose;
            raisim::Mat<3,3> rotm_goal_hand_r, mat_temp, mat_rot_left, rotm_goal_t, rotm_goal_tran_t;

            Eigen::VectorXd final_qpos_r = goal_qpos_r.cast<double>();
            Eigen::VectorXd obj_goal_pos_r_cast;
            obj_goal_pos_r_cast = obj_goal_pos_r.cast<double>();

            set_gc_for_arctic(final_obj_pos_b_, final_obj_pos_t_, obj_goal_pos_r_cast);

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
            raisim::quatInvQuatMul(quat_obj_init_t, quat_goal_hand_w, quat_goal_hand_r);
            raisim::quatToRotMat(quat_goal_hand_r, rotm_goal_hand_r);
            raisim::RotmatToEuler(rotm_goal_hand_r, euler_goal_pose);

            final_pose_r2_ = goal_pose_r.cast<double>();
            final_pose_r2_.head(3) = euler_goal_pose.e(); // change the orientation of hand final pose to the object frame

            mano_r_->setBasePos(goal_qpos_r.head(3));
            mano_r_->setBaseOrientation(root_pose_world_r_);
            final_qpos_r.head(6).setZero();
            mano_r_->setGeneralizedCoordinate(final_qpos_r);

            raisim::Vec<3> ee_goal_exact_r;
            /// Compute and convert hand 3D joint positions into object relative frame
            for(int i = 0; i < num_bodyparts; i++){
                mano_r_->getFramePosition(body_parts_r_[i], ee_goal_exact_r);
                Position_r[0] = ee_goal_exact_r[0] - final_obj_pos_t_[0];
                Position_r[1] = ee_goal_exact_r[1] - final_obj_pos_t_[1];
                Position_r[2] = ee_goal_exact_r[2] - final_obj_pos_t_[2];

                raisim::matvecmul(Obj_orientation_t, Position_r, Rel_fpos);

                final_ee_pos_r2_[i*3] = Rel_fpos[0];
                final_ee_pos_r2_[i*3+1] = Rel_fpos[1];
                final_ee_pos_r2_[i*3+2] = Rel_fpos[2];
            }
            num_active_contacts_r2_ = float(goal_contacts_r.sum());
            final_contact_array_r2_ = goal_contacts_r.cast<double>();
            k_contact_r2_ = 1.0 / num_active_contacts_r2_;
        }


        /// This function is used to set user specified goals that the policy is conditioned on
        /// Obj_pos: Object goal state in global frame (8), 3 translation + 4 quaternion rotation + 1 joint angle
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
            raisim::Vec<4> quat_goal_hand_w, quat_goal_hand_r, quat_obj_init_t, quat_obj_init_b;
            raisim::Vec<3> euler_goal_pose;
            raisim::Mat<3,3> rotm_goal_hand_r, rotm_goal_hand_l, mat_temp, mat_rot_left, rotm_goal_b, rotm_goal_tran_b;

            final_obj_angle_[0] = obj_goal_angle[0];


            Eigen::VectorXd final_qpos_r = goal_qpos_r.cast<double>();
            Eigen::VectorXd final_qpos_l = goal_qpos_l.cast<double>();
            Eigen::VectorXd obj_goal_pos_cast;
            obj_goal_pos_cast = obj_goal_pos.cast<double>();
            set_gc_for_arctic(final_obj_pos_b_, final_obj_pos_t_, obj_goal_pos_cast);
            /// set final object pose

            /// convert object and handpose pose to rotation matrix format
            raisim::quatToRotMat(final_obj_pos_b_.tail(4), Obj_orientation_temp_b); // Obj_orientation_temp: orientation of object goal pose, in matrix
            raisim::quatToRotMat(final_obj_pos_t_.tail(4), Obj_orientation_temp_t);

            quat_obj_init_b = final_obj_pos_b_.tail(4).cast<double>(); //quat_obj_init: orientation of object goal pose, in quaternion
            quat_obj_init_t = final_obj_pos_t_.tail(4).cast<double>();

            raisim::transpose(Obj_orientation_temp_b, Obj_orientation_b);
            raisim::transpose(Obj_orientation_temp_t, Obj_orientation_t); // Obj_orientation: relative orientation of world coordinate to object

            raisim::eulerToQuat(goal_pose_l.head(3), quat_goal_hand_w);
            raisim::quatToRotMat(quat_goal_hand_w, root_pose_world_l_);

            /// Compute and set object relative goal hand pose
            raisim::quatToRotMat(quat_obj_init_b, rotm_goal_b);
            raisim::transpose(rotm_goal_b, rotm_goal_tran_b);
            raisim::matmul(rotm_goal_tran_b, root_pose_world_l_, rotm_goal_hand_l);
            raisim::RotmatToEuler(rotm_goal_hand_l, euler_goal_pose);

            final_pose_l_ = goal_pose_l.cast<double>();
            final_pose_l_.head(3) = euler_goal_pose.e().cast<double>();

            mano_l_->setBasePos(goal_qpos_l.head(3));
            mano_l_->setBaseOrientation(root_pose_world_l_);
            final_qpos_l.head(6).setZero();
            mano_l_->setGeneralizedCoordinate(final_qpos_l);

            raisim::Vec<3> ee_goal_exact_l;
            /// Compute and convert hand 3D joint positions into object relative frame
            for(int i = 0; i < num_bodyparts; i++){
                mano_l_->getFramePosition(body_parts_l_[i], ee_goal_exact_l);
                Position_l[0] = ee_goal_exact_l[0] - final_obj_pos_b_[0];
                Position_l[1] = ee_goal_exact_l[1] - final_obj_pos_b_[1];
                Position_l[2] = ee_goal_exact_l[2] - final_obj_pos_b_[2];

                raisim::matvecmul(Obj_orientation_b, Position_l, Rel_fpos);

                final_ee_pos_l_[i*3] = Rel_fpos[0];
                final_ee_pos_l_[i*3+1] = Rel_fpos[1];
                final_ee_pos_l_[i*3+2] = Rel_fpos[2];
            }

            /// Intialize and set goal contact array
            num_active_contacts_r_ = float(goal_contacts_r.sum());
            num_active_contacts_l_ = float(goal_contacts_l.sum());
            final_contact_array_r_ = goal_contacts_r.cast<double>();
            final_contact_array_l_ = goal_contacts_l.cast<double>();

            for(int i = 0; i < num_contacts ;i++){
                contact_body_idx_l_[i] =  mano_l_->getBodyIdx(contact_bodies_l_[i]);
                contactMapping_l_.insert(std::pair<int,int>(int(mano_l_->getBodyIdx(contact_bodies_l_[i])),i));
            }
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
            obj_goal_pos_true = obj_goal_pos.cast<double>();
            obj_goal_pos_true[7] = obj_goal_angle[0];
            set_gc_for_arctic(final_obj_pos_b_, final_obj_pos_t_, obj_goal_pos_true);
        }

        void set_obj_goal(const Eigen::Ref<EigenVec>& obj_goal_angle,
                          const Eigen::Ref<EigenVec>& obj_goal_pos)
        {
            Eigen::VectorXd obj_goal_pos_true;
            obj_goal_pos_true = obj_goal_pos.cast<double>();
            obj_goal_pos_true[7] = obj_goal_angle[0];
            set_gc_for_arctic(final_obj_pos_b_, final_obj_pos_t_, obj_goal_pos_true);
            if (visualizable_) {
                arcticVisual->setGeneralizedCoordinate(obj_goal_pos_true);
            }
        }

        /// This function takes an environment step given an action (51DoF) input
        float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {}

        float* step2(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {
            stage_flag = false;
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
                raisim::quatToRotMat(final_obj_pos_t_.tail(4),rotmat_final_obj_pos);
                raisim::transpose(rotmat_final_obj_pos, rotmat_final_obj_pos_trans);
                raisim::matvecmul(rotmat_final_obj_pos, final_ee_pos_r2_.head(3), Fpos_world_r);
                raisim::vecadd(obj_pos_raisim_t, Fpos_world_r); //Obj_Position before
                if (visualizable_)
                {
                    spheres[0]->setPosition(pos_goal_r.e());
                }


                pos_goal_r = action_r.head(3);
                raisim::vecsub(pos_goal_r, init_root_r_, act_pos_r);
                raisim::matvecmul(init_or_r_,obj_pos_raisim_t,act_or_pose_r);

                actionMean_r_.head(3) = (act_or_pose_r.e())*std::min(1.0,(0.001*root_guiding_counter_r_));
                actionMean_r_.head(3) += gc_r_.head(3);

                raisim::Mat<3, 3> rotmat_gc, rotmat_gc_trans, rotmat_obj_pose, posegoal_rotmat;
                raisim::Vec<4> quat_gc, quat_pg, quat_diff;
                raisim::Vec<3> euler_obj_pose_goal, euler_obj_pose_curr, diff_obj_pose, rot_goal_euler;
                pose_goal_r = action_r.segment(3,3);
                raisim::eulerToQuat(pose_goal_r,quat_pg);
                raisim::quatToRotMat(final_obj_pos_t_.tail(4),rotmat_gc);
                raisim::matmul(init_or_r_, rotmat_gc, rotmat_gc_trans);
                raisim::RotmatToEuler(rotmat_gc_trans, euler_obj_pose_goal);

                raisim::matmul(init_or_r_, Obj_orientation_temp_t, rotmat_obj_pose);
                raisim::RotmatToEuler(rotmat_obj_pose, euler_obj_pose_curr);

                raisim::vecsub(euler_obj_pose_goal, euler_obj_pose_curr, diff_obj_pose);


                for (int i = 0; i < 3; i++) {
                    if (diff_obj_pose[i] > pi_)
                        diff_obj_pose[i] -= 2*pi_;
                    else if (diff_obj_pose[i] < -pi_)
                        diff_obj_pose[i] += 2*pi_;
                }

                actionMean_r_.segment(3,3) = rel_obj_pose_r_ * std::min(1.0,(0.0005*root_guiding_counter_r_));
                actionMean_r_.segment(3,3) += gc_r_.segment(3,3);
                root_guiding_counter_r_ += 1;

            }

            if (motion_synthesis_l)
            {
                raisim::quatToRotMat(final_obj_pos_b_.tail(4),rotmat_final_obj_pos);
                raisim::transpose(rotmat_final_obj_pos, rotmat_final_obj_pos_trans);
                raisim::matvecmul(rotmat_final_obj_pos, final_ee_pos_l_.head(3), Fpos_world_l);
                raisim::vecadd(obj_pos_raisim_b, Fpos_world_l); //Obj_Position before
                if (visualizable_)
                {
                    spheres[1]->setPosition(pos_goal_l.e());
                }


                pos_goal_l = action_l.head(3);
                raisim::vecsub(pos_goal_l, init_root_l_, act_pos_l);
                raisim::matvecmul(init_or_l_,obj_pos_raisim_b,act_or_pose_l);

                actionMean_l_.head(3) = (act_or_pose_l.e())*std::min(1.0,(0.001*root_guiding_counter_l_));
                actionMean_l_.head(3) += gc_l_.head(3);

                raisim::Mat<3, 3> rotmat_gc, rotmat_gc_trans, rotmat_obj_pose, posegoal_rotmat;
                raisim::Vec<4> quat_gc, quat_pg, quat_diff;
                raisim::Vec<3> euler_obj_pose_goal, euler_obj_pose_curr, diff_obj_pose, rot_goal_euler;
                pose_goal_l = action_l.segment(3,3);
                raisim::eulerToQuat(pose_goal_l,quat_pg);
                raisim::quatToRotMat(final_obj_pos_b_.tail(4),rotmat_gc);
                raisim::matmul(init_or_l_, rotmat_gc, rotmat_gc_trans);
                raisim::RotmatToEuler(rotmat_gc_trans, euler_obj_pose_goal);

                raisim::matmul(init_or_l_, Obj_orientation_temp_b, rotmat_obj_pose);
                raisim::RotmatToEuler(rotmat_obj_pose, euler_obj_pose_curr);

                raisim::vecsub(euler_obj_pose_goal, euler_obj_pose_curr, diff_obj_pose);


                for (int i = 0; i < 3; i++) {
                    if (diff_obj_pose[i] > pi_)
                        diff_obj_pose[i] -= 2*pi_;
                    else if (diff_obj_pose[i] < -pi_)
                        diff_obj_pose[i] += 2*pi_;
                }

                actionMean_l_.segment(3,3) = rel_obj_pose_l_ * std::min(1.0,(0.0005*root_guiding_counter_l_));
                actionMean_l_.segment(3,3) += gc_l_.segment(3,3);
                root_guiding_counter_l_ += 1;

            }

            /// The following applies the wrist guidance technique (compare with paper)
            if (root_guided_l){
                if (left_kind_idx == 9){
                    grasp_num_l += 1;
                }
                /// Retrieve current object pose
                auto bottom_id = arctic->getBodyIdx("bottom");
                auto top_id = arctic->getBodyIdx("top");
                arctic->getPosition(bottom_id, Obj_Position_b); //Obj_Position: object base position at this time
                arctic->getPosition(top_id, Obj_Position_t);
                arctic->getOrientation(bottom_id, Obj_orientation_temp_b); //Obj_orientation_temp: object bottom orientation at this time, in matrix
                arctic->getOrientation(top_id, Obj_orientation_temp_t);
                raisim::rotMatToQuat(Obj_orientation_temp_b, obj_quat_b); //obj_quat: object bottom orientation at this time, in quat
                raisim::rotMatToQuat(Obj_orientation_temp_t, obj_quat_t);

                /// Convert final root hand translation back from (current) object into world frame
                raisim::matvecmul(Obj_orientation_temp_b, final_ee_pos_l_.head(3), Fpos_world_l);
                raisim::vecadd(Obj_Position_b, Fpos_world_l);

                raisim::vecsub(Fpos_world_l, init_root_l_, act_pos_l); // compute distance of current root to initial root in world frame
                raisim::matvecmul(init_or_l_, act_pos_l, act_or_pose_l); // rotate the world coordinate into hand's origin frame (from the start of the episode)

                actionMean_l_.head(3) = act_or_pose_l.e();
            }

            if (root_guided_r){
                if (right_kind_idx == 9){
                    grasp_num_r += 1;
                }
                /// Retrieve current object pose
                auto bottom_id = arctic->getBodyIdx("bottom");
                auto top_id = arctic->getBodyIdx("top");
                arctic->getPosition(bottom_id, Obj_Position_b); //Obj_Position: object base position at this time
                arctic->getPosition(top_id, Obj_Position_t);
                arctic->getOrientation(bottom_id, Obj_orientation_temp_b); //Obj_orientation_temp: object bottom orientation at this time, in matrix
                arctic->getOrientation(top_id, Obj_orientation_temp_t);
                raisim::rotMatToQuat(Obj_orientation_temp_b, obj_quat_b); //obj_quat: object bottom orientation at this time, in quat
                raisim::rotMatToQuat(Obj_orientation_temp_t, obj_quat_t);

                /// Convert final root hand translation back from (current) object into world frame

                raisim::matvecmul(Obj_orientation_temp_t, final_ee_pos_r2_.head(3), Fpos_world_r);
                raisim::vecadd(Obj_Position_t, Fpos_world_r);

                raisim::vecsub(Fpos_world_r, init_root_r_, act_pos_r); // compute distance of current root to initial root in world frame
                raisim::matvecmul(init_or_r_, act_pos_r, act_or_pose_r); // rotate the world coordinate into hand's origin frame (from the start of the episode)

                actionMean_r_.head(3) = act_or_pose_r.e();

            }

            if (root_keep_r){
                pTarget_r_ = gc_set_r_;
            }
            else if (motion_synthesis){
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_.head(6).setZero(); //add wrist bias (first 3DOF) and last pose (48DoF)
                pTarget_r_ += actionMean_r_;
            }
            else{
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (48DoF)

                pTarget_r_bk = pTarget_r_;
            }

            if (root_keep_l){
                actionMean_l_ = gc_set_l_;
            }
            else if (motion_synthesis_l){
                pTarget_l_ = action_l.cast<double>();
                pTarget_l_ = pTarget_l_.cwiseProduct(actionStd_l_); //residual action * scaling
                pTarget_l_.head(6).setZero(); //add wrist bias (first 3DOF) and last pose (48DoF)
                pTarget_l_ += actionMean_l_; //add wrist bias (first 3DOF) and last pose (48DoF)
            }
            else{
                pTarget_l_ = action_l.cast<double>();
                pTarget_l_ = pTarget_l_.cwiseProduct(actionStd_l_); //residual action * scaling
                pTarget_l_ += actionMean_l_; //add wrist bias (first 3DOF) and last pose (48DoF)

                pTarget_l_bk = pTarget_l_;
            }

            /// Clip targets to limits
            Eigen::VectorXd pTarget_clipped_r, pTarget_clipped_l;
            pTarget_clipped_r.setZero(gcDim_);
            pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low_r).cwiseMin(joint_limit_high_r);
            pTarget_clipped_l.setZero(gcDim_);
            pTarget_clipped_l = pTarget_l_.cwiseMax(joint_limit_low_l).cwiseMin(joint_limit_high_l);

            /// Set PD targets (velocity zero)
            mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);
            mano_l_->setPdTarget(pTarget_clipped_l, vTarget_l_);
            //ptarget_clipped = action_l;


            /// Apply N control steps
            for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();
                if(server_) server_->unlockVisualizationServerMutex();
            }

            /// update observation and set new mean to the latest pose
            updateObservation2();
            actionMean_r_ = gc_r_;
            actionMean_l_ = gc_l_;

            /// Compute general reward terms
            pose_reward_r_ = -(rel_pose_r_).norm();
            pos_reward_r_ = -rel_body_pos_r_.cwiseProduct(finger_weights_).squaredNorm();

            pose_reward_l_ = -(rel_pose_l_).norm();
            pos_reward_l_ = -rel_body_pos_l_.cwiseProduct(finger_weights_).squaredNorm();
            obj_angle_reward_ = - rel_obj_angle_.norm();

            /// Compute regularization rewards
            rel_obj_vel_reward_r_ = rel_obj_vel_t.squaredNorm();
            body_vel_reward_r_ = bodyLinearVel_r_.squaredNorm();
            body_qvel_reward_r_ = bodyAngularVel_r_.squaredNorm();
            contact_reward_r_ = k_contact_r2_*(rel_contacts_r_.sum());
            impulse_reward_r_ = ((final_contact_array_r2_.cwiseProduct(impulses_r_)).sum());

            rel_obj_vel_reward_l_ = rel_obj_vel_b.squaredNorm();
            body_vel_reward_l_ = bodyLinearVel_l_.squaredNorm();
            body_qvel_reward_l_ = bodyAngularVel_l_.squaredNorm();
            contact_reward_l_ = k_contact_l_*(rel_contacts_l_.sum());
            impulse_reward_l_ = ((final_contact_array_l_.cwiseProduct(impulses_l_)).sum());

            obj_avel_reward_ = obj_avel_.squaredNorm();


            if(isnan(impulse_reward_r_))
                impulse_reward_r_ = 0.0;
            if(isnan(impulse_reward_l_))
                impulse_reward_l_ = 0.0;

            obj_vel_reward_r_ = Obj_linvel_t.e().squaredNorm();
            obj_qvel_reward_r_ = Obj_qvel_t.e().squaredNorm();
            obj_vel_reward_l_ = Obj_linvel_b.e().squaredNorm();
            obj_qvel_reward_l_ = Obj_qvel_b.e().squaredNorm();
            obj_pos_reward_r_ = -rel_obj_goal_pos_t_.squaredNorm();
            obj_pose_reward_r_ = -rel_obj_pose_world_t_.e().squaredNorm();
            obj_pos_reward_l_ = -rel_obj_goal_pos_b_.squaredNorm();
            obj_pose_reward_l_ = -rel_obj_pose_world_b_.e().squaredNorm();

            if (right_kind_idx == 7){
                rewards_r_.record("pos_reward", 0);
                rewards_r_.record("pose_reward", 0);
                rewards_r_.record("contact_reward", 0);
                rewards_r_.record("impulse_reward", 0);
                rewards_r_.record("rel_obj_vel_reward_", 0);
                rewards_r_.record("body_vel_reward_", 0);
                rewards_r_.record("body_qvel_reward_", 0);
                rewards_r_.record("torque", 0);
                rewards_r_.record("obj_vel_reward_", 0);
                rewards_r_.record("obj_qvel_reward_", 0);
                rewards_r_.record("obj_angle_reward_", 0);
                rewards_r_.record("obj_avel_reward_", 0);
                rewards_r_.record("obj_pos_reward_", 0);
            }
            else if (right_kind_idx == 8){
                rewards_r_.record("pos_reward", std::max(-10.0, pos_reward_r_));
                rewards_r_.record("pose_reward", std::max(-10.0, pose_reward_r_));
                rewards_r_.record("contact_reward", std::max(-10.0, contact_reward_r_));
                rewards_r_.record("impulse_reward", std::min(impulse_reward_r_, (obj_weight_t_+obj_weight_b_)*5));
                rewards_r_.record("rel_obj_vel_reward_", std::max(0.0, rel_obj_vel_reward_r_));
                rewards_r_.record("body_vel_reward_", std::max(0.0,body_vel_reward_r_));
                rewards_r_.record("body_qvel_reward_", std::max(0.0,body_qvel_reward_r_));
                rewards_r_.record("torque", std::max(0.0, mano_r_->getGeneralizedForce().squaredNorm()));
                rewards_r_.record("obj_vel_reward_", std::max(-10.0, obj_vel_reward_l_));
                rewards_r_.record("obj_qvel_reward_", std::max(-10.0, obj_qvel_reward_l_));
                rewards_r_.record("obj_angle_reward_", std::max(-10.0, obj_angle_reward_));
                rewards_r_.record("obj_avel_reward_", std::max(-10.0, obj_avel_reward_));
                rewards_r_.record("obj_pos_reward_", std::max(-10.0, obj_pos_reward_l_));
            }
            else if (right_kind_idx == 9){
                rewards_r_.record("pos_reward", std::max(-10.0, pos_reward_r_));
                rewards_r_.record("pose_reward", std::max(-10.0, pose_reward_r_));
                rewards_r_.record("contact_reward", std::max(-10.0, contact_reward_r_));
                rewards_r_.record("impulse_reward", std::min(impulse_reward_r_, (obj_weight_t_+obj_weight_b_)*5));
                rewards_r_.record("rel_obj_vel_reward_", std::max(0.0, rel_obj_vel_reward_r_));
                rewards_r_.record("body_vel_reward_", std::max(0.0,body_vel_reward_r_));
                rewards_r_.record("body_qvel_reward_", std::max(0.0,body_qvel_reward_r_));
                rewards_r_.record("torque", std::max(0.0, mano_r_->getGeneralizedForce().squaredNorm()));
                rewards_r_.record("obj_vel_reward_", std::max(-10.0, obj_vel_reward_l_));
                rewards_r_.record("obj_qvel_reward_", std::max(-10.0, obj_qvel_reward_l_));
                rewards_r_.record("obj_angle_reward_", 0);
                rewards_r_.record("obj_avel_reward_", 0);
                rewards_r_.record("obj_pos_reward_", std::max(-10.0, obj_pos_reward_l_));
            }
            else{
                rewards_r_.record("pos_reward", 0);
                rewards_r_.record("pose_reward", 0);
                rewards_r_.record("contact_reward", 0);
                rewards_r_.record("impulse_reward", 0);
                rewards_r_.record("rel_obj_vel_reward_", 0);
                rewards_r_.record("body_vel_reward_", 0);
                rewards_r_.record("body_qvel_reward_", 0);
                rewards_r_.record("torque", 0);
                rewards_r_.record("obj_vel_reward_", 0);
                rewards_r_.record("obj_qvel_reward_", 0);
                rewards_r_.record("obj_angle_reward_", 0);
                rewards_r_.record("obj_avel_reward_", 0);
                rewards_r_.record("obj_pos_reward_", 0);
            }

            if (left_kind_idx == 7){
                rewards_l_.record("pos_reward", 0);
                rewards_l_.record("pose_reward", 0);
                rewards_l_.record("contact_reward", 0);
                rewards_l_.record("impulse_reward", 0);
                rewards_l_.record("rel_obj_vel_reward_", 0);
                rewards_l_.record("body_vel_reward_", 0);
                rewards_l_.record("body_qvel_reward_", 0);
                rewards_l_.record("torque", 0);
                rewards_l_.record("obj_angle_reward_", 0);
                rewards_l_.record("obj_avel_reward_",0);
                rewards_l_.record("obj_vel_reward_", 0);
                rewards_l_.record("obj_qvel_reward_", 0);
                rewards_l_.record("obj_pos_reward_", 0);
            }
            else if (left_kind_idx == 8){
                rewards_l_.record("pos_reward", std::max(-10.0, pos_reward_l_));
                rewards_l_.record("pose_reward", std::max(-10.0, pose_reward_l_));
                rewards_l_.record("contact_reward", std::max(-10.0, contact_reward_l_));
                rewards_l_.record("impulse_reward", std::min(impulse_reward_l_, (obj_weight_t_+obj_weight_b_)*5));
                rewards_l_.record("rel_obj_vel_reward_", std::max(0.0, rel_obj_vel_reward_l_));
                rewards_l_.record("body_vel_reward_", std::max(0.0,body_vel_reward_l_));
                rewards_l_.record("body_qvel_reward_", std::max(0.0,body_qvel_reward_l_));
                rewards_l_.record("torque", std::max(0.0, mano_l_->getGeneralizedForce().squaredNorm()));
                rewards_l_.record("obj_angle_reward_", std::max(-10.0, obj_angle_reward_));
                rewards_l_.record("obj_avel_reward_", std::max(-10.0, obj_avel_reward_));
                rewards_l_.record("obj_vel_reward_", std::max(-10.0, obj_vel_reward_l_));
                rewards_l_.record("obj_qvel_reward_", std::max(-10.0, obj_qvel_reward_l_));
                rewards_l_.record("obj_pos_reward_", std::max(-10.0, obj_pos_reward_l_));
            }
            else if (left_kind_idx == 9){
                rewards_l_.record("pos_reward", std::max(-10.0, pos_reward_l_));
                rewards_l_.record("pose_reward", std::max(-10.0, pose_reward_l_));
                rewards_l_.record("contact_reward", std::max(-10.0, contact_reward_l_));
                rewards_l_.record("impulse_reward", std::min(impulse_reward_l_, (obj_weight_t_+obj_weight_b_)*5));
                rewards_l_.record("rel_obj_vel_reward_", std::max(0.0, rel_obj_vel_reward_l_));
                rewards_l_.record("body_vel_reward_", std::max(0.0,body_vel_reward_l_));
                rewards_l_.record("body_qvel_reward_", std::max(0.0,body_qvel_reward_l_));
                rewards_l_.record("torque", std::max(0.0, mano_l_->getGeneralizedForce().squaredNorm()));
                rewards_l_.record("obj_angle_reward_", 0);
                rewards_l_.record("obj_avel_reward_", 0);
                rewards_l_.record("obj_vel_reward_", std::max(-10.0, obj_vel_reward_l_));
                rewards_l_.record("obj_qvel_reward_", std::max(-10.0, obj_qvel_reward_l_));
                rewards_l_.record("obj_pos_reward_", std::max(-10.0, obj_pos_reward_l_));
            }
            else{
                rewards_l_.record("pos_reward", 0);
                rewards_l_.record("pose_reward", 0);
                rewards_l_.record("contact_reward", 0);
                rewards_l_.record("impulse_reward", 0);
                rewards_l_.record("rel_obj_vel_reward_", 0);
                rewards_l_.record("body_vel_reward_", 0);
                rewards_l_.record("body_qvel_reward_", 0);
                rewards_l_.record("torque", 0);
                rewards_l_.record("obj_angle_reward_", 0);
                rewards_l_.record("obj_avel_reward_",0);
                rewards_l_.record("obj_vel_reward_", 0);
                rewards_l_.record("obj_qvel_reward_", 0);
                rewards_l_.record("obj_pos_reward_", 0);
            }

            rewards_sum_[0] = rewards_r_.sum();
            rewards_sum_[1] = rewards_l_.sum();

            return rewards_sum_;
        }

        float* step_imitate(const Eigen::Ref<EigenVec>& action_r,
                            const Eigen::Ref<EigenVec>& action_l,
                            const Eigen::Ref<EigenVec>& obj_pose_step_r,
                            const Eigen::Ref<EigenVec>& hand_ee_step_r,
                            const Eigen::Ref<EigenVec>& hand_pose_step_r,
                            const Eigen::Ref<EigenVec>& obj_pose_step_l,
                            const Eigen::Ref<EigenVec>& hand_ee_step_l,
                            const Eigen::Ref<EigenVec>& hand_pose_step_l,
                            const bool imitate_right,
                            const bool imitate_left)
        final {

        }

        /// This function computes and updates the observation/state space
        void updateObservation() {
        }

        void updateObservation2() {
            // update observation
            raisim::Vec<4> quat, quat_hand, quat_obj_init;
            raisim::Vec<3> body_vel, obj_frame_diff_r, obj_frame_diff_l, obj_frame_diff_w, obj_frame_diff_h_r, obj_frame_diff_h_l, euler_hand_r, euler_hand_l, sphere_pos, norm_pos, rel_wbody_root, final_obj_euler, euler_obj, rel_rbody_root, rel_body_table_r, rel_body_table_l, rel_obj_init_b, rel_obj_init_t, rel_objpalm_r, rel_objpalm_l, rel_obj_pose_r3;
            raisim::Mat<3,3> rot, rot_mult_r, rot_mult_l, body_orientation_transpose_r, body_orientation_transpose_l, palm_world_pose_mat_r, palm_world_pose_mat_l, palm_world_pose_mat_trans_r, palm_world_pose_mat_trans_l, obj_pose_wrist_mat_r, obj_pose_wrist_mat_l, rel_pose_mat, final_obj_rotmat_temp, diff_obj_pose_mat, final_obj_wrist, obj_wrist, obj_wrist_trans, final_obj_pose_mat, mat_temp, mat_rot_left;
            raisim::Mat<3,3> obj_orientation_temp_trans_b, obj_orientation_temp_trans_t;


            contacts_r_.setZero();
            rel_contacts_r_.setZero();
            impulses_r_.setZero();
            contacts_l_.setZero();
            rel_contacts_l_.setZero();
            impulses_l_.setZero();

            /// Get updated hand state
            mano_r_->getState(gc_r_, gv_r_);
            mano_l_->getState(gc_l_, gv_l_);

            arctic->getState(gc_obj_, gv_obj_);

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

            raisim::RotmatToEuler(Obj_orientation_temp_b, obj_euler_b_);
            raisim::RotmatToEuler(Obj_orientation_temp_t, obj_euler_t_);

            // distance from current position to final position, in world coord
            rel_obj_goal_pos_b_[0] = obj_pos_set_b[0] - Obj_Position_b[0];
            rel_obj_goal_pos_b_[1] = obj_pos_set_b[1] - Obj_Position_b[1];
            rel_obj_goal_pos_b_[2] = obj_pos_set_b[2] - Obj_Position_b[2];
            rel_obj_goal_pos_t_[0] = obj_pos_set_t[0] - Obj_Position_t[0];
            rel_obj_goal_pos_t_[1] = obj_pos_set_t[1] - Obj_Position_t[1];
            rel_obj_goal_pos_t_[2] = obj_pos_set_t[2] - Obj_Position_t[2];



            // relative distance in wrist frame
            raisim::matvecmul(init_or_l_, rel_obj_goal_pos_b_, rel_obj_pos_wrist_l_);
            raisim::matvecmul(init_or_r_, rel_obj_goal_pos_t_, rel_obj_pos_wrist_r_);

            obj_angle_[0] = arctic->getGeneralizedCoordinate().e()[7];
            obj_avel_[0] = arctic->getGeneralizedVelocity().e()[6];
            rel_obj_angle_ = final_obj_angle_ - obj_angle_;


            raisim::transpose(Obj_orientation_temp_b, Obj_orientation_b);
            raisim::transpose(Obj_orientation_temp_t, Obj_orientation_t);

            /// compute relative hand pose to final pose
            rel_pose_r_ = final_pose_r2_ - gc_r_.tail(gcDim_-3);
            rel_pose_l_ = final_pose_l_ - gc_l_.tail(gcDim_-3);

            /// compute object pose in wrist frame
            mano_r_->getFrameOrientation(body_parts_r_[0], palm_world_pose_mat_r);
            mano_l_->getFrameOrientation(body_parts_l_[0], palm_world_pose_mat_l);
            raisim::transpose(palm_world_pose_mat_r,palm_world_pose_mat_trans_r);
            raisim::transpose(palm_world_pose_mat_l,palm_world_pose_mat_trans_l);

            //distance from current orientation to final orientation, in world coord
            raisim::transpose(Obj_orientation_temp_b, obj_orientation_temp_trans_b);
            raisim::matmul(Obj_orientation_set_b.e().transpose(), Obj_orientation_temp_b, diff_obj_pose_mat);
            raisim::Mat<3,3> diff_obj_pose_mat_wrist;
            raisim::matmul(palm_world_pose_mat_trans_l, diff_obj_pose_mat, diff_obj_pose_mat_wrist);
            raisim::RotmatToEuler(diff_obj_pose_mat, rel_obj_pose_world_b_);
            raisim::RotmatToEuler(diff_obj_pose_mat_wrist, rel_obj_pose_wrist_b_);

            raisim::transpose(Obj_orientation_temp_t, obj_orientation_temp_trans_t);
            raisim::matmul(Obj_orientation_set_t.e().transpose(), Obj_orientation_temp_t, diff_obj_pose_mat);
            raisim::RotmatToEuler(diff_obj_pose_mat, rel_obj_pose_world_t_);

            raisim::matmul(palm_world_pose_mat_trans_r, Obj_orientation_temp_t, obj_pose_wrist_mat_r);
            raisim::RotmatToEuler(obj_pose_wrist_mat_r, obj_pose_r_); // obj_pose_: object pose in wrist frame
            raisim::matmul(palm_world_pose_mat_trans_l, Obj_orientation_temp_b, obj_pose_wrist_mat_l);
            raisim::RotmatToEuler(obj_pose_wrist_mat_l, obj_pose_l_); // obj_pose_: object pose in wrist frame
            raisim::Mat<3,3> obj_pose_diff_wrist_mat;
            raisim::matmul(init_obj_ori_wrist_l.e().transpose(), obj_pose_wrist_mat_l, obj_pose_diff_wrist_mat);
            raisim::RotmatToEuler(obj_pose_diff_wrist_mat, obj_pose_diff_wrist);


            //Calculate wrist pose in object frame
            raisim::Vec<3> wrist_temp;
            raisim::matvecmul(init_rot_r_, gc_r_.head(3), wrist_temp);
            raisim::vecadd(init_root_r_, wrist_temp);
            raisim::vecsub(Obj_Position_t, wrist_temp);
            raisim::matvecmul(Obj_orientation_t, wrist_temp, wrist_pos_obj_r_);

            mano_r_->getFrameOrientation(body_parts_r_[0], Body_orientation_r);
            raisim::Mat<3,3> wrist_pose_obj_r_mat;
            raisim::matmul(Obj_orientation_t, Body_orientation_r, wrist_pose_obj_r_mat);
            raisim::RotmatToEuler(wrist_pose_obj_r_mat, wrist_pose_obj_r_);

            /// iterate over all hand parts to compute relative distances, poses, etc.
            for(int i = 0; i < num_bodyparts ; i++){
                mano_r_->getFramePosition(body_parts_r_[i], Position_r); // Position: body position of a body in mano
                mano_r_->getFrameOrientation(body_parts_r_[i], Body_orientation_r); // Body_orientation:
                mano_l_->getFramePosition(body_parts_l_[i], Position_l); // Position: body position of a body in mano
                mano_l_->getFrameOrientation(body_parts_l_[i], Body_orientation_l); // Body_orientation:
                /// for the hand root, compute relevant features
                if (i==0)
                {
                    wrist_pos_l = Position_l;
                    wrist_ori_l = Body_orientation_l;
                    raisim::transpose(Body_orientation_r, body_orientation_transpose_r);
                    rel_objpalm_r[0] = Position_r[0]-Obj_Position_t[0];
                    rel_objpalm_r[1] = Position_r[1]-Obj_Position_t[1];
                    rel_objpalm_r[2] = Position_r[2]-Obj_Position_t[2];

                    rel_objpalm_pos_r_ = Body_orientation_r.e().transpose()*rel_objpalm_r.e();

                    raisim::transpose(Body_orientation_l, body_orientation_transpose_l);
                    rel_objpalm_l[0] = Position_l[0]-Obj_Position_b[0];
                    rel_objpalm_l[1] = Position_l[1]-Obj_Position_b[1];
                    rel_objpalm_l[2] = Position_l[2]-Obj_Position_b[2];

                    rel_objpalm_pos_l_ = Body_orientation_l.e().transpose()*rel_objpalm_l.e();
                    rel_obj_pos_b[0] = init_rel_objpalm_pos_l[0]-rel_objpalm_pos_l_[0];
                    rel_obj_pos_b[1] = init_rel_objpalm_pos_l[1]-rel_objpalm_pos_l_[1];
                    rel_obj_pos_b[2] = init_rel_objpalm_pos_l[2]-rel_objpalm_pos_l_[2];

                    rel_body_table_r[0] = 0.0;
                    rel_body_table_r[1] = 0.0;
                    rel_body_table_r[2] = Position_r[2]-0.5;
                    rel_body_table_pos_r_ = Body_orientation_r.e().transpose()*rel_body_table_r.e(); // z-distance to the table in wrist coordinates
                    rel_body_table_l[0] = 0.0;
                    rel_body_table_l[1] = 0.0;
                    rel_body_table_l[2] = Position_l[2]-0.5;
                    rel_body_table_pos_l_ = Body_orientation_l.e().transpose()*rel_body_table_l.e();

                    rel_obj_init_b[0] = obj_pos_set_b[0] - Obj_Position_b[0];
                    rel_obj_init_b[1] = obj_pos_set_b[1] - Obj_Position_b[1];
                    rel_obj_init_b[2] = obj_pos_set_b[2] - Obj_Position_b[2];
                    rel_obj_init_t[0] = obj_pos_set_t[0] - Obj_Position_t[0];
                    rel_obj_init_t[1] = obj_pos_set_t[1] - Obj_Position_t[1];
                    rel_obj_init_t[2] = obj_pos_set_t[2] - Obj_Position_t[2];

                    rel_obj_pos_r_ = Body_orientation_r.e().transpose()*rel_obj_init_t.e(); // object displacement from initial position in wrist coordinates
                    rel_obj_pos_l_ = Body_orientation_l.e().transpose()*rel_obj_init_b.e();

                    raisim::matmul(Obj_orientation_t, Body_orientation_r, rot_mult_r); // current global wirst pose in object relative frame
                    raisim::RotmatToEuler(rot_mult_r, euler_hand_r);
                    raisim::matmul(Obj_orientation_b, Body_orientation_l, rot_mult_l);
                    raisim::RotmatToEuler(rot_mult_l, euler_hand_l);

                    rel_pose_r_.head(3) = final_pose_r2_.head(3) - euler_hand_r.e(); // difference between target and current global wrist pose
                    rel_pose_l_.head(3) = final_pose_l_.head(3) - euler_hand_l.e();

                    bodyLinearVel_r_ =  gv_r_.segment(0, 3);
                    bodyAngularVel_r_ = gv_r_.segment(3, 3);
                    bodyLinearVel_l_ =  gv_l_.segment(0, 3);
                    bodyAngularVel_l_ = gv_l_.segment(3, 3);

                    mano_r_->getFramePosition(body_parts_r_[0], palm_world_pos_r_);
                    mano_l_->getFramePosition(body_parts_l_[0], palm_world_pos_l_);

                    mano_l_->getFrameAngularVelocity(body_parts_l_[0], angular_vel_l);
                    mano_l_->getFrameVelocity(body_parts_l_[0], vel_l);

                    mano_r_->getFrameAngularVelocity(body_parts_r_[0], angular_vel_r);
                    mano_r_->getFrameVelocity(body_parts_r_[0], vel_r);

                    raisim::Vec<3> temp_rel_vel_l, temp_rel_vel_r, temp_rel_ang_vel_l, temp_rel_ang_vel_r;
                    temp_rel_vel_l[0] = Obj_linvel_b[0] - vel_l[0];
                    temp_rel_vel_l[1] = Obj_linvel_b[1] - vel_l[1];
                    temp_rel_vel_l[2] = Obj_linvel_b[2] - vel_l[2];

                    temp_rel_vel_r[0] = Obj_linvel_t[0] - vel_r[0];
                    temp_rel_vel_r[1] = Obj_linvel_t[1] - vel_r[1];
                    temp_rel_vel_r[2] = Obj_linvel_t[2] - vel_r[2];

                    temp_rel_ang_vel_l[0] = Obj_qvel_b[0] - angular_vel_l[0];
                    temp_rel_ang_vel_l[1] = Obj_qvel_b[1] - angular_vel_l[1];
                    temp_rel_ang_vel_l[2] = Obj_qvel_b[2] - angular_vel_l[2];

                    temp_rel_ang_vel_r[0] = Obj_qvel_t[0] - angular_vel_r[0];
                    temp_rel_ang_vel_r[1] = Obj_qvel_t[1] - angular_vel_r[1];
                    temp_rel_ang_vel_r[2] = Obj_qvel_t[2] - angular_vel_r[2];

                    rel_obj_vel_t = Body_orientation_r.e().transpose() * temp_rel_vel_r.e(); // object velocity in wrist frame
                    rel_obj_qvel_t = Body_orientation_r.e().transpose() * temp_rel_ang_vel_r.e();
                    rel_obj_vel_b = Body_orientation_l.e().transpose() * temp_rel_vel_l.e();
                    rel_obj_qvel_b = Body_orientation_l.e().transpose() * temp_rel_ang_vel_l.e();

                    raisim::quatToRotMat(final_obj_pos_b_.segment(3,4), final_obj_pose_mat); // final_obj_pose_mat: final object orientation in matrix
                    raisim::matmul(init_or_r_, final_obj_pose_mat, final_obj_wrist); // object final pose in wrist frame
                    raisim::matmul(init_or_r_, Obj_orientation_temp_b, obj_wrist); // object pose in wrist frame
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

                obj_frame_diff_r[0] = final_ee_pos_r2_[i * 3]- Rel_fpos[0];
                obj_frame_diff_r[1] = final_ee_pos_r2_[i * 3 + 1] - Rel_fpos[1];
                obj_frame_diff_r[2] = final_ee_pos_r2_[i * 3 + 2] - Rel_fpos[2]; // distance between target 3D positions and current 3D positions in object frame

                Position_l[0] = Position_l[0] - Obj_Position_b[0];
                Position_l[1] = Position_l[1] - Obj_Position_b[1];
                Position_l[2] = Position_l[2] - Obj_Position_b[2];
                raisim::matvecmul(Obj_orientation_b, Position_l, Rel_fpos);

                obj_frame_diff_l[0] = final_ee_pos_l_[i * 3] - Rel_fpos[0];
                obj_frame_diff_l[1] = final_ee_pos_l_[i * 3 + 1] - Rel_fpos[1];
                obj_frame_diff_l[2] = final_ee_pos_l_[i * 3 + 2] - Rel_fpos[2]; // distance between target 3D positions and current 3D positions in object frame

                obj_body_pos_r_[i * 3] = obj_frame_diff_r[0];
                obj_body_pos_r_[i * 3 + 1] = obj_frame_diff_r[1];
                obj_body_pos_r_[i * 3 + 2] = obj_frame_diff_r[2];
                obj_body_pos_l_[i * 3] = obj_frame_diff_l[0];
                obj_body_pos_l_[i * 3 + 1] = obj_frame_diff_l[1];
                obj_body_pos_l_[i * 3 + 2] = obj_frame_diff_l[2];

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
                    raisim::matvecmul(Obj_orientation_temp_t, {final_ee_pos_r2_[i * 3], final_ee_pos_r2_[i * 3 + 1], final_ee_pos_r2_[i * 3 + 2]}, sphere_pos);
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
                if (right_kind_idx == 8){
                    if (contact_list_obj[contact.getPairContactIndexInPairObject()].getlocalBodyIndex() != top_id) continue;
                }
                contacts_r_[contactMapping_r_[contact.getlocalBodyIndex()]] = 1;
                impulses_r_[contactMapping_r_[contact.getlocalBodyIndex()]] = contact.getImpulse().norm();
            }

            for(auto& contact: mano_l_->getContacts()) {
                if (contact.skip() || contact.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                if (left_kind_idx == 8){
                    if (contact_list_obj[contact.getPairContactIndexInPairObject()].getlocalBodyIndex() != top_id) continue;
                }
                contacts_l_[contactMapping_l_[contact.getlocalBodyIndex()]] = 1;
                impulses_l_[contactMapping_l_[contact.getlocalBodyIndex()]] = contact.getImpulse().norm();
            }
            force_r_.setZero(3);
            force_l_.setZero(3);
            torque_r_.setZero(3);
            torque_l_.setZero(3);
            force_norm_r_ = 0.0;
            force_norm_l_ = 0.0;

            for(auto& contact: contact_list_obj) {
                if (contact.skip() ||
                    contact.getPairObjectIndex() != mano_r_->getIndexInWorld() &&
                    contact.getPairObjectIndex() != mano_l_->getIndexInWorld())
                    continue;
                Eigen::Vector3d impulse_w, impulse_obj;
                impulse_w = contact.getContactFrame().e().transpose() * contact.getImpulse().e();
                if (!contact.isObjectA()) impulse_w = -impulse_w;
                impulse_obj = obj_orientation_temp_trans_b.e() * impulse_obj;
                Eigen::Vector3d contact_pos_w, contact_pos_obj;
                contact_pos_w = contact.getPosition().e();
                contact_pos_obj = contact_pos_w - Obj_Position_b.e();
                contact_pos_obj = obj_orientation_temp_trans_b.e() * contact_pos_obj;
                Eigen::Vector3d moment_obj = contact_pos_obj.cross(impulse_obj);
                if (contact.getPairObjectIndex() == mano_r_->getIndexInWorld()) {
                    force_r_ += impulse_obj;
                    torque_r_ += moment_obj;
                    force_norm_r_ += impulse_obj.norm();
                }
                else {
                    force_l_ += impulse_obj;
                    torque_l_ += moment_obj;
                    force_norm_l_ += impulse_obj.norm();
                }
            }

            /// compute relative target contact vector, i.e., which goal contacts are currently in contact
            rel_contacts_r_ = final_contact_array_r2_.cwiseProduct(contacts_r_);
            rel_contacts_l_ = final_contact_array_l_.cwiseProduct(contacts_l_);

            raisim::RotmatToEuler(wrist_ori_l, wrist_l_euler_world);


            raisim::Vec<3> body_vel_r_w, body_avel_r_w, body_vel_l_w, body_avel_l_w;
            raisim::Vec<3> body_vel_r_o, body_avel_r_o, body_vel_l_o, body_avel_l_o;

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

            raisim::Mat<3,3> wrist_ori_r;
            mano_r_->getFrameOrientation(body_parts_r_[0], wrist_ori_r);
            Eigen::Vector3d rotation_axis_obj, rotation_axis_w, rotation_axis_h, rotation_axis_l;
            Eigen::Vector3d arm_in_obj, arm_in_w, arm_in_wrist, wrist_in_obj, proj_in_obj;
            Eigen::Vector3d arm_in_obj_l, arm_in_w_l, arm_in_wrist_l, wrist_in_obj_l, proj_in_obj_l;

            wrist_in_obj = Obj_orientation_temp_b.e().transpose()*rel_objpalm_r.e();
            proj_in_obj.setZero();
            proj_in_obj[2] = wrist_in_obj[2];
            arm_in_obj.setZero();
            arm_in_obj[0] = -wrist_in_obj[0];
            arm_in_obj[1] = -wrist_in_obj[1];
            arm_in_obj[2] = 0;
            arm_in_w = Obj_orientation_temp_b.e() * arm_in_obj;
            arm_in_wrist = wrist_ori_r.e().transpose() * arm_in_w;

            wrist_in_obj_l = Obj_orientation_temp_b.e().transpose()*rel_objpalm_l.e();
            proj_in_obj_l.setZero();
            proj_in_obj_l[2] = wrist_in_obj_l[2];
            arm_in_obj_l.setZero();
            arm_in_obj_l[0] = -wrist_in_obj_l[0];
            arm_in_obj_l[1] = -wrist_in_obj_l[1];
            arm_in_obj_l[2] = 0;
            arm_in_w_l = Obj_orientation_temp_b.e() * arm_in_obj_l;
            arm_in_wrist_l = wrist_ori_l.e().transpose() * arm_in_w_l;

            rotation_axis_obj.setZero();
            rotation_axis_obj[2] = 1;
            rotation_axis_w = Obj_orientation_temp_b.e() * rotation_axis_obj;
            rotation_axis_h = wrist_ori_r.e().transpose() * rotation_axis_w;
            rotation_axis_l = wrist_ori_l.e().transpose() * rotation_axis_w;

            obDouble_r_ << gc_r_.tail(gcDim_ - 6),      // (mirror) 51, generalized coordinate
                    bodyLinearVel_r_obs_,  // (mirror) 3, wrist linear velocity
                    bodyAngularVel_r_obs_, // (mirror) 3, wrist angular velocity
                    gv_r_.tail(gvDim_ - 6), // (mirror) 45, joint anglular velocity, global

                    rel_body_pos_r_,    //  (x mirror) 63, joint position relative to target position in wrist coord,   ******
                    rel_pose_r_,  // (need to mirror the first 3 dimension) 48, angle between current pose and final pose, wrist pose in object coord

                    rel_objpalm_pos_r_, // (x mirror) 3, relative position between object and wrist in wrist coordinates
                    rel_obj_vel_t,  // (x mirror) 3, object velocity in wrist frame, regularization
                    rel_obj_qvel_t, // (yz mirror) 3, object angular velocity in wrist frame, regularization

                    final_contact_array_r2_, // 16, goal contact array for all parts ******
                    impulses_r_, // 16, impulse array
                    rel_contacts_r_, // 16, relative target contact vector, i.e., which goal contacts are currently in contact  ******

                    arm_in_wrist,
                    arm_in_wrist.norm(),  //norm of arm
                    obj_weight_b_,  // weight of bottom
                    obj_weight_t_, // weight of top
                    rotation_axis_h, // rotation axis in wrist frame

                    obj_angle_, // current object angle
                    obj_avel_, // object angular velocity

                    rel_obj_angle_; // relative object angle to the goal   ******

             obDouble_l_ << gc_l_.tail(gcDim_ - 6),      // (mirror) 45, generalized coordinate
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

                    arm_in_wrist_l,
                    arm_in_wrist_l.norm(),  //norm of arm
                    obj_weight_b_,  // weight of bottom
                    obj_weight_t_, // weight of top
                    rotation_axis_l, // rotation axis in wrist frame

                    obj_angle_, // current object angle
                    obj_avel_, // object angular velocity
                    rel_obj_angle_; // relative object angle to the goal;

        }


        /// Set observation in wrapper to current observation
        void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
            ob_r = obDouble_r_.cast<float>();
            ob_l = obDouble_l_.cast<float>();
        }

        void get_global_state(Eigen::Ref<EigenVec> gs) {
            gs = global_state_.cast<float>();
        }

        /// This function is only relevant for testing
        /// It increases the gain of the root control and lowers the table surface
        void set_rootguidance() final {}

        void control_switch(int left, int right) {
            left_kind_idx = left;
            right_kind_idx = right;

            if (left == 7){
                mano_l_->getState(gc_set_l_, gv_set_l_);
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                root_guided_l = false;
                root_keep_l = true;
                motion_synthesis_l = false;
                jointPgain.head(3).setConstant(500);
                jointDgain.head(3).setConstant(0.1);
                jointPgain.tail(nJoints_).setConstant(50.0);
                jointDgain.tail(nJoints_).setConstant(0.2);
                mano_l_->setPdGains(jointPgain, jointDgain);
            }
            else if (left == 8){
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                root_guided_l = true;
                root_keep_l = false;
                motion_synthesis_l = false;
                jointPgain.head(3).setConstant(50);
                jointDgain.head(3).setConstant(0.1);
                jointPgain.tail(nJoints_).setConstant(50.0);
                jointDgain.tail(nJoints_).setConstant(0.2);
                mano_l_->setPdGains(jointPgain, jointDgain);
            }
            else if (left == 9){
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                root_guided_l = true;
                root_keep_l = false;
                motion_synthesis_l = false;
                jointPgain.head(3).setConstant(50);
                jointDgain.head(3).setConstant(0.1);
                jointPgain.tail(nJoints_).setConstant(50.0);
                jointDgain.tail(nJoints_).setConstant(0.2);
                mano_l_->setPdGains(jointPgain, jointDgain);
            }
            else {
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                root_guided_l = false;
                root_keep_l = false;
                motion_synthesis_l = true;
                jointPgain.head(3).setConstant(1000);
                jointDgain.head(3).setConstant(5.0);
                jointPgain.tail(nJoints_).setConstant(50.0);
                jointDgain.tail(nJoints_).setConstant(1.0);
                mano_l_->setPdGains(jointPgain, jointDgain);
            }

            if (right == 7){
                mano_r_->getState(gc_set_r_, gv_set_r_);
                motion_synthesis = false;
                root_guided_r = false;
                root_keep_r = true;
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(500);
                jointDgain.head(3).setConstant(0.1);
                jointPgain.tail(nJoints_).setConstant(50.0);
                jointDgain.tail(nJoints_).setConstant(0.2);
                mano_r_->setPdGains(jointPgain, jointDgain);
            }
            else if (right == 8){
                motion_synthesis = false;
                root_guided_r = true;
                root_keep_r = false;
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(50);
                jointDgain.head(3).setConstant(0.1);
                jointPgain.tail(nJoints_).setConstant(50.0);
                jointDgain.tail(nJoints_).setConstant(0.2);
                mano_r_->setPdGains(jointPgain, jointDgain);
            }
            else if (right == 9){
                motion_synthesis = false;
                root_guided_r = true;
                root_keep_r = false;
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(50);
                jointDgain.head(3).setConstant(0.1);
                jointPgain.tail(nJoints_).setConstant(50.0);
                jointDgain.tail(nJoints_).setConstant(0.2);
                mano_r_->setPdGains(jointPgain, jointDgain);
            }
            else {
                motion_synthesis = true;
                root_guided_r = false;
                root_keep_r = false;
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(1000);
                jointDgain.head(3).setConstant(5.0);
                jointPgain.tail(nJoints_).setConstant(50.0);
                jointDgain.tail(nJoints_).setConstant(1.0);
                mano_r_->setPdGains(jointPgain, jointDgain);
            }

        }

        /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
        bool isTerminalState(float& terminalReward) final {

            if(obDouble_l_.hasNaN() || obDouble_r_.hasNaN() || global_state_.hasNaN() ){
                return true;
            }

            return false;
        }

        void set_pregrasp_imitate(const Eigen::Ref<EigenVec>& obj_pregrasp_pos_r,
                                  const Eigen::Ref<EigenVec>& ee_pregrasp_pos_r,
                                  const Eigen::Ref<EigenVec>& pregrasp_pose_r,
                                  const Eigen::Ref<EigenVec>& obj_pregrasp_pos_l,
                                  const Eigen::Ref<EigenVec>& ee_pregrasp_pos_l,
                                  const Eigen::Ref<EigenVec>& pregrasp_pose_l)
        {
            raisim::Vec<4> quat_goal_hand_w, quat_goal_hand_r, quat_obj_init_t, quat_obj_init_b;
            raisim::Vec<3> euler_goal_pose;
            raisim::Mat<3,3> rotm_goal_hand_r, mat_temp, mat_rot_left, rotm_goal_t, rotm_goal_tran_t;

            /// set final object pose

            /// convert object and handpose pose to rotation matrix format
            raisim::quatToRotMat(obj_pregrasp_pos_r.tail(4), Obj_orientation_temp_t);
            raisim::quatToRotMat(obj_pregrasp_pos_l.tail(4), Obj_orientation_temp_b);

            quat_obj_init_t = obj_pregrasp_pos_r.tail(4); //quat_obj_init: orientation of object goal pose, in quaternion
            quat_obj_init_b = obj_pregrasp_pos_l.tail(4);

            raisim::transpose(Obj_orientation_temp_t, Obj_orientation_t); // Obj_orientation: relative orientation of world coordinate to object
            raisim::transpose(Obj_orientation_temp_b, Obj_orientation_b);

            raisim::eulerToQuat(pregrasp_pose_l.head(3), quat_goal_hand_w);

            /// Compute and set object relative goal hand pose
            raisim::quatInvQuatMul(quat_obj_init_b, quat_goal_hand_w, quat_goal_hand_r);
            raisim::quatToRotMat(quat_goal_hand_r, rotm_goal_hand_r);
            raisim::RotmatToEuler(rotm_goal_hand_r, euler_goal_pose);

            pregrasp_pose_l_ = pregrasp_pose_l.cast<double>();
            pregrasp_pose_l_.head(3) = euler_goal_pose.e(); // change the orientation of hand final pose to the object frame

            raisim::eulerToQuat(pregrasp_pose_r.head(3), quat_goal_hand_w);

            /// Compute and set object relative goal hand pose
            raisim::quatInvQuatMul(quat_obj_init_t, quat_goal_hand_w, quat_goal_hand_r);
            raisim::quatToRotMat(quat_goal_hand_r, rotm_goal_hand_r);
            raisim::RotmatToEuler(rotm_goal_hand_r, euler_goal_pose);

            pregrasp_pose_r_ = pregrasp_pose_r.cast<double>();
            pregrasp_pose_r_.head(3) = euler_goal_pose.e(); // change the orientation of hand final pose to the object frame

            /// Compute and convert hand 3D joint positions into object relative frame
            Position_l[0] = ee_pregrasp_pos_l[0] - obj_pregrasp_pos_l[0];
            Position_l[1] = ee_pregrasp_pos_l[1] - obj_pregrasp_pos_l[1];
            Position_l[2] = ee_pregrasp_pos_l[2] - obj_pregrasp_pos_l[2];

            raisim::matvecmul(Obj_orientation_b, Position_l, Rel_fpos);

            pregrasp_ee_pos_l_[0] = Rel_fpos[0];
            pregrasp_ee_pos_l_[1] = Rel_fpos[1];
            pregrasp_ee_pos_l_[2] = Rel_fpos[2];

            // right hand
            Position_r[0] = ee_pregrasp_pos_r[0] - obj_pregrasp_pos_r[0];
            Position_r[1] = ee_pregrasp_pos_r[1] - obj_pregrasp_pos_r[1];
            Position_r[2] = ee_pregrasp_pos_r[2] - obj_pregrasp_pos_r[2];

            raisim::matvecmul(Obj_orientation_t, Position_r, Rel_fpos);

            pregrasp_ee_pos_r_[0] = Rel_fpos[0];
            pregrasp_ee_pos_r_[1] = Rel_fpos[1];
            pregrasp_ee_pos_r_[2] = Rel_fpos[2];
        }

        void set_gc_for_arctic(Eigen::VectorXd& gc_for_b, Eigen::VectorXd& gc_for_t, const Eigen::VectorXd& gc_arctic)
        {
            raisim::Vec<3> obj_goal_trans_b, obj_goal_trans_t;
            raisim::Mat<3,3> obj_goal_ori_mat_b, obj_goal_ori_mat_t, base_rot;
            raisim::Vec<4> obj_goal_ori_b, obj_goal_ori_t;
            Eigen::VectorXd obj_goal_angle;


            raisim::quatToRotMat(gc_arctic.segment(3,4), base_rot);
            obj_goal_angle.setZero(1);
            obj_goal_angle[0] = gc_arctic[7];

            arctic->setGeneralizedCoordinate(gc_arctic);

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




    private:
        int gcDim_, gvDim_, nJoints_;
        int gcDim_obj, gvDim_obj, nJoints_obj;
        bool visualizable_ = false;
        raisim::ArticulatedSystem* mano_;
        Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
        Eigen::VectorXd pTarget_r_bk, pTarget_l_bk;
        Eigen::VectorXd gc_l_, gv_l_, pTarget_l_, pTarget6_l_, vTarget_l_, gc_set_l_, gv_set_l_;
        Eigen::VectorXd freezedTarget_l_, freezedTarget_r_;
        Eigen::VectorXd gc_l_obs_, gv_l_obs_;
        Eigen::VectorXd gc_obj_, gv_obj_;
        Eigen::VectorXd obj_pos_init_, obj_pos_init_b_, obj_pos_init_t_;
        Eigen::VectorXd gen_force_r_, gen_force_l_, final_obj_pos_b_, final_obj_pos_t_, final_pose_r_, final_pose_r2_, final_pose_l_, final_ee_pos_r_, final_ee_pos_r2_, final_ee_pos_l_, final_contact_array_r_, final_contact_array_r2_, final_contact_array_l_, contact_body_idx_r_, contact_body_idx_l_, final_vertex_normals_r_, final_vertex_normals_l_;
        Eigen::VectorXd initial_gc_l;
        Eigen::VectorXd pregrasp_pose_l_, pregrasp_ee_pos_l_, pregrasp_pose_r_, pregrasp_ee_pos_r_;
        Eigen::VectorXd final_obj_angle_;
        Eigen::VectorXd rel_obj_goal_pos_b_, rel_obj_goal_pos_t_;
        Eigen::VectorXd force_r_, force_l_, torque_r_, torque_l_;
        int left_kind_idx = 0;
        int right_kind_idx = 0;
        double force_norm_r_ = 0.0, force_norm_l_ = 0.0;
        double terminalRewardCoeff_ = -10.;
        double pose_reward_r_= 0.0, pose_reward_l_= 0.0;
        double pos_reward_r_ = 0.0, pos_reward_l_ = 0.0;
        double contact_reward_r_= 0.0, contact_reward_l_= 0.0;
        double rel_obj_pos_reward_l_ = 0.0;
        double rel_obj_vel_reward_r_ = 0.0, rel_obj_vel_reward_l_ = 0.0;
        double body_vel_reward_r_ = 0.0, body_vel_reward_l_ = 0.0;
        double body_qvel_reward_r_ = 0.0, body_qvel_reward_l_ = 0.0;
        double obj_pose_reward_r_ = 0.0, obj_pose_reward_l_ = 0.0;
        double obj_pos_reward_r_ = 0.0, obj_pos_reward_l_ = 0.0;
        double obj_vel_reward_r_ = 0.0, obj_vel_reward_l_ = 0.0;
        double obj_qvel_reward_r_ = 0.0, obj_qvel_reward_l_ = 0.0;
        double k_obj = 50;
        double k_pose = 0.5;
        double k_ee = 1.0;
        double k_contact_r_ = 1.0, k_contact_r2_ = 1.0, k_contact_l_ = 1.0;
        double ray_length = 0.05;
        double num_active_contacts_r_, num_active_contacts_r2_;
        double num_active_contacts_l_;
        double impulse_reward_r_ = 0.0, impulse_reward_l_ = 0.0;
        double obj_angle_reward_ = 0.0, obj_avel_reward_ = 0.0;
        double obj_weight_t_, obj_weight_b_, obj_weight_;
        Eigen::VectorXd joint_limit_high_r, joint_limit_high_l, joint_limit_low_r, joint_limit_low_l, actionMean_r_, actionMean_l_, actionStd_r_, actionStd_l_, obDouble_r_, obDouble_l_, global_state_, rel_pose_r_, rel_pose_l_, finger_weights_, rel_obj_pos_r_, rel_obj_pos_l_, rel_objpalm_pos_r_, rel_objpalm_pos_l_, rel_body_pos_r_, rel_body_pos_l_, rel_contact_pos_r_, rel_contact_pos_l_, rel_contacts_r_, rel_contacts_l_, contacts_r_, contacts_l_, impulses_r_, impulses_l_, rel_obj_pose_r_, rel_obj_pose_l_;
        Eigen::VectorXd obj_body_pos_r_, obj_body_pos_l_;
        Eigen::Vector3d bodyLinearVel_r_, bodyLinearVel_l_, bodyLinearVel_l_obs_, bodyLinearVel_r_obs_, bodyAngularVel_r_, bodyAngularVel_l_, bodyAngularVel_l_obs_, bodyAngularVel_r_obs_, rel_obj_qvel_b, rel_obj_qvel_t, rel_obj_vel_b, rel_obj_vel_t, up_pose_r, up_pose_l, rel_body_table_pos_r_, rel_body_table_pos_l_;
        std::set<size_t> footIndices_;
        raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
        raisim::Cylinder *cylinder;
        raisim::Box *box_obj;
        raisim::Box *box;
        raisim::Box *stage;
        raisim::ArticulatedSystem *arctic, *mano_l_, *mano_r_;
        raisim::ArticulatedSystemVisual *arcticVisual;
        raisim::Vec<3> temp_torque;
        int rf_dim = 6;
        int num_obj = 4;
        int num_contacts = 16;
        int num_joint = 17;
        int num_bodyparts = 21;
        int root_guiding_counter_r_ = 0, root_guiding_counter_l_ = 0;
        int obj_idx_;
        int grasp_num_l = 0, grasp_num_r = 0;
        double pgain4 = 500.0;
        bool root_guided=false;
        bool root_guided_l = false;
        bool root_keep_l = false;
        bool root_keep_r = false;
        bool root_pregrasp_l = false;
        bool root_guided_r = false;
        bool cylinder_mesh=false;
        bool box_obj_mesh=false;
        bool first_reset_=true;
        bool no_pose_state = false;
        bool nohierarchy = false;
        bool contact_pruned = false;
        bool motion_synthesis = false;
        bool motion_synthesis_l = false;
        bool add_disturbance = false;
        float rewards_sum_[2];
        bool stage_flag = true;
        bool left_hold = true;
        raisim::Vec<3> pose_goal_r, pos_goal_r, up_vec_r, up_gen_vec_r, obj_pose_r_, Position_r, Obj_Position_b, Rel_fpos, Obj_linvel_b, Obj_qvel_b, Fpos_world_r, palm_world_pos_r_, init_root_r_, init_obj_;
        raisim::Vec<3> pose_goal_l, pos_goal_l, up_vec_l, up_gen_vec_l, obj_pose_l_, Position_l, Obj_Position_t,           Obj_linvel_t, Obj_qvel_t, Fpos_world_l, palm_world_pos_l_, init_root_l_;
        raisim::Vec<3> vel_l, vel_r, angular_vel_l, angular_vel_r, obj_trans_init_b;
        raisim::Vec<3> obj_euler_b_, obj_euler_t_, wrist_pos_l, init_rel_objpalm_pos_l, rel_obj_pos_b;
        raisim::Mat<3,3> wrist_ori_l, Obj_orientation_b, Obj_orientation_t, Obj_orientation_temp_b, Obj_orientation_temp_t, Body_orientation_r, Body_orientation_l, init_or_r_, init_or_l_, root_pose_world_r_, root_pose_world_l_, init_rot_r_, init_rot_l_, init_obj_rot_;
        raisim::Mat<3,3> Obj_orientation_set_b, Obj_orientation_set_t;
        raisim::Vec<3> obj_pos_set_t, obj_pos_set_b, rand_force, rand_torque, wrist_l_euler_world;
        double rand_counter, left_hold_counter;
        raisim::Mat<3,3> init_obj_ori_wrist_l;
        raisim::Vec<4> obj_quat_b, obj_quat_t;
        Eigen::VectorXd obj_angle_, obj_avel_, rel_obj_angle_;
        raisim::Vec<3> wrist_pos_obj_l_, wrist_pose_obj_l_, wrist_pos_obj_r_, wrist_pose_obj_r_;
        raisim::Vec<3> rel_obj_pose_world_b_, rel_obj_pose_world_t_, rel_obj_pose_wrist_b_, obj_pose_diff_wrist;
        raisim::Vec<3>  rel_obj_pos_wrist_l_, rel_obj_pos_wrist_r_;
        std::vector<int> contact_idxs_;
        Eigen::Vector3d stage_dim;
        Eigen::Vector3d stage_pos;
        std::string body_parts_r_[21] = {"right_wrist_0rz",
                                         "right_index1_x", "right_index2_x", "right_index3_x",
                                         "right_middle1_x", "right_middle2_x", "right_middle3_x",
                                         "right_pinky1_x", "right_pinky2_x", "right_pinky3_x",
                                         "right_ring1_x", "right_ring2_x", "right_ring3_x",
                                         "right_thumb1_x", "right_thumb2_x", "right_thumb3_x",
                                         "right_thumb_tip", "right_index_tip", "right_middle_tip", "right_ring_tip", "right_pinky_tip",
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
        std::vector<raisim::Vec<2>> joint_limits_r_, joint_limits_l_;
        raisim::PolyLine *line;
        const double pi_ = 3.14159265358979323846;
    };
}