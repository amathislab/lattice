import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.pose_v0 import PoseEnvV0


class CustomPoseEnv(PoseEnvV0):
    def _setup(
        self,
        viz_site_targets: tuple = None,  # site to use for targets visualization []
        target_jnt_range: dict = None,  # joint ranges as tuples {name:(min, max)}_nq
        target_jnt_value: list = None,  # desired joint vector [des_qpos]_nq
        reset_type="init",  # none; init; random; sds
        target_type="generate",  # generate; switch; fixed
        obs_keys: list = PoseEnvV0.DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = PoseEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS,
        pose_thd=0.35,
        weight_bodyname=None,
        weight_range=None,
        sds_distance=0,
        target_distance=1,  # for non-SDS curriculum, the target is set at a fraction of the full distance
        **kwargs,
    ):
        self.reset_type = reset_type
        self.target_type = target_type
        self.pose_thd = pose_thd
        self.weight_bodyname = weight_bodyname
        self.weight_range = weight_range
        self.sds_distance = sds_distance
        self.target_distance = target_distance

        # resolve joint demands
        if target_jnt_range:
            self.target_jnt_ids = []
            self.target_jnt_range = []
            for jnt_name, jnt_range in target_jnt_range.items():
                self.target_jnt_ids.append(self.sim.model.joint_name2id(jnt_name))
                self.target_jnt_range.append(jnt_range)
            self.target_jnt_range = np.array(self.target_jnt_range)
            self.target_jnt_value = np.mean(
                self.target_jnt_range, axis=1
            )  # pseudo targets for init
        else:
            self.target_jnt_value = target_jnt_value

        BaseV0._setup(
            self,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            sites=viz_site_targets,
            **kwargs,
        )

    def reset(self):
        # udpate wegith
        if self.weight_bodyname is not None:
            bid = self.sim.model.body_name2id(self.weight_bodyname)
            gid = self.sim.model.body_geomadr[bid]
            weight = self.np_random.uniform(
                low=self.weight_range[0], high=self.weight_range[1]
            )
            self.sim.model.body_mass[bid] = weight
            self.sim_obsd.model.body_mass[bid] = weight
            # self.sim_obsd.model.geom_size[gid] = self.sim.model.geom_size[gid] * weight/10
            self.sim.model.geom_size[gid][0] = 0.01 + 2.5 * weight / 100
            # self.sim_obsd.model.geom_size[gid][0] = weight/10

        # update target
        if self.target_type == "generate":
            # use target_jnt_range to generate targets
            self.update_target(restore_sim=True)
        elif self.target_type == "fixed":
            self.update_target(restore_sim=True)
        else:
            print("{} Target Type not found ".format(self.target_type))

        # update init state
        if self.reset_type is None or self.reset_type == "none":
            # no reset; use last state
            obs = self.get_obs()
        elif self.reset_type == "init":
            # reset to init state
            obs = BaseV0.reset(self)
        elif self.reset_type == "random":
            # reset to random state
            jnt_init = self.np_random.uniform(
                high=self.sim.model.jnt_range[:, 1], low=self.sim.model.jnt_range[:, 0]
            )
            obs = BaseV0.reset(self, reset_qpos=jnt_init)
        elif self.reset_type == "sds":
            init_qpos = self.init_qpos.copy()
            init_qvel = self.init_qvel.copy()
            target_qpos = self.target_jnt_value.copy()
            qpos = (1 - self.sds_distance) * target_qpos + self.sds_distance * init_qpos
            self.robot.reset(qpos, init_qvel)
            obs = self.get_obs()
        else:
            print("Reset Type not found")

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info.update(info.get("rwd_dict"))
        return obs, reward, done, info

    def render(self, mode):
        return self.sim.render(mode=mode)

    def get_target_pose(self):
        full_distance_target_pose = super().get_target_pose()
        init_pose = self.init_qpos.copy()
        target_pose = init_pose + self.target_distance * (
            full_distance_target_pose - init_pose
        )
        return target_pose
