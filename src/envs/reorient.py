# pylint: disable=attribute-defined-outside-init, dangerous-default-value, protected-access, abstract-method, arguments-renamed
import collections
import numpy as np
from myosuite.envs.env_base import MujocoEnv
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.myochallenge.reorient_v0 import ReorientEnvV0
from myosuite.utils.quat_math import euler2quat


class CustomReorientEnv(ReorientEnvV0):
    def get_reward_dict(self, obs_dict):
        pos_dist_new = np.abs(np.linalg.norm(self.obs_dict["pos_err"], axis=-1))
        rot_dist_new = np.abs(np.linalg.norm(self.obs_dict["rot_err"], axis=-1))
        pos_dist_diff = self.pos_dist - pos_dist_new
        rot_dist_diff = self.rot_dist - rot_dist_new
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )
        drop = pos_dist_new > self.drop_th

        rwd_dict = collections.OrderedDict(
            (
                # Perform reward tuning here --
                # Update Optional Keys section below
                # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
                # Examples: Env comes pre-packaged with two keys pos_dist and rot_dist
                # Optional Keys
                ("pos_dist", -1.0 * pos_dist_new),
                ("rot_dist", -1.0 * rot_dist_new),
                ("pos_dist_diff", pos_dist_diff),
                ("rot_dist_diff", rot_dist_diff),
                ("alive", ~drop),
                # Must keys
                ("act_reg", -1.0 * act_mag),
                ("sparse", -rot_dist_new - 10.0 * pos_dist_new),
                (
                    "solved",
                    (
                        (pos_dist_new < self.pos_th)
                        and (rot_dist_new < self.rot_th)
                        and (not drop)
                    )
                    * np.ones((1, 1)),
                ),
                ("done", drop),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )

        # Sucess Indicator
        self.sim.model.site_rgba[self.success_indicator_sid, :2] = (
            np.array([0, 2]) if rwd_dict["solved"] else np.array([2, 0])
        )
        return rwd_dict

    def _setup(
        self,
        obs_keys: list = ReorientEnvV0.DEFAULT_OBS_KEYS,
        weighted_reward_keys: list = ReorientEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS,
        goal_pos=(0.0, 0.0),  # goal position range (relative to initial pos)
        goal_rot=(0.785, 0.785),  # goal rotation range (relative to initial rot)
        obj_size_change=0,  # object size change (relative to initial size)
        obj_friction_change=(
            0,
            0,
            0,
        ),  # object friction change (relative to initial size)
        pos_th=0.025,  # position error threshold
        rot_th=0.262,  # rotation error threshold
        drop_th=0.200,  # drop height threshold
        enable_rsi=False,
        rsi_distance_pos=0,
        rsi_distance_rot=0,
        goal_rot_x=None,
        goal_rot_y=None,
        goal_rot_z=None,
        guided_trajectory_steps=None,
        **kwargs,
    ):
        self.already_reset = False
        self.object_sid = self.sim.model.site_name2id("object_o")
        self.goal_sid = self.sim.model.site_name2id("target_o")
        self.success_indicator_sid = self.sim.model.site_name2id("target_ball")
        self.goal_bid = self.sim.model.body_name2id("target")
        self.object_bid = self.sim.model.body_name2id("Object")
        self.goal_init_pos = self.sim.data.site_xpos[self.goal_sid].copy()
        self.goal_init_rot = self.sim.model.body_quat[self.goal_bid].copy()
        self.goal_obj_offset = (
            self.sim.data.site_xpos[self.goal_sid]
            - self.sim.data.site_xpos[self.object_sid]
        )  # visualization offset between target and object
        self.goal_pos = goal_pos
        self.goal_rot = goal_rot
        self.pos_th = pos_th
        self.rot_th = rot_th
        self.drop_th = drop_th
        self.rsi = enable_rsi
        self.rsi_distance_pos = rsi_distance_pos
        self.rsi_distance_rot = rsi_distance_rot
        self.goal_rot_x = goal_rot_x
        self.goal_rot_y = goal_rot_y
        self.goal_rot_z = goal_rot_z
        self.guided_trajectory_steps = guided_trajectory_steps
        self.pos_dist = 0
        self.rot_dist = 0

        # setup for object randomization
        self.target_gid = self.sim.model.geom_name2id("target_dice")
        self.target_default_size = self.sim.model.geom_size[self.target_gid].copy()

        object_bid = self.sim.model.body_name2id("Object")
        self.object_gid0 = self.sim.model.body_geomadr[object_bid]
        self.object_gidn = self.object_gid0 + self.sim.model.body_geomnum[object_bid]
        self.object_default_size = self.sim.model.geom_size[
            self.object_gid0 : self.object_gidn
        ].copy()
        self.object_default_pos = self.sim.model.geom_pos[
            self.object_gid0 : self.object_gidn
        ].copy()

        self.obj_size_change = {"high": obj_size_change, "low": -obj_size_change}
        self.obj_friction_range = {
            "high": self.sim.model.geom_friction[self.object_gid0 : self.object_gidn]
            + obj_friction_change,
            "low": self.sim.model.geom_friction[self.object_gid0 : self.object_gidn]
            - obj_friction_change,
        }

        BaseV0._setup(
            self,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            **kwargs,
        )
        self.init_qpos[:-7] *= 0  # Use fully open as init pos
        self.init_qpos[0] = -1.5  # Palm up

    def reset(self, reset_qpos=None, reset_qvel=None):

        # First sample the target position and orientation of the die
        self.episode_goal_pos = self.sample_goal_position()
        self.episode_goal_rot = self.sample_goal_orientation()

        # Then get the initial position and orientation of the die
        if self.rsi:
            object_init_pos = (
                self.rsi_distance_pos * self.goal_init_pos
                + (1 - self.rsi_distance_pos) * self.episode_goal_pos
                - self.goal_obj_offset
            )
            object_init_rot = (
                self.rsi_distance_rot * self.goal_init_rot
                + (1 - self.rsi_distance_rot) * self.episode_goal_rot
            )
        else:
            object_init_pos = self.goal_init_pos - self.goal_obj_offset
            object_init_rot = self.goal_init_rot

        # Set the position of the object
        self.sim.model.body_pos[self.object_bid] = object_init_pos
        self.sim.model.body_quat[self.object_bid] = object_init_rot

        # Create the target trajectory and set the initial position
        self.goal_pos_traj, self.goal_rot_traj = self.create_goal_trajectory(
            object_init_pos, object_init_rot
        )
        self.counter = 0
        self.set_die_pos_rot(self.counter)

        # Die friction changes
        self.sim.model.geom_friction[
            self.object_gid0 : self.object_gidn
        ] = self.np_random.uniform(**self.obj_friction_range)

        # Die and Target size changes
        del_size = self.np_random.uniform(**self.obj_size_change)
        # adjust size of target
        self.sim.model.geom_size[self.target_gid] = self.target_default_size + del_size
        # adjust size of die
        self.sim.model.geom_size[self.object_gid0 : self.object_gidn - 3][:, 1] = (
            self.object_default_size[:-3][:, 1] + del_size
        )
        self.sim.model.geom_size[self.object_gidn - 3 : self.object_gidn] = (
            self.object_default_size[-3:] + del_size
        )
        # adjust boundary of die
        object_gpos = self.sim.model.geom_pos[self.object_gid0 : self.object_gidn]
        self.sim.model.geom_pos[self.object_gid0 : self.object_gidn] = (
            object_gpos
            / abs(object_gpos + 1e-16)
            * (abs(self.object_default_pos) + del_size)
        )

        obs = MujocoEnv.reset(self, reset_qpos, reset_qvel)
        self.pos_dist = np.abs(np.linalg.norm(self.obs_dict["pos_err"], axis=-1))
        self.rot_dist = np.abs(np.linalg.norm(self.obs_dict["rot_err"], axis=-1))
        self.already_reset = True
        return obs

    def sample_goal_position(self):
        goal_pos = self.goal_init_pos + self.np_random.uniform(
            high=self.goal_pos[1], low=self.goal_pos[0], size=3
        )
        return goal_pos

    def sample_goal_orientation(self):
        x_low, x_high = (
            self.goal_rot_x[self.np_random.choice(len(self.goal_rot_x))]
            if self.goal_rot_x is not None
            else self.goal_rot
        )
        y_low, y_high = (
            self.goal_rot_y[self.np_random.choice(len(self.goal_rot_y))]
            if self.goal_rot_y is not None
            else self.goal_rot
        )
        z_low, z_high = (
            self.goal_rot_z[self.np_random.choice(len(self.goal_rot_z))]
            if self.goal_rot_z is not None
            else self.goal_rot
        )

        goal_rot_x = self.np_random.uniform(x_low, x_high)
        goal_rot_y = self.np_random.uniform(y_low, y_high)
        goal_rot_z = self.np_random.uniform(z_low, z_high)
        goal_rot_quat = euler2quat(np.array([goal_rot_x, goal_rot_y, goal_rot_z]))
        return goal_rot_quat

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.pos_dist = np.abs(np.linalg.norm(self.obs_dict["pos_err"], axis=-1))
        self.rot_dist = np.abs(np.linalg.norm(self.obs_dict["rot_err"], axis=-1))
        info.update(info.get("rwd_dict"))

        if self.already_reset:
            self.counter += 1
            self.set_die_pos_rot(self.counter)
        return obs, reward, done, info

    def create_goal_trajectory(self, object_init_pos, object_init_rot):
        traj_len = 1000  # Assumes it is larger than the episode len

        pos_traj = np.ones((traj_len, 3)) * self.episode_goal_pos
        rot_traj = np.ones((traj_len, 4)) * self.episode_goal_rot

        if (
            self.guided_trajectory_steps is not None
        ):  # Softly reach the target position and orientation
            goal_init_pos = object_init_pos + self.goal_obj_offset
            guided_pos_traj = np.linspace(
                goal_init_pos, self.episode_goal_pos, self.guided_trajectory_steps
            )
            pos_traj[: self.guided_trajectory_steps] = guided_pos_traj

            guided_rot_traj = np.linspace(
                object_init_rot, self.episode_goal_rot, self.guided_trajectory_steps
            )
            rot_traj[: self.guided_trajectory_steps] = guided_rot_traj
        return pos_traj, rot_traj

    def set_die_pos_rot(self, counter):
        self.sim.model.body_pos[self.goal_bid] = self.goal_pos_traj[counter]
        self.sim.model.body_quat[self.goal_bid] = self.goal_rot_traj[counter]
