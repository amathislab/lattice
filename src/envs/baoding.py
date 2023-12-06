# pylint: disable=attribute-defined-outside-init, dangerous-default-value, protected-access, abstract-method, arguments-renamed, import-error
import collections
import random
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.myochallenge.baoding_v1 import WHICH_TASK, BaodingEnvV1, Task
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from envs.environment_factory import EnvironmentFactory


class CustomBaodingEnv(BaodingEnvV1):
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_dist_1": 5.0,
        "pos_dist_2": 5.0,
        "alive": 0.0,
        "act_reg": 0.0,
        # "palm_up": 0.0,
    }

    def get_reward_dict(self, obs_dict):
        # tracking error
        target1_dist = np.linalg.norm(obs_dict["target1_err"], axis=-1)
        target2_dist = np.linalg.norm(obs_dict["target2_err"], axis=-1)
        target_dist = target1_dist + target2_dist
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )

        # detect fall
        object1_pos = (
            obs_dict["object1_pos"][:, :, 2]
            if obs_dict["object1_pos"].ndim == 3
            else obs_dict["object1_pos"][2]
        )
        object2_pos = (
            obs_dict["object2_pos"][:, :, 2]
            if obs_dict["object2_pos"].ndim == 3
            else obs_dict["object2_pos"][2]
        )
        is_fall_1 = object1_pos < self.drop_th
        is_fall_2 = object2_pos < self.drop_th
        is_fall = np.logical_or(is_fall_1, is_fall_2)  # keep both balls up

        rwd_dict = collections.OrderedDict(
            (
                # Perform reward tuning here --
                # Update Optional Keys section below
                # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) to update final rewards
                # Examples: Env comes pre-packaged with two keys pos_dist_1 and pos_dist_2
                # Optional Keys
                ("pos_dist_1", -1.0 * target1_dist),
                ("pos_dist_2", -1.0 * target2_dist),
                ("alive", ~is_fall),
                # ("palm_up", palm_up_reward),
                # Must keys
                ("act_reg", -1.0 * act_mag),
                ("sparse", -target_dist),
                (
                    "solved",
                    (target1_dist < self.proximity_th)
                    * (target2_dist < self.proximity_th)
                    * (~is_fall),
                ),
                ("done", is_fall),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )

        # Sucess Indicator
        self.sim.model.geom_rgba[self.object1_gid, :2] = (
            np.array([1, 1])
            if target1_dist < self.proximity_th
            else np.array([0.5, 0.5])
        )
        self.sim.model.geom_rgba[self.object2_gid, :2] = (
            np.array([0.9, 0.7])
            if target1_dist < self.proximity_th
            else np.array([0.5, 0.5])
        )

        return rwd_dict

    def _add_noise_to_palm_position(
        self, qpos: np.ndarray, noise: float = 1
    ) -> np.ndarray:
        assert 0 <= noise <= 1, "Noise must be between 0 and 1"

        # pronation-supination of the wrist
        # noise = 1 corresponds to 10 degrees from facing up (one direction only)
        qpos[0] = self.np_random.uniform(
            low=-np.pi / 2, high=-np.pi / 2 + np.pi / 18 * noise
        )

        # ulnar deviation of wrist:
        # noise = 1 corresponds to 10 degrees on either side
        qpos[1] = self.np_random.uniform(
            low=-np.pi / 18 * noise, high=np.pi / 18 * noise
        )

        # extension flexion of the wrist
        # noise = 1 corresponds to 10 degrees on either side
        qpos[2] = self.np_random.uniform(
            low=-np.pi / 18 * noise, high=np.pi / 18 * noise
        )

        return qpos

    def _add_noise_to_finger_positions(
        self, qpos: np.ndarray, noise: float = 1
    ) -> np.ndarray:
        assert 0 <= noise <= 1, "Noise parameter must be between 0 and 1"

        # thumb all joints
        # noise = 1 corresponds to 10 degrees on either side
        qpos[3:7] = self.np_random.uniform(
            low=-np.pi / 18 * noise, high=np.pi / 18 * noise
        )

        # finger joints
        # noise = 1 corresponds to 30 degrees bent instead of fully open
        qpos[[7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22]] = self.np_random.uniform(
            low=0, high=np.pi / 6 * noise
        )

        # finger abduction (sideways angle)
        # noise = 1 corresponds to 5 degrees on either side
        qpos[[8, 12, 16, 20]] = self.np_random.uniform(
            low=-np.pi / 36 * noise, high=np.pi / 36 * noise
        )

        return qpos

    def reset(self, reset_pose=None, reset_vel=None, reset_goal=None, time_period=None):
        self.which_task = self.sample_task()
        if self.rsi:
            # MODIFICATION: randomize starting target position along the cycle
            random_phase = np.random.uniform(low=-np.pi, high=np.pi)
        else:
            random_phase = 0
        self.ball_1_starting_angle = 3.0 * np.pi / 4.0 + random_phase
        self.ball_2_starting_angle = -1.0 * np.pi / 4.0 + random_phase

        # reset counters
        self.counter = 0
        self.x_radius = self.np_random.uniform(
            low=self.goal_xrange[0], high=self.goal_xrange[1]
        )
        self.y_radius = self.np_random.uniform(
            low=self.goal_yrange[0], high=self.goal_yrange[1]
        )

        # reset goal
        if time_period is None:
            time_period = self.np_random.uniform(
                low=self.goal_time_period[0], high=self.goal_time_period[1]
            )
        self.goal = (
            self.create_goal_trajectory(time_step=self.dt, time_period=time_period)
            if reset_goal is None
            else reset_goal.copy()
        )

        # reset scene (MODIFIED from base class MujocoEnv)
        qpos = self.init_qpos.copy() if reset_pose is None else reset_pose
        qvel = self.init_qvel.copy() if reset_vel is None else reset_vel
        self.robot.reset(qpos, qvel)

        if self.rsi:
            if np.random.uniform(0, 1) < self.rsi_probability:
                self.step(np.zeros(39))

                # update ball positions
                obs = self.get_obs().copy()
                qpos[23] = obs[35]  # ball 1 x-position
                qpos[24] = obs[36]  # ball 1 y-position
                qpos[30] = obs[38]  # ball 2 x-position
                qpos[31] = obs[39]  # ball 2 y-position

        if self.noise_balls:
            # update balls x,y,z positions with relative noise
            for i in [23, 24, 25, 30, 31, 32]:
                qpos[i] += np.random.uniform(
                    low=-self.noise_balls, high=self.noise_balls
                )

        if self.noise_palm:
            qpos = self._add_noise_to_palm_position(qpos, self.noise_palm)

        if self.noise_fingers:
            qpos = self._add_noise_to_finger_positions(qpos, self.noise_fingers)

        if self.rsi or self.noise_palm or self.noise_fingers or self.noise_balls:
            self.set_state(qpos, qvel)

        return self.get_obs()

    def _setup(
        self,
        frame_skip: int = 10,
        drop_th=1.25,  # drop height threshold
        proximity_th=0.015,  # object-target proximity threshold
        goal_time_period=(5, 5),  # target rotation time period
        goal_xrange=(0.025, 0.025),  # target rotation: x radius (0.03)
        goal_yrange=(0.028, 0.028),  # target rotation: x radius (0.02 * 1.5 * 1.2)
        obs_keys: list = BaodingEnvV1.DEFAULT_OBS_KEYS,
        weighted_reward_keys: list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        task=None,
        enable_rsi=False,  # random state init for balls
        noise_palm=0,  # magnitude of noise for palm (between 0 and 1)
        noise_fingers=0,  # magnitude of noise for fingers (between 0 and 1)
        noise_balls=0,  # relative magnitude of noise for the balls (1 is 100% relative noise)
        rsi_probability=1,  # probability of implementing RSI
        **kwargs,
    ):

        # user parameters
        self.task = task
        self.which_task = self.sample_task()
        self.rsi = enable_rsi
        self.noise_palm = noise_palm
        self.noise_fingers = noise_fingers
        self.drop_th = drop_th
        self.proximity_th = proximity_th
        self.goal_time_period = goal_time_period
        self.goal_xrange = goal_xrange
        self.goal_yrange = goal_yrange
        self.noise_balls = noise_balls
        self.rsi_probability = rsi_probability

        # balls start at these angles
        #   1= yellow = top right
        #   2= pink = bottom left
        self.ball_1_starting_angle = 3.0 * np.pi / 4.0
        self.ball_2_starting_angle = -1.0 * np.pi / 4.0

        # init desired trajectory, for rotations
        self.center_pos = [-0.0125, -0.07]  # [-.0020, -.0522]
        self.x_radius = self.np_random.uniform(
            low=self.goal_xrange[0], high=self.goal_xrange[1]
        )
        self.y_radius = self.np_random.uniform(
            low=self.goal_yrange[0], high=self.goal_yrange[1]
        )

        self.counter = 0
        self.goal = self.create_goal_trajectory(
            time_step=frame_skip * self.sim.model.opt.timestep, time_period=6
        )

        # init target and body sites
        self.object1_sid = self.sim.model.site_name2id("ball1_site")
        self.object2_sid = self.sim.model.site_name2id("ball2_site")
        self.object1_gid = self.sim.model.geom_name2id("ball1")
        self.object2_gid = self.sim.model.geom_name2id("ball2")
        self.target1_sid = self.sim.model.site_name2id("target1_site")
        self.target2_sid = self.sim.model.site_name2id("target2_site")
        self.sim.model.site_group[self.target1_sid] = 2
        self.sim.model.site_group[self.target2_sid] = 2

        BaseV0._setup(
            self,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            frame_skip=frame_skip,
            **kwargs,
        )

        # reset position
        self.init_qpos[:-14] *= 0  # Use fully open as init pos
        self.init_qpos[0] = -1.57  # Palm up

    def sample_task(self):
        if self.task is None:
            return Task(WHICH_TASK)
        else:
            if self.task == "cw":
                return Task(Task.BAODING_CW)
            elif self.task == "ccw":
                return Task(Task.BAODING_CCW)
            elif self.task == "random":
                return Task(random.choice(list(Task)))
            else:
                raise ValueError("Unknown task for baoding: ", self.task)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info.update(info.get("rwd_dict"))
        return obs, reward, done, info


class CustomBaodingP2Env(BaodingEnvV1):
    def _setup(
        self,
        frame_skip: int = 10,
        drop_th=1.25,  # drop height threshold
        proximity_th=0.015,  # object-target proximity threshold
        goal_time_period=(5, 5),  # target rotation time period
        goal_xrange=(0.025, 0.025),  # target rotation: x radius (0.03)
        goal_yrange=(0.028, 0.028),  # target rotation: x radius (0.02 * 1.5 * 1.2)
        obj_size_range=(0.018, 0.024),  # Object size range. Nominal 0.022
        obj_mass_range=(0.030, 0.300),  # Object weight range. Nominal 43 gms
        obj_friction_change=(0.2, 0.001, 0.00002),
        task_choice="fixed",  # fixed/ random
        obs_keys: list = BaodingEnvV1.DEFAULT_OBS_KEYS,
        weighted_reward_keys: list = BaodingEnvV1.DEFAULT_RWD_KEYS_AND_WEIGHTS,
        enable_rsi=False,  # random state init for balls
        rsi_probability=1,  # probability of implementing RSI
        balls_overlap=False,
        overlap_probability=0,
        limit_init_angle=None,
        beta_init_angle=None,
        beta_ball_size=None,
        beta_ball_mass=None,
        noise_fingers=0,
        **kwargs,
    ):
        # user parameters
        self.task_choice = task_choice
        self.which_task = (
            self.np_random.choice(Task) if task_choice == "random" else Task(WHICH_TASK)
        )
        self.drop_th = drop_th
        self.proximity_th = proximity_th
        self.goal_time_period = goal_time_period
        self.goal_xrange = goal_xrange
        self.goal_yrange = goal_yrange
        self.rsi = enable_rsi
        self.rsi_probability = rsi_probability
        self.balls_overlap = balls_overlap
        self.overlap_probability = overlap_probability
        self.noise_fingers = noise_fingers
        self.limit_init_angle = limit_init_angle
        self.beta_init_angle = beta_init_angle
        self.beta_ball_size = beta_ball_size
        self.beta_ball_mass = beta_ball_mass

        # balls start at these angles
        #   1= yellow = top right
        #   2= pink = bottom left

        if np.random.uniform(0, 1) < self.overlap_probability:
            self.ball_1_starting_angle = 3.0 * np.pi / 4.0
            self.ball_2_starting_angle = -1.0 * np.pi / 4.0
        else:
            self.ball_1_starting_angle = 1.0 * np.pi / 4.0
            self.ball_2_starting_angle = self.ball_1_starting_angle - np.pi

        # init desired trajectory, for rotations
        self.center_pos = [-0.0125, -0.07]  # [-.0020, -.0522]
        self.x_radius = self.np_random.uniform(
            low=self.goal_xrange[0], high=self.goal_xrange[1]
        )
        self.y_radius = self.np_random.uniform(
            low=self.goal_yrange[0], high=self.goal_yrange[1]
        )

        self.counter = 0
        self.goal = self.create_goal_trajectory(
            time_step=frame_skip * self.sim.model.opt.timestep, time_period=6
        )

        # init target and body sites
        self.object1_bid = self.sim.model.body_name2id("ball1")
        self.object2_bid = self.sim.model.body_name2id("ball2")
        self.object1_sid = self.sim.model.site_name2id("ball1_site")
        self.object2_sid = self.sim.model.site_name2id("ball2_site")
        self.object1_gid = self.sim.model.geom_name2id("ball1")
        self.object2_gid = self.sim.model.geom_name2id("ball2")
        self.target1_sid = self.sim.model.site_name2id("target1_site")
        self.target2_sid = self.sim.model.site_name2id("target2_site")
        self.sim.model.site_group[self.target1_sid] = 2
        self.sim.model.site_group[self.target2_sid] = 2

        # setup for task randomization
        self.obj_mass_range = {"low": obj_mass_range[0], "high": obj_mass_range[1]}
        self.obj_size_range = {"low": obj_size_range[0], "high": obj_size_range[1]}
        self.obj_friction_range = {
            "low": self.sim.model.geom_friction[self.object1_gid] - obj_friction_change,
            "high": self.sim.model.geom_friction[self.object1_gid]
            + obj_friction_change,
        }

        BaseV0._setup(
            self,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            frame_skip=frame_skip,
            **kwargs,
        )

        # reset position
        self.init_qpos[:-14] *= 0  # Use fully open as init pos
        self.init_qpos[0] = -1.57  # Palm up

    def get_reward_dict(self, obs_dict):
        # tracking error
        target1_dist = np.linalg.norm(obs_dict["target1_err"], axis=-1)
        target2_dist = np.linalg.norm(obs_dict["target2_err"], axis=-1)
        target_dist = target1_dist + target2_dist
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )

        # detect fall
        object1_pos = (
            obs_dict["object1_pos"][:, :, 2]
            if obs_dict["object1_pos"].ndim == 3
            else obs_dict["object1_pos"][2]
        )
        object2_pos = (
            obs_dict["object2_pos"][:, :, 2]
            if obs_dict["object2_pos"].ndim == 3
            else obs_dict["object2_pos"][2]
        )
        is_fall_1 = object1_pos < self.drop_th
        is_fall_2 = object2_pos < self.drop_th
        is_fall = np.logical_or(is_fall_1, is_fall_2)  # keep both balls up

        rwd_dict = collections.OrderedDict(
            (
                # Perform reward tuning here --
                # Update Optional Keys section below
                # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
                # Examples: Env comes pre-packaged with two keys pos_dist_1 and pos_dist_2
                # Optional Keys
                ("pos_dist_1", -1.0 * target1_dist),
                ("pos_dist_2", -1.0 * target2_dist),
                # Must keys
                ("act_reg", -1.0 * act_mag),
                ("alive", ~is_fall),
                ("sparse", -target_dist),
                (
                    "solved",
                    (target1_dist < self.proximity_th)
                    * (target2_dist < self.proximity_th)
                    * (~is_fall),
                ),
                ("done", is_fall),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )

        # Sucess Indicator
        self.sim.model.geom_rgba[self.object1_gid, :2] = (
            np.array([1, 1])
            if target1_dist < self.proximity_th
            else np.array([0.5, 0.5])
        )
        self.sim.model.geom_rgba[self.object2_gid, :2] = (
            np.array([0.9, 0.7])
            if target1_dist < self.proximity_th
            else np.array([0.5, 0.5])
        )

        return rwd_dict

    def _add_noise_to_finger_positions(
        self, qpos: np.ndarray, noise: float = 1
    ) -> np.ndarray:
        assert 0 <= noise <= 1, "Noise parameter must be between 0 and 1"

        # thumb all joints
        qpos[4:7] = self.np_random.uniform(
            low=-np.pi / 18 * noise, high=np.pi / 18 * noise
        )

        # finger joints
        qpos[[7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22]] = self.np_random.uniform(
            low=0, high=np.pi / 6 * noise
        )

        return qpos

    def reset(self, reset_pose=None, reset_vel=None, reset_goal=None, time_period=None):

        # reset task
        if self.task_choice == "random":
            self.which_task = self.np_random.choice(Task)

            if np.random.uniform(0, 1) <= self.overlap_probability:
                self.ball_1_starting_angle = 3.0 * np.pi / 4.0
            elif self.limit_init_angle is not None:
                random_phase = self.np_random.uniform(
                    low=-self.limit_init_angle, high=self.limit_init_angle
                )

                if self.beta_init_angle is not None:
                    random_phase = (
                        self.np_random.beta(
                            self.beta_init_angle[0], self.beta_init_angle[1]
                        )
                        * 2
                        * np.pi
                        - np.pi
                    )

                self.ball_1_starting_angle = 3.0 * np.pi / 4.0 + random_phase
            else:
                self.ball_1_starting_angle = self.np_random.uniform(
                    low=0, high=2 * np.pi
                )

            self.ball_2_starting_angle = self.ball_1_starting_angle - np.pi
        # reset counters
        self.counter = 0
        self.x_radius = self.np_random.uniform(
            low=self.goal_xrange[0], high=self.goal_xrange[1]
        )
        self.y_radius = self.np_random.uniform(
            low=self.goal_yrange[0], high=self.goal_yrange[1]
        )

        # reset goal
        if time_period is None:
            time_period = self.np_random.uniform(
                low=self.goal_time_period[0], high=self.goal_time_period[1]
            )
        self.goal = (
            self.create_goal_trajectory(time_step=self.dt, time_period=time_period)
            if reset_goal is None
            else reset_goal.copy()
        )

        # balls mass changes
        self.sim.model.body_mass[self.object1_bid] = self.np_random.uniform(
            **self.obj_mass_range
        )  # call to mj_setConst(m,d) is being ignored. Derive quantities wont be updated. Die is simple shape. So this is reasonable approximation.
        self.sim.model.body_mass[self.object2_bid] = self.np_random.uniform(
            **self.obj_mass_range
        )  # call to mj_setConst(m,d) is being ignored. Derive quantities wont be updated. Die is simple shape. So this is reasonable approximation.

        if self.beta_ball_mass is not None:
            self.sim.model.body_mass[self.object1_bid] = (
                self.np_random.beta(self.beta_ball_mass[0], self.beta_ball_mass[1])
                * (self.obj_mass_range["high"] - self.obj_mass_range["low"])
                + self.obj_mass_range["low"]
            )
            self.sim.model.body_mass[self.object2_bid] = (
                self.np_random.beta(self.beta_ball_mass[0], self.beta_ball_mass[1])
                * (self.obj_mass_range["high"] - self.obj_mass_range["low"])
                + self.obj_mass_range["low"]
            )
        # balls friction changes
        self.sim.model.geom_friction[self.object1_gid] = self.np_random.uniform(
            **self.obj_friction_range
        )
        self.sim.model.geom_friction[self.object2_gid] = self.np_random.uniform(
            **self.obj_friction_range
        )

        # balls size changes
        self.sim.model.geom_size[self.object1_gid] = self.np_random.uniform(
            **self.obj_size_range
        )
        self.sim.model.geom_size[self.object2_gid] = self.np_random.uniform(
            **self.obj_size_range
        )

        if self.beta_ball_size is not None:
            self.sim.model.geom_size[self.object1_gid] = (
                self.np_random.beta(self.beta_ball_size[0], self.beta_ball_size[1])
                * (self.obj_size_range["high"] - self.obj_size_range["low"])
                + self.obj_size_range["low"]
            )
            self.sim.model.geom_size[self.object2_gid] = (
                self.np_random.beta(self.beta_ball_size[0], self.beta_ball_size[1])
                * (self.obj_size_range["high"] - self.obj_size_range["low"])
                + self.obj_size_range["low"]
            )
        # reset scene
        qpos = self.init_qpos.copy() if reset_pose is None else reset_pose
        qvel = self.init_qvel.copy() if reset_vel is None else reset_vel
        self.robot.reset(qpos, qvel)

        if self.rsi and np.random.uniform(0, 1) < self.rsi_probability:
            random_phase = np.random.uniform(low=-np.pi, high=np.pi)
            self.ball_1_starting_angle = 3.0 * np.pi / 4.0 + random_phase
            self.ball_2_starting_angle = -1.0 * np.pi / 4.0 + random_phase
            
            # # reset scene (MODIFIED from base class MujocoEnv)
            self.robot.reset(qpos, qvel)
            self.step(np.zeros(39))
            # update ball positions
            obs_dict = self.get_obs_dict(self.sim)
            target_1_pos = obs_dict["target1_pos"]
            target_2_pos = obs_dict["target2_pos"]
            qpos[23] = target_1_pos[0]  # ball 1 x-position
            qpos[24] = target_1_pos[1]  # ball 1 y-position
            qpos[30] = target_2_pos[0]  # ball 2 x-position
            qpos[31] = target_2_pos[1]  # ball 2 y-position
            self.set_state(qpos, qvel)

            if self.balls_overlap is False:
                self.ball_1_starting_angle = self.np_random.uniform(
                    low=0, high=2 * np.pi
                )
                self.ball_2_starting_angle = self.ball_1_starting_angle - np.pi

        if self.noise_fingers is not None:
            qpos = self._add_noise_to_finger_positions(qpos, self.noise_fingers)
            self.set_state(qpos, qvel)

        return self.get_obs()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info.update(info.get("rwd_dict"))
        return obs, reward, done, info
