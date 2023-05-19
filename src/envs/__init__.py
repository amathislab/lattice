import os
import gym
import myosuite
import numpy as np
from definitions import ROOT_DIR  # pylint: disable=import-error
from myosuite.envs.myo import register_env_with_variants


myosuite_path = os.path.join(ROOT_DIR, "data", "myosuite")

# MyoChallenge Baoding: Phase1 env
gym.envs.registration.register(
    id="CustomMyoChallengeBaodingP1-v1",
    entry_point="envs.baoding:CustomBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": myosuite_path + "/assets/hand/myo_hand_baoding.mjb",
        "normalize_act": True,
        # 'goal_time_period': (5, 5),
        "goal_xrange": (0.025, 0.025),
        "goal_yrange": (0.028, 0.028),
    },
)

# MyoChallenge Die: Phase2 env
gym.envs.registration.register(
    id="CustomMyoChallengeDieReorientP2-v0",
    entry_point="envs.reorient:CustomReorientEnv",
    max_episode_steps=150,
    kwargs={
        "model_path": myosuite_path + "/assets/hand/myo_hand_die.mjb",
        "normalize_act": True,
        "frame_skip": 5,
        # Randomization in goals
        'goal_pos': (-.020, .020),  # +- 2 cm
        'goal_rot': (-3.14, 3.14),   # +-180 degrees
        # Randomization in physical properties of the die
        'obj_size_change': 0.007, # +-7mm delta change in object size
        'obj_friction_change': (0.2, 0.001, 0.00002) # nominal: 1.0, 0.005, 0.0001
    },
)

register_env_with_variants(id='CustomMyoElbowPoseRandom-v0',
        entry_point='envs.pose:CustomPoseEnv',
        max_episode_steps=100,
        kwargs={
            'model_path': myosuite_path+'/assets/arm/myo_elbow_1dof6muscles.mjb',
            'target_jnt_range': {'r_elbow_flex':(0, 2.27),},
            'viz_site_targets': ('wrist',),
            'normalize_act': True,
            'pose_thd': .175,
            'reset_type': 'random'
        }
    )

register_env_with_variants(id='CustomMyoFingerPoseRandom-v0',
        entry_point='envs.pose:CustomPoseEnv',
        max_episode_steps=100,
        kwargs={
            'model_path': myosuite_path + '/assets/finger/myo_finger_v0.mjb',
            'target_jnt_range': {'IFadb':(-.2, .2),
                                'IFmcp':(-.4, 1),
                                'IFpip':(.1, 1),
                                'IFdip':(.1, 1)
                                },
            'viz_site_targets': ('IFtip',),
            'normalize_act': True,
        }
    )

# Hand-Joint posing ==============================
# Create ASL envs ==============================
jnt_namesHand=['pro_sup', 'deviation', 'flexion', 'cmc_abduction', 'cmc_flexion', 'mp_flexion', 'ip_flexion', 'mcp2_flexion', 'mcp2_abduction', 'pm2_flexion', 'md2_flexion', 'mcp3_flexion', 'mcp3_abduction', 'pm3_flexion', 'md3_flexion', 'mcp4_flexion', 'mcp4_abduction', 'pm4_flexion', 'md4_flexion', 'mcp5_flexion', 'mcp5_abduction', 'pm5_flexion', 'md5_flexion']

ASL_qpos={}
ASL_qpos[0]='0 0 0 0.5624 0.28272 -0.75573 -1.309 1.30045 -0.006982 1.45492 0.998897 1.26466 0 1.40604 0.227795 1.07614 -0.020944 1.46103 0.06284 0.83263 -0.14399 1.571 1.38248'.split(' ')
ASL_qpos[1]='0 0 0 0.0248 0.04536 -0.7854 -1.309 0.366605 0.010473 0.269258 0.111722 1.48459 0 1.45318 1.44532 1.44532 -0.204204 1.46103 1.44532 1.48459 -0.2618 1.47674 1.48459'.split(' ')
ASL_qpos[2]='0 0 0 0.0248 0.04536 -0.7854 -1.13447 0.514973 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 1.44532 -0.204204 1.46103 1.44532 1.48459 -0.2618 1.47674 1.48459'.split(' ')
ASL_qpos[3]='0 0 0 0.3384 0.25305 0.01569 -0.0262045 0.645885 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 1.571 -0.036652 1.52387 1.45318 1.40604 -0.068068 1.39033 1.571'.split(' ')
ASL_qpos[4]='0 0 0 0.6392 -0.147495 -0.7854 -1.309 0.637158 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 0.306345 -0.010472 0.400605 0.133535 0.21994 -0.068068 0.274925 0.01571'.split(' ')
ASL_qpos[5]='0 0 0 0.3384 0.25305 0.01569 -0.0262045 0.645885 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 0.306345 -0.010472 0.400605 0.133535 0.21994 -0.068068 0.274925 0.01571'.split(' ')
ASL_qpos[6]='0 0 0 0.6392 -0.147495 -0.7854 -1.309 0.637158 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 0.306345 -0.010472 0.400605 0.133535 1.1861 -0.2618 1.35891 1.48459'.split(' ')
ASL_qpos[7]='0 0 0 0.524 0.01569 -0.7854 -1.309 0.645885 -0.006982 0.128305 0.111722 0.510575 0 0.37704 0.117825 1.28036 -0.115192 1.52387 1.45318 0.432025 -0.068068 0.18852 0.149245'.split(' ')
ASL_qpos[8]='0 0 0 0.428 0.22338 -0.7854 -1.309 0.645885 -0.006982 0.128305 0.194636 1.39033 0 1.08399 0.573415 0.667675 -0.020944 0 0.06284 0.432025 -0.068068 0.18852 0.149245'.split(' ')
ASL_qpos[9]='0 0 0 0.5624 0.28272 -0.75573 -1.309 1.30045 -0.006982 1.45492 0.998897 0.39275 0 0.18852 0.227795 0.667675 -0.020944 0 0.06284 0.432025 -0.068068 0.18852 0.149245'.split(' ')

# ASl Eval envs for each numerals
for k in ASL_qpos.keys():
    register_env_with_variants(id='CustomMyoHandPose'+str(k)+'Fixed-v0',
            entry_point='envs.pose:CustomPoseEnv',
            max_episode_steps=100,
            kwargs={
                'model_path': myosuite_path + '/assets/hand/myo_hand_pose.mjb',
                'viz_site_targets': ('THtip','IFtip','MFtip','RFtip','LFtip'),
                'target_jnt_value': np.array(ASL_qpos[k],'float'),
                'normalize_act': True,
                'pose_thd': .7,
                'reset_type': "init",        # none, init, random
                'target_type': 'fixed',      # generate/ fixed
            }
    )

# ASL Train Env
m = np.array([ASL_qpos[i] for i in range(10)]).astype(float)
Rpos = {}
for i_n, n  in enumerate(jnt_namesHand):
    Rpos[n]=(np.min(m[:,i_n]), np.max(m[:,i_n]))

register_env_with_variants(id='CustomMyoHandPoseRandom-v0',  #reconsider
        entry_point='envs.pose:CustomPoseEnv',
        max_episode_steps=100,
        kwargs={
            'model_path': myosuite_path + '/assets/hand/myo_hand_pose.mjb',
            'viz_site_targets': ('THtip','IFtip','MFtip','RFtip','LFtip'),
            'target_jnt_range': Rpos,
            'normalize_act': True,
            'pose_thd': .8,
            'reset_type': "random",         # none, init, random
            'target_type': 'generate',      # generate/ fixed
        }
    )


# Pen twirl
register_env_with_variants(id='CustomMyoHandPenTwirlRandom-v0',
        entry_point='envs.pen:CustomPenEnv',
        max_episode_steps=100,
        kwargs={
            'model_path': myosuite_path + '/assets/hand/myo_hand_pen.mjb',
            'normalize_act': True,
            'frame_skip': 5,
        }
    )