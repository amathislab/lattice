import gym
import pybullet_envs


class EnvironmentFactory:
    """Static factory to instantiate and register gym environments by name."""

    @staticmethod
    def create(env_name, **kwargs):
        """Creates an environment given its name as a string, and forwards the kwargs
        to its __init__ function.

        Args:
            env_name (str): name of the environment

        Raises:
            ValueError: if the name of the environment is unknown

        Returns:
            gym.env: the selected environment
        """
        # make myosuite envs
        if env_name == "MyoFingerPoseRandom":
            return gym.make("myoFingerPoseRandom-v0")
        elif env_name == "MyoFingerReachRandom":
            return gym.make("myoFingerReachRandom-v0")
        elif env_name == "MyoHandReachRandom":
            return gym.make("myoHandReachRandom-v0")
        elif env_name == "MyoElbowReachRandom":
            return gym.make("myoElbowReachRandom-v0")
        elif env_name == "CustomMyoBaodingBallsP1":
            return gym.make("CustomMyoChallengeBaodingP1-v1", **kwargs)
        elif env_name == "CustomMyoReorientP2":
            return gym.make("CustomMyoChallengeDieReorientP2-v0", **kwargs)
        elif env_name == "CustomMyoElbowPoseRandom":
            return gym.make("CustomMyoElbowPoseRandom-v0", **kwargs)
        elif env_name == "CustomMyoFingerPoseRandom":
            return gym.make("CustomMyoFingerPoseRandom-v0", **kwargs)
        elif env_name == "CustomMyoHandPoseRandom":
            return gym.make("CustomMyoHandPoseRandom-v0", **kwargs)
        elif env_name == "CustomMyoPenTwirlRandom":
            return gym.make("CustomMyoHandPenTwirlRandom-v0", **kwargs)
        elif env_name == "WalkerBulletEnv":
            return gym.make("Walker2DBulletEnv-v0", **kwargs)
        elif env_name == "HalfCheetahBulletEnv":
            return gym.make("HalfCheetahBulletEnv-v0", **kwargs)
        elif env_name == "AntBulletEnv":
            return gym.make("AntBulletEnv-v0", **kwargs)
        elif env_name == "HopperBulletEnv":
            return gym.make("HopperBulletEnv-v0", **kwargs)
        elif env_name == "HumanoidBulletEnv":
            return gym.make("HumanoidBulletEnv-v0", **kwargs)
        else:
            raise ValueError("Environment name not recognized:", env_name)
