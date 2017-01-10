"""
Environment normalizer.
The function is meant to transform actions in [-1..1] scale
and postprocess observations
"""

import numpy as np
import gym
from gym import Wrapper



def show_env_param(env):
    envpar = {}
    envpar['state_shape'] = env.observation_space.shape
    envpar['action_shape'] = env.action_space.shape
    envpar['action_high'] = env.action_space.high
    envpar['action_low'] = env.action_space.low
    envpar['state_high'] = env.observation_space.high
    envpar['state_low'] = env.observation_space.low

    s = env.reset()

    print('--------------------------------------------------')
    print('PARAMETERS')
    for par in envpar:
        print('%s :' % par, envpar[par])
    print('--------------------------------------------------')

def make_norm_env(env, normalize=True):
    """ crate a new environment class with actions and states normalized to [-1,1] """
    acsp = env.action_space
    obsp = env.observation_space
    if not type(acsp)==gym.spaces.box.Box:
        raise RuntimeError('Environment with continous action space (i.e. Box) required.')
    if not type(obsp)==gym.spaces.box.Box:
        raise RuntimeError('Environment with continous observation space (i.e. Box) required.')

    env_type = type(env)

    class EnvNormDef(Wrapper):
        def __init__(self, env_, normalize=True):
            super(EnvNormDef, self).__init__(env_)

            if normalize:
                # Normalization parameters (scale,mean) for observation,actions and rewards
                self.get_obs_norm_par()
                self.get_action_norm_par()
                self.get_reward_norm_par()
            else:
                # Keep scales = 1. and means = 0. (untouched spaces)
                self.get_default_norm_par()

            self.make_spaces(normalize=normalize)


        def make_spaces(self, normalize):
            # Script that checks spaces consistency
            def assertEqual(a, b):
                assert np.all(a == b), "{} != {}".format(a, b)

            # Check and assign transformed spaces
            self.observation_space = gym.spaces.Box(self.norm_observation(obsp.low),
                                                    self.norm_observation(obsp.high))

            # self.action_space = gym.spaces.Box(-np.ones_like(acsp.high), np.ones_like(acsp.high))
            if normalize:
                self.action_space = gym.spaces.Box(-np.ones_like(acsp.high),
                                                   np.ones_like(acsp.high))
                assertEqual(self.revert_norm_action(self.action_space.high), acsp.high)
                assertEqual(self.revert_norm_action(self.action_space.low), acsp.low)

        def name(self):
            return self.env.spec.id

        def get_default_norm_par(self):
            self.reward_scale = 1.
            self.reward_mean = 0.
            self.obs_mean = 0.
            self.obs_scale = 1.
            self.action_mean = 0.
            self.action_scale = 1.

        def get_reward_norm_par(self):
            self.reward_scale = 1.
            self.reward_mean = 0.

        def get_obs_norm_par(self):
            # Observation space
            if np.any(obsp.high < 1e10):
                h = obsp.high
                l = obsp.low
                obs_span = h-l
                self.obs_mean = (h + l) / 2.
                self.obs_scale = obs_span / 2.
            else:
                self.obs_mean = np.zeros_like(obsp.high)
                self.obs_scale = np.ones_like(obsp.high)

        def get_action_norm_par(self):
            # Action space
            h = acsp.high
            l = acsp.low
            action_span = (h-l)
            self.action_mean = (h + l) / 2.
            self.action_scale = action_span / 2.

        def norm_observation(self, obs):
            obs = np.reshape(obs, [1,-1]) # Making a row array
            return (obs - self.obs_mean) / self.obs_scale

        def revert_norm_action(self, action):
            return self.action_scale * action + self.action_mean

        def norm_reward(self, reward):
            ''' has to be applied manually otherwise it makes the reward_threshold invalid '''
            return self.reward_scale * reward + self.reward_mean

        def _step(self, action):
            ac_f = np.clip(self.revert_norm_action(action), self.action_space.low, self.action_space.high)
            obs, reward, term, info = self.env.step(ac_f) # super function
            obs_f = self.norm_observation(obs)
            reward = reward * np.ones([1,]) #I've had a weird situation where reward was returned as a list, but I need a const
            reward = reward[0]
            return obs_f, reward, term, info

        def _reset(self):
            return self.norm_observation(self.env.reset())

        # def _close(self):
        #     super(EnvNormDef, self)._close()
        #
        #     # _monitor will not be set if super(Monitor, self).__init__ raises, this check prevents a confusing error message
        #     if getattr(self, '_monitor', None):
        #         self._monitor.close()

        def solved_reward_val(self):
            """
            Criteria for an env to be solved
            :return:
            """
            return 1000.0

        def max_time_steps(self):
            return 10000

        def stop_check(self, total_reward, time_steps):
            return (total_reward > self.solved_reward_val()) or (time_steps > self.max_time_steps())

        @staticmethod
        def rgb2gray(rgb):
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray

        def repeat_steps(self):
            return 1

    class EnvNorm__Pendulum_v0(EnvNormDef):
        def solved_reward_val(self):
            return -150
        def max_time_steps(self):
            return 1000

    class EnvNorm__InvertedPendulum_v1(EnvNormDef):
        def solved_reward_val(self):
            return 10000
        def max_time_steps(self):
            return 10000

    class EnvNorm__Reacher_v1(EnvNormDef):
        """
        Special class for the Reacher-v1: Different scale of rewards
        """
        def get_reward_norm_par(self):
            self.obs_scale[6] = 40.
            self.obs_scale[7] = 20.
            self.r_sc = 200.
            self.r_c = 0.




    class EnvNorm__CarRacing_v0(EnvNormDef):
        """
         Special class for the CarRacing_v0: Different observation processing
         """
        def make_spaces(self, normalize):
            # Script that checks spaces consistency
            def assertEqual(a, b):
                assert np.all(a == b), "{} != {}".format(a, b)

            # Check and assign transformed spaces
            self.observation_space = gym.spaces.Box(self.convert_obs(obsp.low),
                                                    self.convert_obs(obsp.high))

            # self.action_space = gym.spaces.Box(-np.ones_like(acsp.high), np.ones_like(acsp.high))
            if normalize:
                self.action_space = gym.spaces.Box(-np.ones_like(acsp.high),
                                                   np.ones_like(acsp.high))
                assertEqual(self.revert_norm_action(self.action_space.high), acsp.high)
                assertEqual(self.revert_norm_action(self.action_space.low), acsp.low)

        def convert_obs(self, obs):
            obs_norm = self.norm_observation(obs)
            # obs_conv = np.stack([obs_norm, obs_norm, obs_norm], axis=2)
            obs_conv = np.expand_dims(obs_norm, axis=2)
            obs_conv = np.repeat(obs_conv, self.repeat_steps(), axis=2)
            return obs_conv

        def _reset(self):
            self.obs_cur = self.convert_obs(self.env.reset())
            self.obs_prev = np.copy(self.obs_cur)
            self.reward_sum_ep = 0
            return self.obs_cur

        def norm_observation(self, obs):
            # obs = obs[0:80:2,8:88:2]
            obs = obs[0:80, 8:88]
            obs = (obs - self.obs_mean) / self.obs_scale
            obs_gry = self.rgb2gray(obs)
            # print ('obs_shape = ', obs_gry.shape)
            return obs_gry

        def get_obs_norm_par(self):
            self.obs_mean = 128.
            self.obs_scale = 128.
            # self.obs_scale = 1.

        # def step(self, action):
        #     ac_f = np.clip(self.revert_norm_action(action), self.action_space.low, self.action_space.high)
        #     obs, reward, term, info = self.env.step(ac_f)
        #     obs_f = self.norm_observation(obs)
        #     self.obs_prev = np.copy(self.obs_cur)
        #     self.obs_cur[:,:,0] = obs_f
        #     self.obs_cur[:,:,1] = self.obs_prev[:,:,0]
        #     self.obs_cur[:,:,2] = self.obs_prev[:,:,1]
        #     reward = reward * np.ones([1,]) #I've had a weird situation where reward was returned as a list, but I need a const
        #     reward = reward[0]
        #     return self.obs_cur, reward, term, info

        def _step(self, action):
            """
            We repeat few steps in the env with the same action and summing up the reward
            :param action:
            :return:
            """
            ac_f = np.clip(self.revert_norm_action(action), self.action_space.low, self.action_space.high)
            ac_f = ac_f.flatten()

            self.obs_prev = np.copy(self.obs_cur) #just in case :)
            self.obs_cur = np.zeros_like(self.obs_cur)
            rewards = []

            # Repearing steps
            for i in range(self.repeat_steps()-1,-1,-1):
                obs, reward, term, info = self.env.step(ac_f)
                # I've had a weird situation where reward was returned as a list, but I need a const
                reward = reward * np.ones([1, ])
                reward = reward[0]
                rewards.append(reward)

                obs_f = self.norm_observation(obs)
                self.obs_cur[:, :, i] = obs_f
                if term:
                    break
            reward_sum = np.sum(rewards)
            self.reward_sum_ep += reward_sum
            # if self.reward_sum_ep < -1:
            #     term = True

            return self.obs_cur, reward_sum, term, info

        def repeat_steps(self):
            """
            Tells how many steps in the env will be skipped
            :return:
            """
            return 4

    # Convertin env name into a class name
    env_name = env.spec.id
    class_name = 'EnvNorm__' + env_name.replace('-','_')

    # Searching if Env specific class exists
    class_cur = None
    for class_i in locals():
        if class_name == class_i:
            class_cur = class_name
            break

    if class_cur != None:
        EnvClass = locals()[class_name]
        print('--- %s class initilized for %s postprocessing ' % (class_name, env_name))
        fenv = EnvClass(env, normalize)
    else:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('WARNING: NO ENV specific class for preprocessing is found. Default procesing will be applied')
        fenv = EnvNormDef(env, normalize)

    print('----------------------------------------------------------------')
    print('----- ENVIRONMENT AFTER NORMALIZATION: ')
    print('--- Shapes:')
    print('True action space: ' + str(acsp.shape) + ', ' + str(acsp.shape))
    print('True state space: ' + str(obsp.shape) + ', ' + str(obsp.shape))
    print('Normalized action space: ' + str(fenv.action_space.shape) + ', ' + str(fenv.action_space.shape))
    print('Normalized state space: ' + str(fenv.observation_space.shape) + ', ' + str(fenv.observation_space.shape))
    print('--- Limits:')
    print('True action space: ' + str(acsp.low) + ', ' + str(acsp.high))
    print('True state space: ' + str(np.min(obsp.low)) + ', ' + str(np.max(obsp.high)))
    print('Normalized action space: ' + str(fenv.action_space.low) + ', ' + str(fenv.action_space.high))
    print('Normalized state space: ' + str(np.min(fenv.observation_space.low)) + ', ' + str(np.max(fenv.observation_space.high)))
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return fenv