import numpy as np
import pandas as pd
import os
import os.path as osp
import random
import ansys.fluent.core as pyfluent
from decimal import Decimal
import gym
from gym import Env
from gym import spaces
import datetime
import time
import random

"""
Gym environment wrapping a 2D cylinder-flow simulation in Ansys Fluent.

The agent controls the mass-flow-rate of a pair of synthetic jets placed on
the cylinder. The goal is to reduce drag and lift fluctuations.

Key features:
- Continuous action space: jet mass-flow in [min_jet, max_jet].
- State: velocity magnitude at 151 probe locations.
- Reward: linear combination of drag and (absolute) lift.
"""

# Root directory for this project (update if needed)
project_dir = r'E:\\Doctoral_work\\MBRL\\buchong\\Re200_False'


class CylinderEnv2(Env):
    """Environment for 2D flow simulation around a cylinder."""

    def __init__(self, number_steps_execution=50):
        """
        Args:
            number_steps_execution (int): number of Fluent iterations per action.
        """
        # super(self, Env2DCylinder).__init__()
        # Action: jet mass-flow rate [kg/s]
        self.min_jet = 0                                                       # 最低射流速度
        self.max_jet = 0.04                                                     # 最高射流速度


        self.number_steps_execution = number_steps_execution                   # 执行的步数。也就是说一个action执行的步数
        # Continuous action space
        self.action_space = spaces.Box(low=self.min_jet, high=self.max_jet, shape=(1,), dtype=float)

        # Continuous observation space
        # We use velocity magnitude at 151 probes as state
        low = 0.0
        high = 100
        shape = (151,)
        self.observation_space = spaces.Box(low=low, high=high, shape=shape, dtype=float)

        # Episode bookkeeping
        self.episode_number    = 80                                                # episode计数
        self.episode_drags     = np.array([])
        self.episode_lifts     = np.array([])
        self.action_list       = np.array([])

        # Fluent case path
        self.start_case        = 'E:\\Doctoral_work\\MBRL\\buchong\\Re200_False\\case\\200.0000.case.h5.dat'
        self.max_step          = 80

        # Control variable: previous jet value (used in Named Expressions)
        self.jet_vec           = 0
        self.episode_counter   = 80
        # self.start_class(complete_reset=True)

    # ------------------------------------------------------------------ #
    # Initialization and reset
    # ------------------------------------------------------------------ #
    def start_class(self,complete_reset=True):
        """
        Initialize or partially reset the Fluent simulation.

        Args:
            complete_reset (bool):
                - True: start a new Fluent session and read case data.
                - False: reuse existing session and continue from current state.
        """
        if complete_reset == False:
            # Only reset step counter; keep current flowfield
            self.solver_step = 0
        else:
            # Full reset of simulation-related state
            self.solver_step = 0
            self.flow_time = 551.23
            self.accumulated_drag = 0
            self.accumulated_lift = 0

            # Launch Fluent (2D, double precision)
            self.session = pyfluent.launch_fluent(version='2d', precision='double', start_timeout=200000,processor_count=1)
            self.session.tui.define.user_defined.compiled_functions("load", "libudf")
            self.session.tui.define.user_defined.compiled_functions("compile","libudf","yes","inlet.c","","")
            # Read case and data
            self.session.tui.file.read_case_data(self.start_case)
            self.ready_to_use = True
            self.jet_vec = 0

        # Build episode-specific directories and file paths
        self.episode_counter += 1
        self.episode_dir = osp.join(project_dir,r"data",r'episode_' + str(self.episode_counter) +'_' +str(self.episode_number))

        self.flowfield_dir = osp.join(self.episode_dir, r"flowfield")
        self.cdcl_path = osp.join(self.episode_dir, r"cdcl.csv")
        self.action_path = osp.join(self.episode_dir,r"action.csv")
        self.flowfield_name = osp.join(self.flowfield_dir, r"episode_" + str(self.solver_step) + ".csv")

        # Tell Fluent where to write drag/lift history
        self.session.tui.solve.report_files.edit("cdcl","file-name",self.cdcl_path)
        # Get initial observation
        self.new_observation = self.get_obs()


    # ------------------------------------------------------------------ #
    # Core Gym API
    # ------------------------------------------------------------------ #
    def step(self,action):
        """
        Apply one action to the environment.

        Args:
            action (np.ndarray): scalar jet mass-flow rate.

        Returns:
            next_state, reward, done, truncated, info
        """
        # Clip action into valid range
        action = np.clip(action[0],self.min_jet,self.max_jet)
        self.action_list = np.append(self.action_list, action)

        # Build Named Expression for smoothly updating the jet
        TimeExponent = 200*self.flow_time
        Name_Expressions = '''"(1-0.9^(200*(flow_time/1[s]) - {kkk}))*{action} [kg/s] + 0.9^(200*flow_time/1[s] -{kkk})*{jec_vec} [kg/s]"'''.format(
            jec_vec=self.jet_vec, kkk=TimeExponent, action=action)                                                                                            #射流速度的Named_expressions表达式，（1-0.9^((t-491.23)/0.005+1))*action +0.9^((t-491.23)/0.005+1)*jec_vec
        # Update boundary conditions in Fluent
        self.session.tui.define.boundary_conditions.mass_flow_inlet("jet1", "yes", "yes", "no", Name_Expressions, "no", "0", "no", "yes")
        self.session.tui.define.boundary_conditions.mass_flow_outlet("jet2", "yes", "yes", "no", Name_Expressions, "no", "0", "no", "yes")
        # Advance the flow by a fixed number of iterations
        self.session.tui.solve.dual_time_iterate("50", "10")
        self.jet_vec = action
        # Export flowfield and read drag/lift
        self.flowfield_name = osp.join(self.flowfield_dir, r"episode_" + str(self.solver_step+1) + ".csv")
        self.new_observation = self.get_obs()[:,0]
        self.drag, self.lift = self.get_drag_lift()

        self.accumulated_drag += self.drag
        self.accumulated_lift += self.lift

        self.solver_step += 1
        self.flow_time += 0.25
        next_state = np.reshape(self.new_observation, (151,))

        reward = self.compute_reward()

        terminal = False
        if self.solver_step > (self.max_step-1):
            terminal = True
            np.savetxt(self.action_path, self.action_list,delimiter=',')

        truncated = False

        return next_state, reward, terminal, truncated, {}

    def get_obs(self):
        """
        Export velocity magnitude at probe locations and read from CSV.

        Returns:
            np.ndarray: probe velocities (num_probes x 1).
        """
        self.session.tui.file.export.ascii(self.flowfield_name, "probes", ",", "yes", "velocity-magnitude", "quit", "yes")
        new_observation = pd.read_csv(self.flowfield_name, usecols=[3])
        return np.array(new_observation)

    def get_drag_lift(self):
        """
        Read drag and lift coefficients from cdcl.csv.

        Returns:
            (cd_mean, |cl_mean|)
        """
        df = pd.read_csv(self.cdcl_path, skiprows=2, sep=' ')

        # Use the last 50 time steps to compute averages
        last_50_rows = df.tail(50)

        cd_mean = last_50_rows.iloc[:, 1].mean()
        cl_mean = last_50_rows.iloc[:, 2].mean()
        cl_mean = np.abs(cl_mean)
        return cd_mean, cl_mean


    # def __str__(self):
    #     print("Env2DCylinder ---")

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """

        self.ready_to_use = False

    def seed(self, seed):
        """
        Sets the random seed of the environment to the given value (current time, if seed=None).
        Naturally deterministic Environments don't have to implement this method.

        Args:
            seed (int): The seed to use for initializing the pseudo-random number generator (default=epoch time in sec).
        Returns: The actual seed (int) used OR None if Environment did not override this method (no seeding supported).
        """

        # we have a deterministic environment: no need to implement

        return None

    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial_state (np.ndarray)
        """
        if self.episode_number > 80:
            mean_accumulated_drag = self.accumulated_drag / self.max_step
            mean_accumulated_lift = self.accumulated_lift / self.max_step
            print("mean accumulated drag on the whole episode: {}".format(mean_accumulated_drag))

            chance = random.random()

            probability_hard_reset = 0.2

            if chance < probability_hard_reset:
                # print("重置环境学习")
                self.session.exit()
                self.start_class(complete_reset=True)
            else:
                # print("继续环境学习")
                self.start_class(complete_reset=False)

            next_state = np.reshape(self.new_observation,(151,))

            self.episode_number += 1

        else:
            self.start_class(complete_reset=True)
            self.episode_number += 1

            next_state = np.reshape(self.new_observation,(151,))

        return next_state


    def compute_reward(self):
        """
        Reward function.

        Current design: reward is a shifted negative linear combination of
        drag and lift.

        Returns:
            float: reward value.
        """
        return - self.drag - 0.2 * self.lift + 3.13
