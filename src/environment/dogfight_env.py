import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jsbsim


class DogfightEnv(gym.Env):
    """
    Custom Gym environment for dogfight simulation using JSBSim.
    Implements F-16 flight dynamics and RL-compatible interface.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super(DogfightEnv, self).__init__()
        self.render_mode = render_mode

        # JSBSim setup
        self.sim = None
        self.initial_altitude = 10000  # feet
        self.initial_speed = 500       # ft/s

        # Observation space: 12 base + 4 radar + 1 missile = 17
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )

        # Action space: [aileron, elevator, rudder, throttle]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        import os
        root_dir = os.path.dirname(jsbsim.__file__)
        self.sim = jsbsim.FGFDMExec(root_dir, log_lvl=0)

        # Force BODY axis reference BEFORE loading model
        self.sim['metrics/aero-ref'] = 1  # 1 = BODY, 2 = WIND, 3 = STABILITY

        self.sim.load_model('c172p')
        self.sim.set_dt(0.05)

        # Set initial conditions
        self.sim['ic/h-sl-ft'] = self.initial_altitude
        self.sim['ic/ubody-fps'] = self.initial_speed
        self.sim['ic/vbody-fps'] = 0.0
        self.sim['ic/wbody-fps'] = 0.0
        self.sim['ic/phi-rad'] = 0.0
        self.sim['ic/theta-rad'] = 0.0
        self.sim['ic/psi-rad'] = 0.0
        self.sim['ic/alpha-deg'] = 0.0
        self.sim['ic/beta-deg'] = 0.0
        self.sim['ic/gamma-deg'] = 0.0
        self.sim['ic/roc-fpm'] = 0.0

        self.sim.run_ic()

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        # Step simulation
        self.sim.run()

        # Get new state
        observation = self._get_observation()
        reward = self._get_reward()
        terminated = self._is_done()
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        # Base state
        base_obs = np.array([
            self.sim['position/h-sl-ft'],           # Altitude
            self.sim['velocities/ve-fps'],           # Ground speed
            self.sim['attitude/phi-rad'],            # Roll
            self.sim['attitude/theta-rad'],          # Pitch
            self.sim['attitude/psi-rad'],            # Yaw
            self.sim['velocities/p-rad_sec'],        # Roll rate
            self.sim['velocities/q-rad_sec'],        # Pitch rate
            self.sim['velocities/r-rad_sec'],        # Yaw rate
            self.sim['fcs/aileron-cmd-norm'],        # Aileron
            self.sim['fcs/elevator-cmd-norm'],       # Elevator
            self.sim['fcs/rudder-cmd-norm'],         # Rudder
            self.sim['fcs/throttle-cmd-norm'],       # Throttle
        ], dtype=np.float32)

        # Radar data
        radar = self.get_radar_data()
        # Missile warning
        missile_alert = self.check_missile_alert()

        # Concatenate
        return np.concatenate([base_obs, radar, [missile_alert]])

    def _get_reward(self):
        # Simple reward: penalize deviation from level flight
        altitude = self.sim['position/h-sl-ft']
        speed = self.sim['velocities/ve-fps']
        pitch = self.sim['attitude/theta-deg']

        altitude_reward = -abs(altitude - self.initial_altitude) / 1000
        speed_reward = -abs(speed - self.initial_speed) / 100
        pitch_reward = -abs(pitch)

        return altitude_reward + speed_reward + pitch_reward

    def _is_done(self):
        # Terminate if crashed or extreme attitude
        altitude = self.sim['position/h-sl-ft']
        pitch = abs(self.sim['attitude/theta-deg'])
        roll = abs(self.sim['attitude/phi-deg'])

        if altitude < 100:  # Crashed
            return True
        if pitch > 90 or roll > 90:  # Unrecoverable
            return True
        return False

    def get_radar_data(self):
        az = np.random.uniform(-30, 30)
        el = np.random.uniform(-10, 10)
        rng = np.random.uniform(500, 10000)
        closure = np.random.uniform(-200, 300)
        return np.array([az, el, rng, closure], dtype=np.float32)

    def check_missile_alert(self):
        return 1.0 if np.random.rand() < 0.05 else 0.0

    def close(self):
        pass

