from ..base import MultiGridEnv, MultiGrid
from ..objects import *

import gym



class SocialRejection(MultiGridEnv):
    mission = "Forage the berries before dark, don't let the poison in the refuge"
    metadata = {}

    def __init__(
            self,
            *args, 
            config,
            **kwargs):
        if (config.n_clutter is None) == (config.clutter_density is None):
            raise ValueError("Must provide n_clutter xor clutter_density in environment config.")

        super().__init__(*args, width=config.width, height=config.height, reward_decay=config.reward_decay, FLASHING_TIME_POISONED_BERRIES=config.FLASHING_TIME_POISONED_BERRIES, max_steps=config.max_steps, **kwargs)

        if config.clutter_density is not None:
            self.n_clutter = int(config.clutter_density * (self.width-2)*(self.height-2))
        else:
            self.n_clutter = config.n_clutter
        if config.agent_color_space is not None:
            cs = self._get_colors(len(self.agents))
            for ix in range(len(self.agents)):
                self.agents[ix].color = cs[ix]
        self.n_good_berries = config.n_good_berries
        self.n_bad_berries = config.n_bad_berries
        self.good_berry_reward = config.good_berry_reward
        self.poisoned_berry_reward = config.poisoned_berry_reward
        self.wall_x_pos = self.width//5
        # self.reset()
    def _get_colors(self, n):
        colors = ["red", "pink", "blue", "cyan", "purple", "yellow", "olive", "orange"]

        assert n<=len(colors), f"cannot use color_space if n_agents>{len(colors)}, not enough contrasting colors"
        return colors
    def compute_rewards(self):
        
        if self.step_count >= self.max_steps:
            safe_ixs = []
            total_reward = 0
            step_rewards = [0]*len(self.agents)
            for ix, agent in enumerate(self.agents):
                if self._is_in_safe_zone(agent.pos):
                    safe_ixs.append(ix)
                    # every agent in the safe zone gets the aggregated sum of fruit rewards
                    if isinstance(agent.carrying, Berry):
                        total_reward += self.good_berry_reward
                    elif isinstance(agent.carrying, PoisonedBerry):
                        total_reward += self.poisoned_berry_reward
                else :
                    step_rewards[ix] = -1 # get penalty for not returning to the safe zone in time

            for i in safe_ixs:
                step_rewards[i] = total_reward/len(safe_ixs) # divide among survivors, might not be optimal
            return step_rewards
        return None

    def _is_in_safe_zone(self, pose):
        return (pose[0] <= getattr(self, 'wall_x_pos', 0))
    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        
        for _ in range(getattr(self, 'n_clutter', 0)):
            self.place_obj(Wall(), reject_fn=self._is_in_safe_zone, max_tries=100)
            
        for _ in range(getattr(self, 'n_good_berries', 0)):
            self.place_obj(Berry(), reject_fn=self._is_in_safe_zone, max_tries=100)
            
        for _ in range(getattr(self, 'n_bad_berries', 0)):
            self.place_obj(PoisonedBerry(), reject_fn=self._is_in_safe_zone, max_tries=100)
        
        for iy in range(self.height):
            if iy not in [self.height//2, self.height//2+1]:
                self.try_place_obj(Wall(), (getattr(self, 'wall_x_pos', 0), iy))

        self.agent_spawn_kwargs = {}
        self.place_agents(**self.agent_spawn_kwargs)

class WindowdedTextCommChannel():
    def __init__(self, max_msg_len, save_file) -> None:
        self.history = []
        self.current_text = ''
        self.max_msg_len = max_msg_len
        self.f = None
        if save_file is not None:
            self.f = open(save_file, "a+")
    @property
    def action_space(self):
        pass
    def action_space(self):
        pass
    def communicate(self, messages):
        self.history.append(self.current_text)
        self.current_text = ''.join([message[:self.max_msg_len] + '\n' for message in messages])
        if self.f is not None:
            self.f.write(self.current_text)
        return self.current_text

class CommunicationWrapper:
    def __init__(
            self, _env, config):
        self._env = _env
        
        self.comm_channel = WindowdedTextCommChannel(config.max_msg_len, config.text_save_file)
    @property
    def action_space(self):
        return gym.spaces.Dict({"actions": self._env.action_space, "messages": self.comm_channel.action_space})
    def action_space(self):
        return gym.spaces.Dict({"env": self._env.observation_space, "messages": self.comm_channel.observation_space})
    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def step(self, action):
        messages = action["Comm"]
        actions = action["actions"]


    def reset(self):
        return self._env.reset()