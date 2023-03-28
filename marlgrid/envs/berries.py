from ..base import MultiGridEnv, MultiGrid
from ..objects import *


class SocialRejection(MultiGridEnv):
    mission = "Forage the berries before dark, don't let the poison in the refuge"
    metadata = {}

    def __init__(
            self,
            *args, 
            n_clutter=None, 
            clutter_density=None, 
            n_good_berries=1, 
            n_bad_berries=1,
            good_berry_reward=0.1,
            poisoned_berry_reward=-0.8,
            **kwargs):
        if (n_clutter is None) == (clutter_density is None):
            raise ValueError("Must provide n_clutter xor clutter_density in environment config.")

        super().__init__(*args, **kwargs)

        if clutter_density is not None:
            self.n_clutter = int(clutter_density * (self.width-2)*(self.height-2))
        else:
            self.n_clutter = n_clutter
        self.n_good_berries = n_good_berries
        self.n_bad_berries = n_bad_berries
        self.good_berry_reward = good_berry_reward
        self.poisoned_berry_reward = poisoned_berry_reward
        self.wall_x_pos = self.width//5
        # self.reset()

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