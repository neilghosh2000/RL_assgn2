from gym.envs.registration import register


register(id='PuddleWorld-v0',
         entry_point='GridWorld.grid_world:GridWorld')
