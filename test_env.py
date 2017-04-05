from nvs.envs.nvs_env import NVSEnv
from gym.envs.registration import register
import gym

max_steps = 12
seed = 100

register(
    id='New-View-Synthesis-test-v0',
    entry_point='nvs.envs.nvs_env:NVSEnv',
    kwargs={'max_steps': max_steps})

env = gym.make('New-View-Synthesis-test-v0')
env.seed(seed)

# Change the render delay time
env.set_render_delay(0.5)

# Test moving the image clockwise
obs = env.reset()
cw_idx = list(env.actions.values()).index('CW')
cw_action = list(env.actions.keys())[cw_idx]
for i in range(max_steps):
	obs, reward, is_terminal, info = env.step(cw_action)
	env.render()

# Change the render delay time faster
env.set_render_delay(0.1)

# Test moving the image counter clockwise
obs = env.reset()
ccw_idx = list(env.actions.values()).index('CCW')
ccw_action = list(env.actions.keys())[ccw_idx]
for i in range(max_steps):
	obs, reward, is_terminal, info = env.step(ccw_action)
	env.render()

# Test guessing incorrectly
obs = env.reset()
category = env.category
category_idx = list(env.actions.values()).index(category) + 1
category_action = list(env.actions.keys())[category_idx]
obs, reward, is_terminal, info = env.step(category_action)
assert(reward == 0)
assert(not is_terminal)

# Test guessing correctly
obs = env.reset()
category = env.category
category_idx = list(env.actions.values()).index(category)
category_action = list(env.actions.keys())[category_idx]
obs, reward, is_terminal, info = env.step(category_action)
assert(reward == 1)
assert(is_terminal)

# Test max steps
obs = env.reset()
for i in range(max_steps+1):
	obs, reward, is_terminal, info = env.step(0)
	if i < max_steps:
		assert(not is_terminal)
	else:
		assert(is_terminal)
