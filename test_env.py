from nvs.envs import NVSEnv

env = NVSEnv()
obs = env.reset()

# Test moving the image clockwise
cw_idx = list(env.actions.values()).index('CW')
cw_action = list(env.actions.keys())[cw_idx]
for i in range(10):
	obs, reward, is_terminal, info = env.step(cw_action)
	env.render()

# Change the render delay time faster
env.set_render_delay(0.5)

# Test moving the image counter clockwise
ccw_idx = list(env.actions.values()).index('CCW')
ccw_action = list(env.actions.keys())[ccw_idx]
for i in range(10):
	obs, reward, is_terminal, info = env.step(ccw_action)
	env.render()

# Test guessing incorrectly
category = env.category
category_idx = list(env.actions.values()).index(category) + 1
category_action = list(env.actions.keys())[category_idx]
obs, reward, is_terminal, info = env.step(category_action)
assert(reward == 0)
assert(is_terminal == 0)

# Test guessing correctly
category = env.category
category_idx = list(env.actions.values()).index(category)
category_action = list(env.actions.keys())[category_idx]
obs, reward, is_terminal, info = env.step(category_action)
assert(reward == 1)
assert(is_terminal == 1)