import numpy as np
import gym

results = []
env = gym.make('CartPole-v0')
env.monitor.start('cartpole_0-pg-experiment/', force=True)
total_steps = 0
for i in range(40):
    env.reset()
    last_step = 0
    for time_step in range(200):
        exec_action = np.random.choice(env.action_space.n)
        next_state, reward, done, info = env.step(exec_action)
        last_step += 1
        total_steps += 1

        if done:
            break
    print(total_steps)
    results.append(last_step)

print(results)
print(sum(env.monitor.stats_recorder.episode_lengths))


"""
Random agent simulated 1000 games:
AVG:    22.398
STDEV:  11.973
MIN:    8
MAX:    96

MAX (1000000): 189
"""