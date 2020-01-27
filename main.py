import ROOT
import gym
import pandas as pd

env = gym.make('CartPole-v1')
env.reset()

agent = ROOT.HTMAgent()
et = ROOT.et

reward_history = []
learn = True
for n in range(5000):
        observation = env.reset()
        total_reward = 0
        for i in range(1000):
                # env.render() # Somehow this doesn't work on my system

                v0 = observation[0]
                v1 = observation[1]
                v2 = observation[2]
                v3 = observation[3]

                action = agent.compute(v0, v1, v2, v3, learn)
                observation, reward, done, info = env.step(action)
                if learn:
                    agent.learn(-1 if done else reward)

                if done:
                        break
                total_reward += reward
        agent.reset()
        print("Iteration {}, reward = {}".format(n, total_reward))
        reward_history += [total_reward]

df = pd.DataFrame({"reward": reward_history})
savefile = "result_{}learning.csv".format("no_" if learn is False else "")
df.to_csv(savefile)
