import random
import pickle

from gym_env import TicTacToeEnv, agent_by_mark
from agent import MCAgent


def learn(alpha, epsilon, episode_count, Q):
    env = TicTacToeEnv(3)
    agents = [MCAgent('O', epsilon, Q, True), MCAgent('X', epsilon, Q, True)]

    start_mark = 'O'
    reward_list = []
    for i in range(episode_count):

        env.set_start_mark(start_mark)
        state, mark = env.reset()

        done = False
        state_action_list = []
        while not done:
            agent = agent_by_mark(agents, mark)
            actions = env.available_actions()
            action = agent.act(state, actions)

            state_action_list.append((state, action))

            state, reward, done, _ = env.step(action)
            state, mark = state
        
        for state, action in state_action_list:
            Q[state][action] += alpha * (reward - Q[state][action])

        reward_list.append(reward)
    return reward_list

def evaluate_agent():
    with open('src/models/perfect_model_real.pkl', 'rb') as f:
        Q = pickle.load(f)

    episode_count = 1000

    reward_list = learn(0, 0, episode_count, Q)
    print("Reward list: ", reward_list[0:100])
    print(reward_list.count(0))
    print("% of non-draw games", (episode_count-reward_list.count(0))/episode_count)

if __name__ == "__main__":
    episode_count = 100000
    Q = {}

    learn(0.3, 0.2, episode_count, Q)

    with open('src/models/perfect_model_real.pkl', 'wb') as f:
        pickle.dump(Q, f)

    evaluate_agent()