import random
import pickle

from gym_env import TicTacToeEnv, agent_by_mark
from agent import MCAgent

FILE = 'models/perfect_model.pkl'
ALPHA = 0.3
EPSILON = 0.2
EPISODE_COUNT = 100000
EPISODE_COUNT_TEST = 1000
SHOW_TEST_RESULTS = 100

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
    with open(FILE, 'rb') as f:
        Q = pickle.load(f)

    reward_list = learn(0, 0, EPISODE_COUNT_TEST, Q)
    print("Reward list: ", reward_list[:SHOW_TEST_RESULTS])
    print(reward_list.count(0))
    print("% of non-draw games", (EPISODE_COUNT_TEST-reward_list.count(0))/EPISODE_COUNT_TEST)

if __name__ == "__main__":
    Q = {}

    learn(ALPHA, EPSILON, EPISODE_COUNT, Q)

    with open(FILE, 'wb') as f:
        pickle.dump(Q, f)

    evaluate_agent()