import sys

from gym_env import TicTacToeEnv, agent_by_mark, next_mark
from agent import HumanAgent, MCAgent


def play():
    env = TicTacToeEnv(3)
    agents = [HumanAgent('O'), MCAgent('X', 0.2, {}, False)]
    episode = 0
    while True:
        state = env.reset()
        _, mark = state
        done = False
        env.render()
        while not done:
            agent = agent_by_mark(agents, mark)
            env.show_turn(True, mark)
            ava_actions = env.available_actions()
            if isinstance(agent, HumanAgent):
                action = agent.act(ava_actions)
            else:
                action = agent.act(state[0], ava_actions)
            if action is None:
                sys.exit()

            state, reward, done, info = env.step(action)

            print('')
            env.render()
            if done:
                env.show_result(True, mark, reward)
                break
            else:
                _, mark = state
        episode += 1


if __name__ == '__main__':
    play()