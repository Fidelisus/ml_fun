#!/usr/bin/env python
import sys
import click
import pickle

from PyQt5.QtWidgets import QApplication
from window import Window
from agent import *
from env import TicTacToeEnv, agent_by_mark, next_mark

from mc_agent import MCAgent

def get_trained_agent():
    with open('Exercise 3/q_array_dumps/perfect_model.pkl', 'rb') as f:
        Q = pickle.load(f)
    
    return MCAgent('O', 0.5, 0.05, Q)

@click.command(help="Play human agent.")
@click.option('-n', '--show-number', is_flag=True, default=False,
              show_default=True, help="Show location number in the board.")
def play(show_number):
    agents = [HumanAgent('O'), get_trained_agent()]
    game = TicTacToeEnv()

    app = QApplication(sys.argv)
    window = Window(3, 3, game, agents)
    sys.exit(app.exec_())
    


"""
    env = TicTacToeEnv(show_number=show_number)
    episode = 0
    while True:
        state = env.reset()
        _, mark = state
        done = False
        env.render()
        while not done:
            agent = agent_by_mark(agents, next_mark(mark))
            env.show_turn(True, mark)
            ava_actions = env.available_actions()
            action = agent.act(ava_actions)
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
"""

if __name__ == '__main__':
    play()