#!/usr/bin/env python
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

from gym_env import TicTacToeEnv
from game import Game

BOARD_SIZE = 3

def play():
    env = TicTacToeEnv(BOARD_SIZE)

    app = QApplication(sys.argv)
    game = Game(env)
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    play()