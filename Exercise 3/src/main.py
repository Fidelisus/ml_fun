#!/usr/bin/env python
import sys
from PyQt5.QtWidgets import QApplication

from gym_env import TicTacToeEnv

from window import Window

BOARD_SIZE = 3

def play():
    game = TicTacToeEnv(BOARD_SIZE)

    app = QApplication(sys.argv)
    window = Window(game)
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    play()