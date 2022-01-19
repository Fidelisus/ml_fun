import gym
from gym import spaces
from PyQt5.QtWidgets import QMainWindow, QPushButton

from agent import HumanAgent


HORIZONTAL_SPACE = 30
VERTICAL_SPACE = 30
HORIZONTAL_BUTTON_SIZE = 200
VERTICAL_BUTTON_SIZE = 200
HORIZONTAL_NORMAL_BUTTON_SIZE = 200
VERTICAL_NORMAL_BUTTON_SIZE = 50


 
class Window(QMainWindow, gym.Env):
    def __init__(self, m, n, game, agents):
        super().__init__()
        self.game = game
        self.m = m
        self.n = n
        self.agents = agents
        self.agent = self.agent_gen()
        next(self.agent)


        self.setWindowTitle("TicTacToe")
        window_width = HORIZONTAL_SPACE*(self.m+1) + HORIZONTAL_BUTTON_SIZE*self.m
        window_height = VERTICAL_SPACE*(self.n+2) + VERTICAL_BUTTON_SIZE*self.n + VERTICAL_NORMAL_BUTTON_SIZE
        self.setGeometry(1600, 300, window_width, window_height)

        self.buttons = []
        for i in range(self.m):
            for j in range(self.n):
                button = button = QPushButton(' ', self)
                button.setToolTip(str(self.n*i + j + 1))
                button.setGeometry(	HORIZONTAL_SPACE + (HORIZONTAL_SPACE+HORIZONTAL_BUTTON_SIZE)*j,
                                    VERTICAL_SPACE + (VERTICAL_SPACE+VERTICAL_BUTTON_SIZE)*i,
                                    HORIZONTAL_BUTTON_SIZE,
                                    VERTICAL_BUTTON_SIZE)
                button.clicked.connect(self.human_turn)
                self.buttons.append(button)

        self.ai_button = QPushButton('AI', self)
        self.ai_button.setGeometry(	HORIZONTAL_SPACE,
                                    VERTICAL_SPACE + (VERTICAL_SPACE+VERTICAL_BUTTON_SIZE)*self.n,
                                    HORIZONTAL_NORMAL_BUTTON_SIZE,
                                    VERTICAL_NORMAL_BUTTON_SIZE)
        self.ai_button.clicked.connect(self.game_play)

        self.reset_button = QPushButton('RESET', self)
        self.reset_button.setGeometry(	HORIZONTAL_SPACE + (HORIZONTAL_SPACE+HORIZONTAL_BUTTON_SIZE)*(self.m - 1),
                                        VERTICAL_SPACE + (VERTICAL_SPACE+VERTICAL_BUTTON_SIZE)*self.n,
                                        HORIZONTAL_NORMAL_BUTTON_SIZE,
                                        VERTICAL_NORMAL_BUTTON_SIZE)
        self.reset_button.clicked.connect(self.reset)
        self.show()

    def game_play(self):
        agent = next(self.agent)
        if not isinstance(agent, HumanAgent):
            self.ai_turn(agent)
    
    def human_turn(self):
        state = self.game._get_obs()
        _,mark = state
        clicked_button = self.sender()
        clicked_button.setText(mark)
        clicked_button.setEnabled(False)
        state, reward, done, info = self.game.step(self.buttons.index(clicked_button))
        if done:
            self.finish(mark, reward)
        self.game_play()

    def ai_turn(self, agent):
        ava_actions = self.game.available_actions()
        state = self.game._get_obs()
        state, mark = state
        action = agent.act(state, ava_actions)
        clicked_button = self.buttons[action]
        clicked_button.setText(mark)
        clicked_button.setEnabled(False)
        state, reward, done, info = self.game.step(action)
        if done:
            self.finish(mark, reward)
        self.game_play()

    def agent_gen(self):
        while True:
            for agent in self.agents:
                yield agent

    def reset(self):
        self.game.reset()
        for button in self.buttons:
            button.setText(' ')
            button.setEnabled(True)
            self.ai_button.setEnabled(True)

    def finish(self, mark, reward):
        self.game.show_result(True, mark, reward)
        for button in self.buttons:
            button.setEnabled(False)
        self.ai_button.setEnabled(False)
