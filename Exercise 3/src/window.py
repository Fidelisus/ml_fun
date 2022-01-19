import time
import gym
from gym import spaces
from PyQt5.QtWidgets import QMainWindow, QPushButton, QComboBox
from PyQt5.QtGui import QFont

from agent import *


HORIZONTAL_SPACE = 30
VERTICAL_SPACE = 30
HORIZONTAL_BUTTON_SIZE = 200
VERTICAL_BUTTON_SIZE = 200
BUTTON_FONT_SIZE = 72
HORIZONTAL_NORMAL_BUTTON_SIZE = 200
VERTICAL_NORMAL_BUTTON_SIZE = 50

TIME_USED_PER_TURN_AGENT = 0.5
SELECTED_AGENTS = ["HumanAgent", "BaseAgent"]
POSSIBLE_AGENT_TYPES = [HumanAgent, BaseAgent]
AI_AGENT = BaseAgent


 
class Window(QMainWindow, gym.Env):
    def __init__(self, m, n, game):
        super().__init__()
        self.game = game
        self.m = m
        self.n = n
        self.agents = [HumanAgent("O"), BaseAgent("X")]
        self.agent_gen = self.agent_generator()
        self.current_agent = None

        self.setWindowTitle("TicTacToe")
        window_width = HORIZONTAL_SPACE*(self.m+1) + HORIZONTAL_BUTTON_SIZE*self.m
        window_height = VERTICAL_SPACE*(self.n+3) + VERTICAL_BUTTON_SIZE*self.n + VERTICAL_NORMAL_BUTTON_SIZE*2
        self.setGeometry(1600, 300, window_width, window_height)


        self.first_agent = QComboBox()
        for agent in POSSIBLE_AGENT_TYPES:
            self.first_agent.addItem(str(agent))
        self.first_agent.setCurrentText(SELECTED_AGENTS[0])
        self.first_agent.setGeometry(HORIZONTAL_SPACE,
                                            VERTICAL_SPACE,
                                            HORIZONTAL_NORMAL_BUTTON_SIZE,
                                            VERTICAL_NORMAL_BUTTON_SIZE)
        self.first_agent.currentIndexChanged.connect(self.set_agent)
        self.second_agent = QComboBox()
        for agent in POSSIBLE_AGENT_TYPES:
            self.second_agent.addItem(str(agent))
        self.second_agent.setCurrentText(SELECTED_AGENTS[1])
        self.second_agent.setGeometry(	HORIZONTAL_SPACE + (HORIZONTAL_SPACE+HORIZONTAL_BUTTON_SIZE)*(self.m - 1),
                                                VERTICAL_SPACE,
                                                HORIZONTAL_NORMAL_BUTTON_SIZE,
                                                VERTICAL_NORMAL_BUTTON_SIZE)
        self.second_agent.currentIndexChanged.connect(self.set_agent)
        

        self.buttons = []
        for j in range(self.n):
            for i in range(self.m):
                button = button = QPushButton(' ', self)
                button.setToolTip(str(self.m*j + i + 1))
                button.setFont(QFont('Arial', BUTTON_FONT_SIZE))
                button.setGeometry(	HORIZONTAL_SPACE + (HORIZONTAL_SPACE+HORIZONTAL_BUTTON_SIZE)*i,
                                    VERTICAL_SPACE*2 + (VERTICAL_SPACE+VERTICAL_BUTTON_SIZE)*j + VERTICAL_NORMAL_BUTTON_SIZE,
                                    HORIZONTAL_BUTTON_SIZE,
                                    VERTICAL_BUTTON_SIZE)
                button.clicked.connect(self.human_turn)
                self.buttons.append(button)


        self.ai_button = QPushButton('AI', self)
        self.ai_button.setGeometry(	HORIZONTAL_SPACE,
                                    VERTICAL_SPACE*2 + (VERTICAL_SPACE+VERTICAL_BUTTON_SIZE)*self.n + VERTICAL_NORMAL_BUTTON_SIZE,
                                    HORIZONTAL_NORMAL_BUTTON_SIZE,
                                    VERTICAL_NORMAL_BUTTON_SIZE)
        self.ai_button.clicked.connect(self.ai_turn)
        self.reset_button = QPushButton('RESET', self)
        self.reset_button.setGeometry(	HORIZONTAL_SPACE + (HORIZONTAL_SPACE+HORIZONTAL_BUTTON_SIZE)*(self.m - 1),
                                        VERTICAL_SPACE*2 + (VERTICAL_SPACE+VERTICAL_BUTTON_SIZE)*self.n + VERTICAL_NORMAL_BUTTON_SIZE,
                                        HORIZONTAL_NORMAL_BUTTON_SIZE,
                                        VERTICAL_NORMAL_BUTTON_SIZE)
        self.reset_button.clicked.connect(self.reset)


        self.show()
        self.game_play()


    def game_play(self):
        self.current_agent = next(self.agent_gen)
        if not isinstance(self.current_agent, HumanAgent):
            self.ai_turn()

    
    def human_turn(self):
        state, mark, _ = self.before_turn()

        clicked_button = self.sender()
        clicked_button.setText(mark)
        clicked_button.setEnabled(False)

        action = self.buttons.index(clicked_button)

        self.after_turn(mark, action)


    def ai_turn(self):
        if isinstance(self.current_agent, HumanAgent):
            self.current_agent = AI_AGENT(self.current_agent.mark)

        state, mark, ava_actions = self.before_turn()

        action = self.current_agent.act(state, ava_actions)
        
        clicked_button = self.buttons[action]
        clicked_button.setText(mark)
        clicked_button.setEnabled(False)

        self.after_turn(mark, action)


    def before_turn(self):
        for button in self.buttons:
            button.setEnabled(False)

        state = self.game._get_obs()
        _,mark = state
        ava_actions = self.game.available_actions()
        return state, mark, ava_actions

    def after_turn(self, mark, action):
        state, reward, done, info = self.game.step(action)
        if done:
            self.finish(mark, reward)
        else:
            for button in self.buttons:
                if button.text() == ' ':
                    button.setEnabled(True)
            self.game_play()


    def agent_generator(self):
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

    def set_agent(self):
        self.agents[0] = POSSIBLE_AGENT_TYPES[first_agent.currentIndex()]("X")
        self.agents[1] = POSSIBLE_AGENT_TYPES[first_agent.currentIndex()]("O")