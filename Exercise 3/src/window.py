import pickle
import copy
from os.path import exists
#from time import sleep
from functools import partial
from PyQt5.QtWidgets import QMainWindow, QPushButton, QComboBox, QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtGui import QFont

from gym_env import check_game_status, agent_by_mark
from agent import *


FILE = 'models/perfect_model.pkl'
ALPHA = 0.5
EPSILON = 0.2

Q_VALUES = {}
with open(FILE, 'rb') as f:
    LOADED = pickle.load(f)
POSSIBLE_AGENTS = [HumanAgent(""), BaseAgent(""), MCAgent("", 0, Q_VALUES), MCAgent("", 0, LOADED), MCAgent("", 0, {})]

MARKS = ['O', 'X']
SELECTED_AGENTS = (0, 1)
AI_AGENT = copy.copy(POSSIBLE_AGENTS[2])
TIME_USED_PER_TURN_AGENT = 0.5

HORIZONTAL_SPACE = 12
VERTICAL_SPACE = 12
HORIZONTAL_BUTTON_SIZE = 150
VERTICAL_BUTTON_SIZE = 150
BUTTON_FONT_SIZE = 72
HORIZONTAL_NORMAL_BUTTON_SIZE = 150
VERTICAL_NORMAL_BUTTON_SIZE = 50
VERTICAL_PROGRESSBAR_SIZE = 12


 
class Window(QMainWindow):
    def __init__(self, game):
        super().__init__()
        self.game = game
        self.size = game.size
        self.state_action_list = []
        self.agents = [copy.deepcopy(POSSIBLE_AGENTS[SELECTED_AGENTS[0]]), copy.deepcopy(POSSIBLE_AGENTS[SELECTED_AGENTS[1]])]
        self.current_agent_index = 0
        self.current_agent = self.agents[self.current_agent_index]

        self.setWindowTitle("TicTacToe")
        window_width = HORIZONTAL_SPACE*(self.size+1) + HORIZONTAL_BUTTON_SIZE*self.size
        window_height = VERTICAL_SPACE*(self.size+3) + VERTICAL_BUTTON_SIZE*self.size + VERTICAL_NORMAL_BUTTON_SIZE*2
        self.setGeometry(1600, 300, window_width, window_height)

        #progressbar
        self.progressbar = QProgressBar(self)
        self.progressbar.setGeometry(0, 0, window_width, VERTICAL_PROGRESSBAR_SIZE)
        self.progressbar.hide()

        #select agents
        self.first_agent = QComboBox(self)
        for agent in POSSIBLE_AGENTS:
            self.first_agent.addItem(str(agent))
        self.first_agent.setCurrentIndex(SELECTED_AGENTS[0])
        self.first_agent.setGeometry(   HORIZONTAL_SPACE,
                                        VERTICAL_SPACE,
                                        HORIZONTAL_NORMAL_BUTTON_SIZE,
                                        VERTICAL_NORMAL_BUTTON_SIZE)
        self.first_agent.activated.connect(self.set_agents)
        self.second_agent = QComboBox(self)
        for agent in POSSIBLE_AGENTS:
            self.second_agent.addItem(str(agent))
        self.second_agent.setCurrentIndex(SELECTED_AGENTS[1])
        self.second_agent.setGeometry(	HORIZONTAL_SPACE + (HORIZONTAL_SPACE+HORIZONTAL_BUTTON_SIZE)*(self.size - 1),
                                        VERTICAL_SPACE,
                                        HORIZONTAL_NORMAL_BUTTON_SIZE,
                                        VERTICAL_NORMAL_BUTTON_SIZE)
        self.second_agent.activated.connect(self.set_agents)

        self.label = QLabel(self)
        self.label.setGeometry(	HORIZONTAL_SPACE + (HORIZONTAL_SPACE+HORIZONTAL_BUTTON_SIZE),
                                VERTICAL_SPACE,
                                HORIZONTAL_NORMAL_BUTTON_SIZE*(self.size - 2)+HORIZONTAL_SPACE*(self.size - 3),
                                VERTICAL_NORMAL_BUTTON_SIZE)
        self.label.setText(f"{MARKS[0]}'s turn.")


        #TicTacToe board
        self.buttons = []
        for j in range(self.size):
            for i in range(self.size):
                button = button = QPushButton(' ', self)
                button.setToolTip(str(self.size*j + i + 1))
                button.setFont(QFont('Arial', BUTTON_FONT_SIZE))
                button.setGeometry(	HORIZONTAL_SPACE + (HORIZONTAL_SPACE+HORIZONTAL_BUTTON_SIZE)*i,
                                    VERTICAL_SPACE*2 + (VERTICAL_SPACE+VERTICAL_BUTTON_SIZE)*j + VERTICAL_NORMAL_BUTTON_SIZE,
                                    HORIZONTAL_BUTTON_SIZE,
                                    VERTICAL_BUTTON_SIZE)
                button.clicked.connect(self.human_turn)
                self.buttons.append(button)

        #learn buttons
        self.learn_buttons = []
        for j in range(1,3):
            for i in range(1,3):
                episode_count = 10**(2*j+i)
                button = button = QPushButton(f'learn 10^{(2*j+i)}', self)
                button.setToolTip(str(episode_count))
                button.setGeometry(	HORIZONTAL_SPACE + (HORIZONTAL_SPACE+HORIZONTAL_BUTTON_SIZE) + ((HORIZONTAL_NORMAL_BUTTON_SIZE*(self.size - 2)+HORIZONTAL_SPACE*(self.size - 3))//2)*(i-1),
                                    VERTICAL_SPACE*2 + (VERTICAL_SPACE+VERTICAL_BUTTON_SIZE)*self.size + VERTICAL_NORMAL_BUTTON_SIZE + (VERTICAL_NORMAL_BUTTON_SIZE//2)*(j-1),
                                    (HORIZONTAL_NORMAL_BUTTON_SIZE*(self.size - 2)+HORIZONTAL_SPACE*(self.size - 3))//2,
                                    VERTICAL_NORMAL_BUTTON_SIZE//2)
                button.clicked.connect(partial(self.learn, episode_count, Q_VALUES, ALPHA, EPSILON))
                self.learn_buttons.append(button)

        #control buttons
        self.ai_button = QPushButton('AI', self)
        self.ai_button.setGeometry(	HORIZONTAL_SPACE,
                                    VERTICAL_SPACE*2 + (VERTICAL_SPACE+VERTICAL_BUTTON_SIZE)*self.size + VERTICAL_NORMAL_BUTTON_SIZE,
                                    HORIZONTAL_NORMAL_BUTTON_SIZE,
                                    VERTICAL_NORMAL_BUTTON_SIZE)
        self.ai_button.clicked.connect(self.ai_turn)
        self.reset_button = QPushButton('RESET', self)
        self.reset_button.setGeometry(	HORIZONTAL_SPACE + (HORIZONTAL_SPACE+HORIZONTAL_BUTTON_SIZE)*(self.size - 1),
                                        VERTICAL_SPACE*2 + (VERTICAL_SPACE+VERTICAL_BUTTON_SIZE)*self.size + VERTICAL_NORMAL_BUTTON_SIZE,
                                        HORIZONTAL_NORMAL_BUTTON_SIZE,
                                        VERTICAL_NORMAL_BUTTON_SIZE)
        self.reset_button.clicked.connect(self.reset)

        self.show()
        self.set_agents()


    def game_play(self):
        self.current_agent = self.agents[self.current_agent_index]
        if not isinstance(self.current_agent, HumanAgent):
            self.ai_turn()

    
    def human_turn(self):
        state, mark, ava_actions = self.before_turn()

        clicked_button = self.sender()
        clicked_button.setText(mark)
        clicked_button.setEnabled(False)

        action = self.buttons.index(clicked_button)
        self.state_action_list.append((state, action))

        self.after_turn(mark, action)


    def ai_turn(self):
        state, mark, ava_actions = self.before_turn()

        if self.sender() == self.ai_button:
            AI_AGENT.mark = self.current_agent.mark
            action = AI_AGENT.act(state, ava_actions)
        else:
            action = self.current_agent.act(state, ava_actions)
        self.state_action_list.append((state, action))

        #if not any(isinstance(agent, HumanAgent) for agent in self.agents):
            #sleep(TIME_USED_PER_TURN_AGENT)
        
        clicked_button = self.buttons[action]
        clicked_button.setText(mark)
        clicked_button.setEnabled(False)

        #self.after_turn(mark, action)


    def before_turn(self):
        for button in self.buttons:
            button.setEnabled(False)

        state, mark = self.game._get_obs()
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
            if self.current_agent_index == 0:
                self.current_agent_index = 1
                self.label.setText(f"{MARKS[self.current_agent_index]}'s turn.")
            else:
                self.current_agent_index = 0
                self.label.setText(f"{MARKS[self.current_agent_index]}'s turn.")
            self.current_agent = self.agents[self.current_agent_index]
            self.game_play()


    def reset(self):
        self.game.reset()
        self.current_agent_index = 0
        self.label.setText(f"{MARKS[0]}'s turn.")
        self.update_agent_lists()

        self.first_agent.setEnabled(True)
        self.second_agent.setEnabled(True)
        for button in self.buttons:
            button.setText(' ')
            button.setEnabled(True)
        self.ai_button.setEnabled(True)
        for button in self.learn_buttons:
            button.setEnabled(True)
        self.reset_button.setEnabled(True)

        self.game_play()

    def finish(self, mark, reward):
        status = check_game_status(self.game.board)
        assert status >= 0
        if status == 0:
            self.label.setText("DRAW")
        else:
            self.label.setText(f"Winner is {tomark(status)}!")

        for state, action in self.state_action_list:
            if not state in Q_VALUES:
                l = len(state)
                Q_VALUES[state] = {k:v for k,v in zip(range(l), [0]*l)}
            Q_VALUES[state][action] += ALPHA * (reward - Q_VALUES[state][action])

        for button in self.buttons:
            button.setEnabled(False)
        self.ai_button.setEnabled(False)

    def set_agents(self):
        first_index = self.first_agent.currentIndex()
        POSSIBLE_AGENTS[first_index].mark = MARKS[0]
        self.agents[0] = copy.copy(POSSIBLE_AGENTS[first_index])

        second_index = self.second_agent.currentIndex()
        POSSIBLE_AGENTS[second_index].mark = MARKS[1]
        self.agents[1] = copy.copy(POSSIBLE_AGENTS[second_index])

        if self.game.done:
            self.reset()
        else:
            self.game_play()


    def learn(self, episode_count, Q, alpha, epsilon):
        self.progressbar.show()
        self.first_agent.setEnabled(False)
        self.second_agent.setEnabled(False)
        for button in self.buttons:
            button.setEnabled(False)
        self.ai_button.setEnabled(False)
        for button in self.learn_buttons:
            button.setEnabled(False)
        self.reset_button.setEnabled(False)


        agents = [MCAgent(MARKS[0], epsilon, Q), MCAgent(MARKS[1], epsilon, Q)]
        self.game.set_start_mark(MARKS[0])

        for i in range(episode_count):
            done = False
            state, mark = self.game.reset()
            self.state_action_list = []
            while not done:
                agent = agent_by_mark(agents, mark)
                actions = self.game.available_actions()
                action = agent.act(state, actions)

                self.state_action_list.append((state, action))

                state, reward, done, _ = self.game.step(action)
                state, mark = state

            for state, action in self.state_action_list:
                Q[state][action] += alpha * (reward - Q[state][action])
            self.progressbar.setValue(int(((i+1)/episode_count)*100))

        with open(f"{FILE}_{episode_count}", 'wb') as f:
            pickle.dump(Q, f)
        
        self.reset()
        self.progressbar.hide()


    def update_agent_lists(self):
        first_index = self.first_agent.currentIndex()
        second_index = self.second_agent.currentIndex()

        self.first_agent.clear()
        self.second_agent.clear()
        for agent in POSSIBLE_AGENTS:
            self.first_agent.addItem(str(agent))
            self.second_agent.addItem(str(agent))

        self.first_agent.setCurrentIndex(first_index)
        self.second_agent.setCurrentIndex(second_index)
        