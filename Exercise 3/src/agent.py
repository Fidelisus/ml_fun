import random
from time import time

from gym_env import check_game_status, after_action_state, tomark

class HumanAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, ava_actions):
        while True:
            uloc = input("Enter location[1-9], q for quit: ")
            if uloc.lower() == 'q':
                return None
            try:
                action = int(uloc) - 1
                if action not in ava_actions:
                    raise ValueError()
            except ValueError:
                print("Illegal location: '{}'".format(uloc))
            else:
                break

        return action

    def __str__(self):
        return "HumanAgent"

class BaseAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, state, ava_actions):
        for action in ava_actions:
            nstate = after_action_state((state, self.mark), action)
            gstatus = check_game_status(nstate[0])
            if gstatus > 0:
                if tomark(gstatus) == self.mark:
                    return action
        return random.choice(ava_actions)

    def __str__(self):
        return "BaseAgent"


class MCAgent(object):
    def __init__(self, mark, epsilon, Q, winning_strategy):
        self.mark = mark
        self.epsilon = epsilon
        self.Q = Q
        self.winning_strategy = winning_strategy

    def act(self, state, actions):
        if not state in self.Q:
            l = len(state)
            self.Q[state] = {k:v for k,v in zip(range(l), [0]*l)}

        if self.winning_strategy:
            for action in actions:
                nstate = after_action_state((state, self.mark), action)
                gstatus = check_game_status(nstate[0])
                if gstatus > 0:
                    if tomark(gstatus) == self.mark:
                        return action
        return self.egreedy(state, actions)

    def egreedy(self, state, actions):
        r = random.uniform(0, 1)
        if r < self.epsilon:
            return self.get_random_action(actions)
        else:
            return self.get_best_action(state, actions)

    def get_random_action(self, actions):
        return random.choice(actions)

    def get_best_action(self, state, actions):
        values = [self.get_state_val(state, action) for action in actions]

        if self.mark == 'O':
            indices = [i for i, v in enumerate(values) if v == max(values)]
        else:
            indices = [i for i, v in enumerate(values) if v == min(values)]

        i = random.choice(indices)
        return actions[i]

    def get_state_val(self, state, action):
        return self.Q[state][action]

    def count_learned(self):
        return len(list(filter(lambda vs: sum(vs.values())!=0, self.Q.values())))

    def __str__(self):
        return f"MCAgent({self.count_learned()}, {self.winning_strategy})"