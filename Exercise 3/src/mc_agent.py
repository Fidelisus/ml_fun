
import random



from gym_env import TicTacToeEnv, set_log_level_by, agent_by_mark,\
    next_mark, check_game_status, after_action_state, O_REWARD, X_REWARD


DEFAULT_VALUE = 0
EPISODE_COUNT = 50000


Q = {}

default_actions = {0: DEFAULT_VALUE, 1: DEFAULT_VALUE, 2: DEFAULT_VALUE, 3: DEFAULT_VALUE, 4: DEFAULT_VALUE, 5: DEFAULT_VALUE, 6: DEFAULT_VALUE, 7: DEFAULT_VALUE, 8: DEFAULT_VALUE}

set_log_level_by(0)


def get_max_indices(vals):
    m = max(vals)
    return [i for i, v in enumerate(vals) if v == m]

def get_min_indices(vals):
    m = min(vals)
    return [i for i, v in enumerate(vals) if v == m]


class MCAgent(object):
    def __init__(self, mark, alpha, epsilon):
        self.mark = mark
        self.alpha = alpha
        self.epsilon = epsilon



    def act(self, state, actions):
        return self.egreedy(state, actions)



    def egreedy(self, state, actions):
        

        if not state in Q:
            Q[state] = default_actions.copy()


        r = random.random()
        if r < self.epsilon:
            return self.get_random_action(actions)
        else:
            return self.get_best_action(state, actions)



    def get_random_action(self, actions):
        return random.choice(actions)

    def get_best_action(self, state, actions):
        "Returns the best action given a current state and available actions"
        values = []

        for action in actions:
            action_val = self.get_state_val(state, action)
            values.append(action_val)


        if self.mark == 'O':
            indices = get_max_indices(values)
        else:
            indices = get_min_indices(values)

        return random.choice(indices)


    def get_state_val(self, state, action):
        return Q[state][action]
    




def learn(alpha, epsilon):
    global Q
    Q = {}

    env = TicTacToeEnv()
    agents = [MCAgent('O', alpha, epsilon),
                MCAgent('X', alpha, epsilon)]
    
    start_mark = 'O'
    reward_list = []
    for i in range(EPISODE_COUNT):

        env.show_episode(False, i)
        env.set_start_mark(start_mark)
        state, mark = env.reset()

        done = False
        state_action_list = []
        while not done:
            agent = agent_by_mark(agents, mark)
            actions = env.available_actions()
            action = agent.act(state, actions)

            env.show_turn(False, mark)

            if mark == 'O':
                state_action_list.append((state, action))

            state, reward, done, info = env.step(action)
            state, mark = state
        
        env.show_result(False, mark, reward)
        for state, action in state_action_list:
            Q[state][action] += alpha * (reward - Q[state][action])

        reward_list.append(reward)
        start_mark = next_mark(start_mark)
    print(reward_list[-10:])


learn(0.3, 0.1)


            






    
