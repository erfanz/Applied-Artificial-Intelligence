from __future__ import division # makes 1/2 equal float 0.5 and not integer 0
import random
from collections import defaultdict
import sys
import copy
import time


class Qlearner:
    """ The QLearner agent.
    Attributes:
        alpha:      learning rate
        gamma:      future rate discount
        actions:    actions available to the agent (a list of general "action" objects)
        epsilon:    exploration parameter
    """

    def __init__(self, alpha, gamma, actions, epsilon):
        """ Inits Qlearner agent with parameters
        Args:
            alpha, gamma, epsilon:  see above
            actions:                actions available to the agent (a list of general actions)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.state_actions = dict()
        

    def update(self, s, a, r, s_):
        """ Updates the agent's knowledge.
        Args:
            s:  previous state (general "state")
            a:  action taken (general action)
            r:  reward received
            s_: resultant state
        
        Returns:
            nothing
        """
        best_next_action_Q = 0    # this is the default (i.e. either the next state is a new state or the best action in that state has the Q of 0 )
        if s_ in self.state_actions:
             best_action = self.find_best_action(s_)
             best_next_action_Q = self.state_actions[s_][best_action]
        
        
        self.state_actions[s][a] = self.state_actions[s][a] + self.alpha * (r + self.gamma * best_next_action_Q - self.state_actions[s][a])
        
    
    def get_action(self, s):
        """ Gets the best action based on the agent's knowledge and algorithm.
        Args:
            s:  current state
        
        Returns:
            an action (default actions must be randomized)
        """
        
        if s in self.state_actions:
            if random.random() < self.epsilon:
                # explore
                rand_action = random.randrange(len(self.actions))
                return self.state_actions[s].keys()[rand_action]
            else:
                # exploit
                return self.find_best_action(s)
        else:
            # first the state must be built 
            self.state_actions[s] = dict()
            for action in self.actions:
                self.state_actions[s][action] = 0    # initially, all Q values are set to zero
            
            rand_action = random.randrange(len(self.actions))
            return self.state_actions[s].keys()[rand_action]
    
    def find_best_action(self, s):
        """ Gets the best action for already-explored state (i.e. the state must be existing in the Q table)
        Args:
            s: current state
        
        Returns:
            Q value of the best action
        """
        max_q = self.state_actions[s][self.actions[0]]
        best_action = self.actions[0]
        for action in self.state_actions[s].keys():
            if self.state_actions[s][action] > max_q:
                max_q = self.state_actions[s][action]
                best_action = action
        
        return best_action

class Rmax:
    """ The Rmax agent.
    Attributes:
        rmax:       reward for going to or from the absorbing state
        gamma:      future rate discount
        m:          number of times a (state, action) pair needs to be attempted before it
                    computes a reward other than rmax
        actions:    actions available to the agent (a list of actions)
        s_r:        absorbing state
    """

    def __init__(self, rmax, gamma, m, actions):
        """ Inits Rmax agent with parameters.
        Args:
            rmax, gamma,m, actions: see above
        """
        
        self.rmax = rmax
        self.gamma = gamma
        self.m = m
        self.actions = actions
        self.s_r = "absorbing state"
        
        self.count_state_action = defaultdict(dict)
        self.rsum_state_action = defaultdict(dict)
        self.r_state_action = defaultdict(dict)
        self.count_to_s_state_action = defaultdict(dict)
        self.transition = defaultdict(dict)
        self.V_value = defaultdict(int)
        self.V_bestaction = defaultdict(int)
        self.visited = defaultdict(bool)
        
        for action in actions:
            self.r_state_action[self.s_r][action] = self.rmax
            self.transition[self.s_r][action] = defaultdict(dict)
            self.transition[self.s_r][action][self.s_r] = 1


    def update(self, s, a, r, s_):
        """ Updates the agent's knowledge.
        Args:
            s:  previous state
            a:  action taken
            r:  reward received
            s_: resultant state
        
        Returns:
            nothing
        """
        
        change_flag = False # shows whether the values of T and R have changed. If not, we don't need to run value_iteration()
        self.count_state_action[s][a] += 1
        
        if s_ in self.count_to_s_state_action[s][a]:
            self.count_to_s_state_action[s][a][s_] += 1
        else:
            self.count_to_s_state_action[s][a][s_] = 1
            
        self.rsum_state_action[s][a] += r
        if self.count_state_action[s][a] >= self.m:
            # Known state, update model using experience counts
            temp1 = self.r_state_action[s][a]
            self.r_state_action[s][a] = self.rsum_state_action[s][a] / self.count_state_action[s][a]
            if temp1 != self.r_state_action[s][a]:
                change_flag = True
            
            temp = list()
            
            self.transition[s][a][self.s_r] = 0 
            for destination in self.count_to_s_state_action[s][a].keys():
                if destination in self.transition[s][a]:
                    temp.append(self.transition[s][a][destination])
                else:
                    temp.append(-1)
                    
                self.transition[s][a][destination] = self.count_to_s_state_action[s][a][destination] / self.count_state_action[s][a]
                if temp[len(temp)-1] != self.transition[s][a][destination]:
                    change_flag = True
        
        else:
            # Unknown state, set optimistic model transition to absorbing state
            temp2 = self.r_state_action[s][a]
            self.r_state_action[s][a] = self.rmax
            temp3 = self.transition[s][a][self.s_r]
            self.transition[s][a][self.s_r] = 1
            change_flag = True
        
        
        if  self.visited[(s,a, s_)] == False or change_flag:
            self.value_iteration()
            


    def get_action(self, s):
        """ Gets the best action based on the agent's knowledge and algorithm.
        Args:
            s:  current state
        
        Returns:
            returns an action (default actions must be randomized)
        """
        if s in self.r_state_action.keys():
            # already observed state
            return self.V_bestaction[s]
            
        else:
            # a new state. first the state must be built
            self.V_value[s] = 0
            self.V_bestaction[s] = None
            self.transition[s] = defaultdict(dict)
            for a in self.actions:
                self.count_state_action[s][a] = 0
                self.count_to_s_state_action[s][a] = defaultdict(dict)
                self.rsum_state_action[s][a] = 0
                self.r_state_action[s][a] = self.rmax
                self.transition[s][a] = defaultdict(dict)
                self.transition[s][a][self.s_r] = 1    
                
            
            rand_action = random.randrange(len(self.actions))
            return self.actions[rand_action]

  
    def value_iteration(self):
        """Solving an MDP by value iteration. [Fig. 17.4]"""
        R, T  = self.r_state_action, self.transition
        
        V1_value = self.V_value.copy()
        V1_bestaction = defaultdict(int)
        V1_bestaction = self.V_bestaction.copy() 
        

        while True:
            delta = 0
            for s in R.keys():
                if s == self.s_r:
                    V1_value[s] = self.rmax / (1 - self.gamma)
                else:
                    # find the best action
                    max_q = -sys.maxint - 1
                    best_action = None
                    for action in R[s].keys():
                        result = R[s][action]
                        temp = 0
                        for destination in T[s][action].keys():        
                            temp += T[s][action][destination] * self.V_value[destination]


                        result = result + self.gamma * temp
                                       
                        if result >= max_q:
                            if result > max_q or (result == max_q and random.random() > 0.5):
                                max_q = result
                                best_action = action

                    V1_value[s] = max_q
                    V1_bestaction[s] = best_action
                        
            delta = max(delta, abs(V1_value[s] - self.V_value[s]))

            if delta < (1/self.rmax) * (1 - self.gamma) / self.gamma:
            #if delta < (1/self.rmax):
                self.V_value = V1_value.copy()
                self.V_bestaction = V1_bestaction.copy()
                break

            self.V_value = V1_value.copy()
            self.V_bestaction = V1_bestaction.copy()