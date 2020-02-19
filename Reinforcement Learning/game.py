from __future__ import division
import random
from numpy import transpose as tp, array, arange
from math import cos, sin, radians, pi, floor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

cartPole_statelist = []
PUSH_WEIGHT = 5

DEBUG = True
def debug(*args):
    if DEBUG:
        print ' '.join(map(str, args))
"""
Main Class hosts Game
Agent:          Student's agent
Controller:     Controller Class (either wallE or cartPole)
World:          Tells the program what world to expect (for graphics)
"""
class Game:
    def __init__(self, agent, controller, world):
        self.agent = agent
        self.controller = controller
        self.total_reward = 0.0
        self.discounted_reward = 0.0
        self.moves = 0
        self.games = 0
        self.path_length = 0
        self.world = 0
        if world == "wallE":
            self.world = 1

    def run(self):
        if self.world:
            if gfx: gfx.move_player(self.controller.start)

        state = self.controller.start
        if not self.world:
            cartPole_statelist.append(state)

        iteration = 0
        discount = 1.0
        path_length = 0

        # moves = 0
        while True:
            # Get action from agent
            action = self.agent.get_action(state)
            # get reward, next_state and state status (terminated?) from controller
            reward, next_state, terminated = self.controller.transition(state, action)
            # update the agent
            self.agent.update(state, action, reward, next_state)
            if self.world:
                if gfx: gfx.update()
            #else:
            #    cartPole_statelist.append(next_state)

            self.total_reward += reward
            self.discounted_reward += reward * discount
            discount *= gamma

            iteration += 1
            self.moves += 1
            path_length += 1
            state = next_state

            if terminated:
                debug('Success: reached end')
                debug('iterations/total:', '%d/%d' % (iteration, self.moves))
                debug('total reward:', int(round(self.total_reward)))
                debug('total discounted reward:', int(round(self.discounted_reward)))
                debug('')
                self.games += 1
                self.path_length = path_length+1
                return True

            if self.moves > max_moves:
                debug('Reached max_moves')
                debug('iterations/total:', '%d/%d' % (iteration, self.moves))
                debug('total reward:', int(round(self.total_reward)))
                debug('total discounted reward:', int(round(self.discounted_reward)))
                debug('')
                self.games += 1
                return False

# CartPole Controller
class CartPoleController:
    def __init__(self, mc=1, mp=0.1, l=0.5, g=9.8):
        # Initialize a CartPole state
        self.start = CartPole(mc, mp, l, g)
        self.cartPole = self.start

        # Set the parameters at which the state has "failed" 
        self.fail_angle = radians(20)
        self.fail_dist = 2

        # Set the discretization parameters
        self.low_h = -2.5
        self.high_h = 2.5 
        self.step_h = .1 
        self.low_h_dot = -1
        self.high_h_dot = 1 
        self.step_h_dot = .1
        self.low_theta = -pi
        self.high_theta = pi
        self.step_theta = pi/10
        self.low_theta_dot = -pi
        self.high_theta_dot = pi
        self.step_theta_dot = 2*pi/10 
        self.discrete_states = self.cartPole.discrete_states(mc, mp, l, g, self.low_h, self.high_h, 
                                                                self.step_h, self.low_h_dot, 
                                                                self.high_h_dot, self.step_h_dot, 
                                                                self.low_theta, self.high_theta, 
                                                                self.step_theta, self.low_theta_dot, 
                                                                self.high_theta_dot, 
                                                                self.step_theta_dot)
        self.fail_state = CartPole(mc, mp, l, g, t=-2)

    # Calculate the bin num for a val (called by calculate_bin)
    def calculate_individual_bin(self, val, low, high, step):
        bin_num = (val - low)/step
        if val < low:
            bin_num = 0
        if val >= high:
            bin_num = high/step - 1
        return int(floor(bin_num))

    # Calculate the bin num for a state
    def calculate_bin(self, state, low_h, high_h, step_h, low_h_dot, high_h_dot, step_h_dot, low_theta, 
                        high_theta, step_theta, low_theta_dot, high_theta_dot, step_theta_dot):
        h = state.h
        h_dot = state.h_dot
        theta = state.theta
        theta_dot = state.theta_dot

        h_bin = self.calculate_individual_bin(h, low_h, high_h, step_h)
        h_dot_bin = self.calculate_individual_bin(h_dot, low_h_dot, high_h_dot, step_h_dot)
        theta_bin = self.calculate_individual_bin(theta, low_theta, high_theta, step_theta)
        theta_dot_bin = self.calculate_individual_bin(theta_dot, low_theta_dot, high_theta_dot, 
                                                    step_theta_dot)
        num_h_dot = (high_h_dot - low_h_dot)/step_h_dot
        num_theta = (high_theta - low_theta)/step_theta
        num_theta_dot = (high_theta_dot - low_theta_dot)/step_theta_dot

        bin = (h_bin * num_h_dot * num_theta * num_theta_dot + h_dot_bin * num_theta * num_theta_dot + theta_bin * num_theta_dot + theta_dot_bin)
        return int(bin)

    # a reward function
    def reward_riedmiller(self, state):
        if not self.upright(state):
            return -2 * (200 - state.t)
        elif abs(state.theta) <= 0.05 and abs(state.h) < 0.5:
            return 0
        else:
            return -1
    
    # a reward function       
    def reward(self, state):
        if not self.upright(state):
            return -1 * (200 - state.t)

        if abs(state.theta) < radians(12) and abs(state.h) < 2.4:
            return 0
        else:
            return -1

    # Is the pole still upright?
    def upright(self, state):
        if state.h > self.fail_dist or state.h < -self.fail_dist or state.theta > self.fail_angle or state.theta < - self.fail_angle:
            return False
        return True

    # s doesn't do anything, it's just to make transition look the same as in the WalleController
    # transitions from teh current state to the next state through the provided action
    def transition(self, s, a):
        action = 0
        if a == Action.right:
            action = PUSH_WEIGHT
        else:
            action = -PUSH_WEIGHT
        state = self.cartPole.next_state(action)
        self.cartPole = state
        rwrd = self.reward(state)
        
        bin = self.calculate_bin(state, self.low_h, self.high_h, self.step_h, self.low_h_dot, 
                            self.high_h_dot, self.step_h_dot, self.low_theta, self.high_theta, 
                            self.step_theta, self.low_theta_dot, self.high_theta_dot, 
                            self.step_theta_dot)
        disc_state = 0
        f = self.upright(state)
        if f:
            disc_state = self.discrete_states[bin]
        else:
            disc_state = self.fail_state
            state.reset()
        cartPole_statelist.append(state)
        return rwrd, disc_state, not f

# The CartPole state: has a physics engine which tells it how to update itself when provided
# an action
class CartPole():
    def __init__(self, mc, mp, l, g, t=0, h=0, h_dot=0, theta=0, theta_dot=0):
        self.mc = mc
        self.mp = mp
        self.l = l
        self.g = g
        self.mass = self.mc + self.mp

        self.t = t
        self.h = h
        self.h_dot = h_dot
        self.theta = theta
        self.theta_dot = theta_dot

    def __repr__(self):
        string = str(self.h) + ", " + str(self.h_dot) + ", " 
        string += str(self.theta) + ", " + str(self.theta_dot)
        return string

    # Reset the state (when the pole falls over, aka after the fail state)
    def reset(self):
        self.t = 0
        self.h = 0
        self.h_dot = 0
        self.theta = 0
        self.theta_dot = 0

    # determines angular acceleration
    def theta_dot_dot(self, push_t):

        push_term = (- push_t - (self.mp * self.l * pow(self.theta_dot, 2) 
                                 * sin(self.theta))) / self.mass
        n = self.g * sin(self.theta) + cos(self.theta) * push_term


        d = (self.l * 
             (4./3. - ((self.mp * pow(cos(self.theta), 2)) / self.mass)))
        return n / d
               
    # determines the acceleration of the cart
    def h_dot_dot(self, push_t, theta_dot_dot):
        
        h_dot_dot = (push_t + 
                     (self.mp * self.l * 
                      (pow(self.theta_dot, 2) * sin(self.theta) - 
                       theta_dot_dot * cos(self.theta)))) / (self.mass)
        return h_dot_dot

    # Calculates the next state based on the push
    # tau = time step
    def next_state(self, action, tau=0.01):
        theta_dot_dot = self.theta_dot_dot(action)
        h_dot_dot = self.h_dot_dot(action, theta_dot_dot)
        
        next_t = self.t + 1     # NOT the actual time, its the decision time step
        next_h = self.h + tau * self.h_dot
        next_h_dot = self.h_dot + tau * h_dot_dot
        next_theta = self.theta + tau * self.theta_dot
        next_theta_dot = self.theta_dot + tau * theta_dot_dot
        return CartPole(self.mc, self.mp, self.l, self.g, 
                        next_t, next_h, next_h_dot, next_theta,
                        next_theta_dot)

    # Generates the discrete states
    @staticmethod
    def discrete_states(mc, mp, l, g, low_h, high_h, step_h, low_h_dot, high_h_dot, step_h_dot, 
                        low_theta, high_theta, step_theta, low_theta_dot, 
                        high_theta_dot, step_theta_dot):
        states = []
        
        # Initialize all discretized states (as a range)        
        for h in arange(low_h, high_h, step_h):
            for h_dot in arange(low_h_dot, high_h_dot, step_h_dot):
                for theta in arange(low_theta, high_theta, step_theta):
                    for theta_dot in arange(low_theta_dot, high_theta_dot, step_theta_dot):
                        states.append(CartPole(mc, mp, l, g, -1,
                                            h, h_dot, theta, theta_dot))
        return states

# Previously Level; controls WallE
class WallEController():
    def __init__(self, squares):
        self.squares = squares
        self.rows = len(squares)
        self.cols = len(squares[0])
        self.goal_state = ("GOAL STATE",)

        for x in range(self.rows):
            for y in range(self.cols):
                if self.is_start((x,y)): self.start = (x,y)
                if self.is_end((x,y)): self.end = (x,y)

    @classmethod
    def load_level(cls, filename):
        squares = []
        for row in file(filename):
            squares.append(row.strip('\n'))
        return WallEController(squares)

    def is_ladder(self, state):
        (x,y) = state
        return self.squares[x][y] == "H"

    def is_floor(self, state):
        (x,y) = state
        return self.squares[x][y] in "_S"

    def is_gap(self, state):
        (x,y) = state
        return self.squares[x][y] == ' '

    def is_trapdoor(self, state):
        (x,y) = state
        return self.squares[x][y] in "123456789"

    def fall_prob(self, state):
        (x,y) = state
        if self.is_gap(state):
            return 1.0
        if self.is_trapdoor(state):
            return int(self.squares[x][y]) / 10.0
        else:
            return 0.0

    def is_start(self, state):
        (x,y) = state
        return self.squares[x][y] == 'S'

    def is_end(self, state):
        (x,y) = state
        return self.squares[x][y] == 'X'

    def is_goal_state(self, state):
        return state == self.goal_state

    def transition(self, state, action):
        # should never get here
        if self.is_goal_state(state):
            return 0, state, True

        (row,col) = state
        if random.random() < self.fall_prob(state):
            if row == self.rows - 1:
                raise ValueError("can't fall through bottom!")
            if gfx: gfx.player.fall()
            return 0, (row + 1, col), False
        else:
            if col == 0 and (action is Action.left or action is Action.jump_left):
                return -1, state, False
            if col == self.cols - 1 and (action is Action.right or action is Action.jump_right):
                return -1, state, False

            if col == 1 and action is Action.jump_left:
                return -1, state, False
            if col == self.cols - 2 and action is Action.jump_right:
                return -1, state, False


            if action is Action.left:
                if gfx: gfx.player.side(True, False)
                return -1, (row, col - 1), False
            if action is Action.jump_left:
                if gfx: gfx.player.side(True, True)             
                return -1, (row, col - 2), False
            if action is Action.right:
                if gfx: gfx.player.side(False, False)
                return -1, (row, col + 1), False
            if action is Action.jump_right:
                if gfx: gfx.player.side(False, True)
                return -1, (row, col + 2), False

            if action is Action.climb:
                if self.is_ladder(state):
                    if gfx: gfx.player.climb()
                    return -1, (row - 1, col), False
                if self.is_end(state):
                    return 100, self.goal_state, True
                else:
                    return -1, state, False


# Actions!
class Action:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

Action.left = Action("left")
Action.right = Action("right")
Action.jump_left = Action("jump left")
Action.jump_right = Action("jump right")
Action.climb = Action("climb")

# Our hacky visualization for CartPole
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(0, .7))
ax.grid()

pole, = ax.plot([], [], 'bo-', lw=2)
cart, = ax.plot([], [], 'g-', lw=3)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
time_run = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    pole.set_data([], [])
    cart.set_data([], [])
    time_text.set_text('')
    time_run.set_text('')
    return cart, pole, time_text, time_run

def animate(i):
    cart_height = 0.1
    ch = cart_height
    cart_X, cart_Y = tp([(-0.1, 0), (0.1, 0), (0.1, ch), (-0.1, ch),
                         (-0.1, 0)])
                         
    cart.set_data(cart_X + cartPole_statelist[i].h, cart_Y)

    pole_X, pole_Y = tp([(cartPole_statelist[i].h, ch), (cartPole_statelist[i].l * 
                                        sin(cartPole_statelist[i].theta) + 
                                        cartPole_statelist[i].h, cartPole_statelist[i].l * 
                                        cos(cartPole_statelist[i].theta) + ch)])

    pole.set_data(pole_X, pole_Y)

    time_text.set_text(i)
    time_run.set_text(cartPole_statelist[i].t)
    return cart, pole, time_text, time_run


if __name__ == '__main__':
    import agent
    import sys
    import argparse

    p=argparse.ArgumentParser(description="reinforcement learning")
    
    p.add_argument('agent', type=str, help='Qlearner | Rmax | Q | R')
    p.add_argument('-level', type=str, default='easy.lev', help='<a .lev file>')
    p.add_argument('-nographics', action='store_true', help="don't display graphics")
    p.add_argument('-silent', action='store_true', help="don't print stats during game")
    p.add_argument('-delay', type=int, default=0, help='<in seconds>')
    p.add_argument('-eps', type=float, default=0.01, help='a probability')
    p.add_argument('-m', type=int, default=2, help='a positive integer')
    p.add_argument('-iters', type=int, default=10000, help='maximum number of moves agent can make in world')
    
    p.add_argument('-start', type=int, default=0, help='start of cartPole animation')
    p.add_argument('-world', type=str, default="wallE", help="wallE or cartPole")

    args=p.parse_args()

    if args.agent == "Q": args.agent = 'Qlearner'
    if args.agent == "R": args.agent = 'Rmax'
    if args.agent == "walle": args.agent = 'wallE'
    if args.agent == "cartpole": args.agent = 'cartPole'
    if args.agent == "curiosity": args.agent = 'cartPole'
    learner_name = args.agent
    level_name = 'levels/'+args.level
    delay = args.delay
    graphics_off = args.nographics
    m = args.m
    epsilon = args.eps
    max_moves = args.iters
    if args.silent:
        DEBUG = False
    controller = 0
    Action.actions = []
    if args.world == "wallE":
        controller = WallEController.load_level(level_name)
        Action.actions = [Action.left, Action.right, Action.jump_left, Action.jump_right, Action.climb]
    else:
        controller = CartPoleController()
        Action.actions = [Action.left, Action.right]
    if graphics_off:
        gfx = None
    elif args.world == "wallE":
        import platform
        gfx = platform.Graphics(controller.squares)
        platform.DELAY = delay

    gamma = 0.9

    if learner_name == 'Qlearner':
        learner = agent.Qlearner(alpha=0.9, gamma=gamma, actions=Action.actions, epsilon=epsilon)
    elif learner_name == 'Rmax':
        learner = agent.Rmax(rmax=1000, gamma=gamma, m=m, actions=Action.actions)

    game = Game(learner, controller, args.world)
    while game.run():
        pass

    print 'total moves:', game.moves
    print 'average discounted reward (gamma = %.3f): %d' % (gamma, int(round(game.discounted_reward / game.games)))
    print 'optimal path length (according to learning algorithm):', game.path_length

    if args.world == "cartPole":
        cartPole_statelist = cartPole_statelist[args.start:args.iters + 1]
    # Our hacky animation for CartPole
        ani = animation.FuncAnimation(fig, animate, game.moves, interval=25, blit=True, repeat=False, 
                                    init_func=init)

        #ani.save('double_pendulum.mp4', fps=15, clear_temp=True)
        plt.show()
