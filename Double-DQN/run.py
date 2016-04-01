from CNN            import CNN
from double_dqn     import DDQN
from time           import time
from collections    import defaultdict as dd
import matplotlib.pyplot as plt
import tensorflow as tf
import cPickle as pickle
import numpy as np
import ale_python_interface
import os, sys, tty, termios
import random
import cv2


ROM_PATH        = '../../Roms/'
GAME            = 'space_invaders'
OUTPUT_SIZE     = 4     # Number of possible actions; game dependent
IMAGE_SIZE      = 80    # Image height & width
USER_INP        = 0.    # Number of time steps to allow user input for actions
DISPLAY         = False # Whether to display the game screen or not (much faster if false)

# Hyperparameters
GAMMA           = 0.95  # Decay rate of past observations
EXPLORE         = 5000  # Games over which to anneal epsilon
INITIAL_EPSILON = .95   # Starting value of epsilon
FINAL_EPSILON   = 0.07  # Final value of epsilon
BATCH_SIZE      = 100   # Number of experiences in optimization batch
K               = 7     # Select an action every Kth frame, repeat previous for others
S_TERM_MEMORY   = 4     # Number of previous frames in each network eval (short term memory)
L_TERM_MEMORY   = 1e4   # Number of previous frames available for experiential learning



def get_key():
    ''' User input without needing to press enter '''
    actions = { '' :0, 'a':1, 's':4, 'd':3, 'w':2, 'q':5}
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    if ch == 'p': assert(0) # Quit key
    return actions[ch] if ch in actions else 0


class GameState(object):
    ''' Wrapper class to manage ALE game emulator '''
    def __init__(self):
        ale = ale_python_interface.ALEInterface()
        ale.setInt('random_seed', 42)
        ale.setBool('display_screen', DISPLAY)
        ale.loadROM(ROM_PATH + GAME + '.bin')
        self.ale     = ale
        self.actions = ale.getMinimalActionSet()
        self.last_score = self.score = 0

    def frame_step(self, action):
        ''' Step forward a frame by performing <action> and
            determine reward via game score. Game Over has 
            the reward -<total reward> - 20 '''
        idx = list(action == 1).index(True)
        action = self.actions[idx]
        reward = self.ale.act(action)
        self.score += reward

        frame = self.ale.getScreenGrayscale().reshape((210, 160))
        terminal = self.ale.game_over()

        if terminal: 
            self.ale.reset_game()
            reward = -self.score - 20
            self.score, self.last_score = 0, self.score

        return frame, reward, int(terminal)


def run(session, dqn, graph_func, S):
    ''' Trains a Q-Network '''
    saver, writer, merger = graph_func  # Tensorboard helpers
    ms = (time(), S['t'])               # Time variables
    s_max = 0
    
    # Initialize game
    game_state = GameState()

    # Initialize live progress plots
    _, (ax1, ax3) = plt.subplots(2, sharex=False)
    ax2 = ax1.twinx(); ax4 = ax3.twinx()
    plt.ion()
    plt.pause(1e-10)

    def phi(x_next):
        ''' Apply preprocessing to sequence t+1. Currently just resizing
            the image to correct dimensions and adding a channel axis '''
        # phi_next = cv2.cvtColor(cv2.resize(x_next, (80, 80)), cv2.COLOR_BGR2GRAY)
        return cv2.resize(x_next, (IMAGE_SIZE, IMAGE_SIZE))#[..., np.newaxis] 

    # for episode = 1..M:
    while True:
        print 'Game number %i, timestep number %i (%.2f steps/s)' \
                % (S['game_num'], S['t'], (S['t'] - ms[1]) / (time() - ms[0]))

        # Set state variables
        ms = (time(), S['t'])
        Rs = []; Qs = []
        S['game_num'] += 1
        frame_num = 0

        # Populate short term memory
        experience = np.zeros((IMAGE_SIZE, IMAGE_SIZE, S_TERM_MEMORY))
        a_none     = np.zeros(OUTPUT_SIZE)
        a_none[0]  = 1
        for m in range(S_TERM_MEMORY):
            x_next, reward, terminal = game_state.frame_step(a_none)
            experience[..., m]  = phi(x_next)
        last_action = 0

        # for t = 1..T: 
        while not terminal:
            S['t']    += 1
            frame_num += 1

            # Repeat action for K frames
            if frame_num % K == 0:
                # act
                action      = dqn.action(experience[np.newaxis, ...])
                act_array   = np.zeros(OUTPUT_SIZE)
                act_array[action] = 1
                x_next, reward, terminal = game_state.frame_step(act_array)

                # store last transition
                exp_next = np.roll(experience, shift=1, axis=-1)
                exp_next[..., 0] = phi(x_next)
                dqn.store(experience, last_action, reward, exp_next)                

                #train
                dqn.training_step()

                # update current state as last state.
                last_action = action
                experience  = exp_next

            else: 
                act_array   = np.zeros(OUTPUT_SIZE)
                act_array[last_action] = 1
                _,_,terminal = game_state.frame_step(act_array)

            
            # Plot progress
            Rs.append( reward )
            if terminal: 
                s_max = game_state.last_score 
            if not S['t']%500:
                S['rewards_max'].append(max(Rs))
                S['scores_max'].append(s_max)
                # S['Q_max'].append(max(Qs) if Qs else 0)
                # S['Q_min'].append(min(Qs) if Qs else 0)
                Rs = []; Qs = []

                # Clear axes and redraw graph
                # ax1.cla(); plt.sca(ax1); plt.plot(S['Q_max'], 'r')
                # ax2.cla(); plt.sca(ax2); plt.plot(S['Q_min'], 'g', alpha=.5)
                ax3.cla(); plt.sca(ax3); plt.plot(S['rewards_max'], 'r')
                ax4.cla(); plt.sca(ax4); plt.plot(S['scores_max'], 'g')
                plt.pause(1e-10)
            
            # Save progress
            if S['t'] % 10000 == 0:
                saver.save(session, 'saved_networks/' + GAME + '-dqn', global_step=S['t'])
                with open('saved_networks/'+GAME+'-data.pkl','wb+') as f: pickle.dump(S, f)


def load_network(sess, GAME):
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    data = dd(list)
    data['game_num'] = 0
    data['t']        = 0

    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        with open('saved_networks/' + GAME + '-data.pkl', 'rb') as f:
            data = pickle.load(f)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"
    return saver, data        


def main():
    session     = tf.InteractiveSession()
    optimizer   = tf.train.RMSPropOptimizer(learning_rate= 0.001, decay=0.9)
    network     = CNN(OUTPUT_SIZE)
    input_shape = [IMAGE_SIZE, IMAGE_SIZE, S_TERM_MEMORY]
    
    # Create loggers for the graph
    writer = merger = None

    # logdir = os.path.dirname(os.path.realpath(__file__)) + '/log'
    # writer = tf.train.SummaryWriter(logdir, session.graph_def)

    # Create the tensorflow graph
    dqn = DDQN(input_shape, OUTPUT_SIZE, network, optimizer, session,
               random_action_probability    = FINAL_EPSILON,
               exploration_period           = EXPLORE,
               minibatch_size               = BATCH_SIZE,
               discount_rate                = GAMMA,
               max_experience               = L_TERM_MEMORY,
               summary_writer               = writer )

    # Load previous network state if available
    saver, data = load_network(session, GAME)
    if data['t'] == 0: 
        data['epsilon'] = INITIAL_EPSILON

    # Train the network
    # merger = tf.merge_all_summaries()
    run(session, dqn, (saver, writer, merger), data)


if __name__ == "__main__":
    main()
