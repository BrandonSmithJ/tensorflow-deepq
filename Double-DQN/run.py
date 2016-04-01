from time import time
import matplotlib.pyplot as plt
import tensorflow as tf
import cPickle as pickle
import numpy as np
import ale_python_interface
import os, sys, tty, termios
import random
import cv2


ROM_PATH        = '../../ALE/Roms/'
GAME            = 'space_invaders'
OUTPUT_SIZE     = 4     # Number of possible actions; game dependent
IMAGE_SIZE      = 80    # Image height & width
USER_INP        = 0.    # Number of time steps to allow user input for actions

# Hyperparameters
GAMMA           = 0.95  # Decay rate of past observations
EXPLORE         = 500  # Games over which to anneal epsilon
INITIAL_EPSILON = .95   # Starting value of epsilon
FINAL_EPSILON   = 0.07  # Final value of epsilon
BATCH           = 100   # Number of experiences in optimization batch
K               = 7     # Select an action every Kth frame, repeat previous for others
S_TERM_MEMORY   = 4     # Number of previous frames in each network eval (short term memory)
L_TERM_MEMORY   = 3e5   # Number of previous frames available for experiential learning



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
        ale.setBool('display_screen', False)
        ale.loadROM(ROM_PATH + GAME + '.bin')
        self.ale     = ale
        self.actions = ale.getMinimalActionSet()
        self.last_score = self.score = 0

    def frame_step(self, action):
        ''' Step forward a frame by performing <action> and
            determine reward via game score. Game Over has 
            the reward -<total reward> (Unsure of ramifications..) '''
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


def QNetwork(sess, graph_func, placeholders, readout, train_step, S):
    ''' Trains a Q-Network '''
    saver, writer, merger = graph_func  # Tensorboard helpers
    ms = (time(), S['t'])               # Time variables
    D  = []                             # Long term memory
    s_max = 0
    
    # Initialize game
    game_state = GameState()

    # Initialize live progress plots
    _, (ax1, ax3) = plt.subplots(2, sharex=False)
    ax2 = ax1.twinx(); ax4 = ax3.twinx()
    plt.ion()
    plt.pause(1e-10)

    # Create helper functions
    readout_fd  = lambda phi_curr:  {placeholders['input_x']:phi_curr}  # Readout feed_dict
    train_fd    = lambda phi, y, a: {placeholders['input_x']:phi,       # Training step 
                                     placeholders['input_y']:y,
                                     placeholders['actions']:a}
    def phi(x_next):
        ''' Apply preprocessing to sequence t+1. Currently just resizing
            the image to correct dimensions and adding a channel axis '''
        # phi_next = cv2.cvtColor(cv2.resize(x_next, (80, 80)), cv2.COLOR_BGR2GRAY)
        return cv2.resize(x_next, (IMAGE_SIZE, IMAGE_SIZE))#[..., np.newaxis] 

    def play_frame(action, phi_curr):
        ''' Execute <action> in the game and store results in D '''
        x_next, r_curr, terminal = game_state.frame_step(action)

        # Add newest state to front of phi
        phi_next = np.roll(phi_curr, shift=1, axis=-1)
        phi_next[..., 0] = phi(x_next)

        # Store transition (phi_curr, phi_next, a_curr, r_curr, terminal state) in D 
        D.append((phi_curr, phi_next, action, r_curr, terminal))

        # Update values
        if len(D) > L_TERM_MEMORY: D.pop(0)
        return phi_next, terminal


    # for episode = 1..M:
    while True:
        print 'Game number %i, timestep number %i (%.2f steps/s), Epsilon %f' \
                % (S['game_num'], S['t'], (S['t'] - ms[1]) / (time() - ms[0]), S['epsilon'])

        # Set state variables
        ms = (time(), S['t'])
        Rs = []; Qs = []
        S['game_num'] += 1
        if S['epsilon'] > FINAL_EPSILON:
            S['epsilon'] -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # Populate short term memory
        phi_curr  = np.zeros((IMAGE_SIZE, IMAGE_SIZE, S_TERM_MEMORY))
        a_curr    = np.zeros(OUTPUT_SIZE)
        a_curr[0] = 1
        for _ in range(S_TERM_MEMORY+1):
            phi_curr, terminal = play_frame(a_curr, phi_curr)
        D = [] # All initial frames will have invalid phi_curr value

        # for t = 1..T: 
        while not terminal:
            S['t'] += 1
            a_curr = np.zeros(OUTPUT_SIZE)
            action_idx = random.randrange(OUTPUT_SIZE) 

            # Q-Network acts with probability 1-epsilon
            if random.random() >= S['epsilon']:
                readout_t   = readout.eval( feed_dict=readout_fd([phi_curr]) )[0]
                action_idx  = np.argmax(readout_t)
                Qs.append(np.max(readout_t))
            a_curr[action_idx] = 1

            # Repeat action a_curr for K frames
            for _ in range(K):
                phi_curr, terminal = play_frame(a_curr, phi_curr)
                if terminal: 
                    break

            # Sample random indices of states from D
            # indices = random.sample(range(S_TERM_MEMORY, len(D)), min(BATCH, len(D)-S_TERM_MEMORY))
            # states  = [(D[i], D[i-S_TERM_MEMORY+1:i]) for i in indices]
            states = random.sample(D, min(BATCH, len(D)))
            
            # # Get the batch variables
            # actions   = [c[2] for c,_ in states]
            # rewards   = [c[3] for c,_ in states]
            # terminals = [c[4] for c,_ in states]

            # # Gather images into one array for short term memory (current, prior)
            # phi_currs = [np.array([s[0] for s in p] + [c[0],]).T for c,p in states]
            # phi_nexts = [np.array([s[1] for s in p] + [c[1],]).T for c,p in states]
            phi_currs = [s[0] for s in states]
            phi_nexts = [s[1] for s in states]
            actions   = [s[2] for s in states]
            rewards   = [s[3] for s in states]
            terminals = [s[4] for s in states]

            # Set y_j = reward if terminal else reward + gamma * argmax Q(phi_next, a'; theta)
            a_batch = readout.eval( feed_dict=readout_fd(phi_nexts) )
            y_batch = [ rewards[i] + (not terminals[i]) * GAMMA * np.max(a_batch[i]) 
                        for i in range(len(states)) ]

            # Gradient optimization
            result = sess.run([merger, train_step], 
                              feed_dict=train_fd(phi_currs, y_batch, actions))
            writer.add_summary(result[0], S['t'])

            # Plot progress
            rewards = [s[3] for s in D[-K:]]
            Rs.append( max(rewards) )
            if terminal: s_max = game_state.last_score #max(game_state.score, -min(rewards))
            if not S['t']%500:
                S['rewards_max'].append(max(Rs))
                S['scores_max'].append(s_max)
                S['Q_max'].append(max(Qs) if Qs else 0)
                S['Q_min'].append(min(Qs) if Qs else 0)
                Rs = []; Qs = []

                # Clear axes and redraw graph
                ax1.cla(); plt.sca(ax1); plt.plot(S['Q_max'], 'r')
                ax2.cla(); plt.sca(ax2); plt.plot(S['Q_min'], 'g', alpha=.5)
                ax3.cla(); plt.sca(ax3); plt.plot(S['rewards_max'], 'r')
                ax4.cla(); plt.sca(ax4); plt.plot(S['scores_max'], 'g')
                plt.pause(1e-10)
            
            # Save progress
            if S['t'] % 10000 == 0:
                saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=S['t'])
                with open('saved_networks/'+GAME+'-data.pkl','wb+') as f: pickle.dump(S, f)

        


def main():
    # Define the inputs, and their respective type & size
    inputs = {  'input_x'   : [tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, S_TERM_MEMORY]],                
                'actions'   : [tf.float32, [None, OUTPUT_SIZE]],    
                'input_y'   : [tf.float32, [None]],                 
            }

    # Create the tensorflow graph
    sess, placeholders, readout, train_step = create_network(inputs, S_TERM_MEMORY, OUTPUT_SIZE)
    
    # Create loggers for the graph
    logdir = os.path.dirname(os.path.realpath(__file__)) + '/log'
    merger = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(logdir, sess.graph_def)

    # Load previous network state if available
    saver, data = load_network(sess, GAME)
    if data['t'] == 0: data['epsilon'] = INITIAL_EPSILON

    # Train the network
    QNetwork(sess, (saver, writer, merger), placeholders, readout, train_step, data)


if __name__ == "__main__":
    main()
