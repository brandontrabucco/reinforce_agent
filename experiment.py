import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from gym import wrappers

MY_SEED = 126
random.seed(MY_SEED)
np.random.seed(MY_SEED)
tf.set_random_seed(MY_SEED)

class ReinforceAgent(object):
    def __init__(self, action_space, state_space, learning_rate=0.1):
        """Implements the REINFORCE Policy Gradient."""
        self.states  = tf.placeholder(tf.float64, name="states",  shape=[None, sum(state_space.shape)])
        self.actions = tf.placeholder(tf.int64,   name="actions", shape=[None])
        self.rewards = tf.placeholder(tf.float64, name="rewards", shape=[None])
        self.logits = tf.layers.dense(self.states, action_space.n)
        self.samples = tf.squeeze(tf.multinomial(self.logits, 1), axis=1)
        tf.losses.sparse_softmax_cross_entropy(self.actions, self.logits, weights=self.rewards)
        self.learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf.losses.get_total_loss())
        self.sess = tf.Session()
        self.init_op = tf.global_variables_initializer()
    def reset(self):
        """Initializes the neural network weights and bases."""
        self.sess.run(self.init_op)
    def act(self, states):
        """Args: states,  a  float32 tensor shape [None, state_size].
        Returns: actions, an int32   tensor shape [None]."""
        states = np.array(states)
        assert(len(states.shape) == 2 and states.shape[1] == self.states.shape[1])
        return self.sess.run(self.samples, feed_dict={self.states: states})
    def learn(self, states, actions, rewards, next_states):
        """Args: state,      a float32 tensor shape [state_size].
                 action,     a int32   tensor shape [].
                 reward,     a float32 tensor shape [].
                 next_state, a float32 tensor shape [state_size].
        Returns: None."""
        states, actions, rewards, next_states = (np.array(states), 
            np.array(actions), np.array(rewards), np.array(next_states))
        assert(len(states.shape)  == 2 and states.shape[1]  == self.states.shape[1])
        assert(len(actions.shape) == 1 and states.shape[0]  == actions.shape[0])
        assert(len(rewards.shape) == 1 and actions.shape[0] == rewards.shape[0])
        self.sess.run(self.learning_step, feed_dict={
            self.states: states, self.actions: actions, self.rewards: rewards})

class BufferOfSamples(object):
    def __init__(self, capacity, batch_size, use_future=True):
        """Implements a buffer of samples from the Environment."""
        self.capacity = capacity
        self.batch_size = batch_size
        self.use_future = use_future
        self.empty()
    def empty(self):
        """Clears the current buffer of samples."""
        self._buffer = []
        self._episode = []
    def __len__(self):
        """Returns: the total length of the buffer."""
        return len(self._buffer) + len(self._episode)
    def ratio(self):
        """Returns: the fraction of the buffer that is used."""
        return len(self) / self.capacity
    def is_full(self):
        """Returns: whether the bufefr is at capacity."""
        return len(self) >= self.capacity
    def sample(self):
        """Returns: a list of (state, action, reward) tuples"""
        return random.sample(self._buffer, self.batch_size)
    def shrink(self):
        """Shrinks the buffer by one sample if necessary."""
        if self.is_full():
            if len(self._buffer) > 0:
                self._buffer = self._buffer[1:]
            else:
                self._episode = self._episode[1:]
    def add(self, state, action, reward, next_state):
        """Args: state,      a float32 tensor shape [state_size].
                 action,     a int32   tensor shape [].
                 reward,     a float32 tensor shape [].
                 next_state, a float32 tensor shape [state_size].
        Returns: None."""
        self.shrink()
        if self.use_future:
            for content in self._episode:
                content[2] = content[2] + reward
        self._episode = self._episode + [[state, action, reward, next_state]]
    def episode(self):
        """Flags that an episode of samples has finished being collected."""
        self._buffer = self._buffer + self._episode
        self._episode = []

def plot(*means_stds_name, title="", xlabel="", ylabel=""):
    """Generate a colorful plot with the provided data."""
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for indices, means, stds, name in means_stds_name:
        ccolor = np.random.rand(3,)
        plt.fill_between(indices, means - stds, means + stds, color=np.hstack([ccolor, [0.2]]))
        plt.plot(indices,means, color=np.hstack([ccolor, [1.0]]), label=name)
    plt.legend(loc=4)
    plt.savefig(title + ".png")
    plt.close()

def main(args):
    """Run a simulation of the specified task in openAI gym."""
    env = gym.make(args.env_id)
    env.seed(0)
    agent = ReinforceAgent(env.action_space, env.observation_space)
    buffer = BufferOfSamples(args.buffer_size, args.batch_size)
    logged_trials = []
    for t in range(args.num_trials):
        logged_rewards = []
        agent.reset()
        for i in range(args.training_steps):
            while not buffer.is_full():
                state = env.reset()
                done = False
                while not done and not buffer.is_full():
                    action = agent.act([state])[0]
                    next_state, reward, done, _info = env.step(action)
                    buffer.add(state, action, reward, next_state)
                    state = next_state
                    print("\rBuffer fill ratio: {0:.2f} / {1:.2f}".format(buffer.ratio(), 1.0), end="\r")
                buffer.episode()
            states, actions, rewards, next_states = zip(*buffer.sample())
            agent.learn(states, actions, rewards, next_states)
            buffer.empty()
            logged_rewards.append(np.mean(rewards))
            print("On training step {0} average utility was {1}.".format(i, logged_rewards[-1]))
        logged_trials.append(logged_rewards)
    env.close()
    trajectories = np.array(logged_trials)
    plot(
        (np.arange(args.training_steps), np.mean(trajectories, axis=0), 
            np.std(trajectories, axis=0), "REINFORCE Policy Gradient"),
        title="Training A REINFORCE Policy On {0}".format(args.env_id), 
        xlabel="Iteration", 
        ylabel="Expected Future Reward")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', type=str, default='CartPole-v0')
    parser.add_argument('--training_steps', type=int, default=100)
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=512)
    main(parser.parse_args())
