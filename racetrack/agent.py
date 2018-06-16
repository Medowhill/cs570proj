import tensorflow as tf
import numpy as np
import random
import time

from game import Game
from model import DQN


tf.app.flags.DEFINE_boolean("train", False, "Training Mode")
tf.app.flags.DEFINE_boolean("cont", False, "Continuous Training Mode")
tf.app.flags.DEFINE_boolean("rand", False, "Random Play Mode")
tf.app.flags.DEFINE_string("track", "tracks/barto-small.track", "Map file name")
FLAGS = tf.app.flags.FLAGS

MAX_EPISODE = 1000000
TARGET_UPDATE_INTERVAL = 1000
TRAIN_INTERVAL = 4
CHANNEL = 4

# 0: nop, 1: up, 2: up_right, 3: right, 4: down_right, 5: down, 6: down_left, 7: left, 8: up_left
NUM_ACTION = 9

def train(track, width, height, cont):
    sess = tf.Session()

    game = Game(track, width, height, show_game=False)
    brain = DQN(sess, width, height, CHANNEL, NUM_ACTION)

    rewards = tf.placeholder(tf.float32, [None])
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    saver = tf.train.Saver()
    if cont:
        ckpt = tf.train.get_checkpoint_state('model')
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('logs', sess.graph)
    summary_merged = tf.summary.merge_all()

    brain.update_target_network()

    epsilon = 1.0
    time_step = 0
    total_reward_list = []

    if cont:
        OBSERVE = 100
    else:
        OBSERVE = 5000

    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        state = game.reset()
        brain.init_state(state)

        if episode > OBSERVE:
            epsilon = 2000 / episode

        while not terminal:
            if np.random.rand() < epsilon:
                action = random.randrange(NUM_ACTION)
            else:
                action = brain.get_action()

            state, reward, terminal = game.step(action)
            total_reward += reward

            brain.remember(state, action, reward, terminal)

            if episode > OBSERVE and time_step % TRAIN_INTERVAL == 0:
                brain.train()

            if episode > OBSERVE and time_step % TARGET_UPDATE_INTERVAL == 0:
                brain.update_target_network()

            time_step += 1

        if episode % 10 == 0:
            print('Games: %d Score: %d' % (episode + 1, total_reward))

        total_reward_list.append(total_reward)

        if episode % 10 == 0:
            summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
            writer.add_summary(summary, time_step)
            total_reward_list = []

        if episode > OBSERVE and episode % 10000 == 0:
            saver.save(sess, 'model/dqn.ckpt', global_step=episode)


def replay(track, width, height, rand):
    sess = tf.Session()

    game = Game(track, width, height, show_game=True)
    brain = DQN(sess, width, height, CHANNEL, NUM_ACTION)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        state = game.reset()
        brain.init_state(state)

        while not terminal:
            if rand and np.random.rand() < 0.1:
                action = random.randrange(NUM_ACTION)
            else:
                action = brain.get_action()

            state, reward, terminal = game.step(action)
            total_reward += reward

            brain.remember(state, action, reward, terminal)

            time.sleep(0.15)

        print('Games: %d Score: %d' % (episode + 1, total_reward))


def main(_):
    with open(FLAGS.track) as f:
        data = f.readlines()
        width = int(data[0])
        height = int(data[1])
        track = data[2:]

        if FLAGS.cont:
            train(track, width, height, True)
        elif FLAGS.train:
            train(track, width, height, False)
        elif FLAGS.rand:
            replay(track, width, height, True)
        else:
            replay(track, width, height, False)


if __name__ == '__main__':
    tf.app.run()
