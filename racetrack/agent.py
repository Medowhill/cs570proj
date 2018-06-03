import tensorflow as tf
import numpy as np
import random
import time

from game import Game
from model import DQN


tf.app.flags.DEFINE_boolean("train", False, "Training mode")
tf.app.flags.DEFINE_string("track", "tracks/barto-small.track", "Map file name")
FLAGS = tf.app.flags.FLAGS

# 최대 학습 횟수
MAX_EPISODE = 100000000
# 1000번의 학습마다 한 번씩 타겟 네트웍을 업데이트합니다.
TARGET_UPDATE_INTERVAL = 1000
# 4 프레임마다 한 번씩 학습합니다.
TRAIN_INTERVAL = 4
# 학습 데이터를 어느정도 쌓은 후, 일정 시간 이후에 학습을 시작하도록 합니다.
OBSERVE = 5000

# 0: nop, 1: up, 2: up_right, 3: right, 4: down_right, 5: down, 6: down_left, 7: left, 8: up_left
NUM_ACTION = 9

def train(track, width, height):
    sess = tf.Session()

    game = Game(track, width, height, show_game=False)
    brain = DQN(sess, width, height, NUM_ACTION)

    rewards = tf.placeholder(tf.float32, [None])
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('logs', sess.graph)
    summary_merged = tf.summary.merge_all()

    brain.update_target_network()

    epsilon = 1.0
    time_step = 0
    total_reward_list = []

    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        state = game.reset()
        brain.init_state(state)

        if episode > OBSERVE:
            epsilon = 0.25

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

        print('Games: %d Score: %d' % (episode + 1, total_reward))

        total_reward_list.append(total_reward)

        if episode % 10 == 0:
            summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
            writer.add_summary(summary, time_step)
            total_reward_list = []

        if episode > OBSERVE and episode % 10000 == 0:
            saver.save(sess, 'model/dqn.ckpt', global_step=time_step)


def replay(track, width, height):
    sess = tf.Session()

    game = Game(track, width, height, show_game=True)
    brain = DQN(sess, width, height, NUM_ACTION)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        state = game.reset()
        brain.init_state(state)

        while not terminal:
            action = brain.get_action()

            state, reward, terminal = game.step(action)
            total_reward += reward

            brain.remember(state, action, reward, terminal)

            time.sleep(0.1)

        print('Games: %d Score: %d' % (episode + 1, total_reward))


def main(_):
    with open(FLAGS.track) as f:
        data = f.readlines()
        width = int(data[0])
        height = int(data[1])
        track = data[2:]

        if FLAGS.train:
            train(track, width, height)
        else:
            replay(track, width, height)


if __name__ == '__main__':
    tf.app.run()
