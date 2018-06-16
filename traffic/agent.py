import tensorflow as tf
import numpy as np
import random
import time
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow
import os

from game import Game
from model import DQN


tf.app.flags.DEFINE_boolean("cont", False, "Continuous Training Mode")
tf.app.flags.DEFINE_boolean("test", False, "Testing Mode")
tf.app.flags.DEFINE_boolean("train", False, "학습모드. 게임을 화면에 보여주지 않습니다.")
FLAGS = tf.app.flags.FLAGS

# 최대 학습 횟수
MAX_EPISODE = 100000000
# 1000번의 학습마다 한 번씩 타겟 네트웍을 업데이트합니다.
TARGET_UPDATE_INTERVAL = 1000
# 4 프레임마다 한 번씩 학습합니다.
TRAIN_INTERVAL = 4
# 학습 데이터를 어느정도 쌓은 후, 일정 시간 이후에 학습을 시작하도록 합니다.
OBSERVE = 250

# 0: nop, 1: up, 2: right, 3: down, 4: left
NUM_ACTION = 5
SCREEN_WIDTH = 5
SCREEN_HEIGHT = 5
OBS_NUM = 3
BUN_NUM = 3
#SCREEN_WIDTH = 10
#SCREEN_HEIGHT = 10
#OBS_NUM = 10
#BUN_NUM = 10
CHANNEL = 4

def train(cont):
    sess = tf.Session()

    game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, OBS_NUM, BUN_NUM, show_game=False)
    brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, CHANNEL, NUM_ACTION)

    rewards = tf.placeholder(tf.float32, [None])
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    saver = tf.train.Saver()
    if cont:
        sess.run(tf.global_variables_initializer())

        ckpt = str(tf.train.get_checkpoint_state('model')) 
        i = ckpt.find("\"") + 1
        j = ckpt.find("\"", i)
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt[i:j])
        var_to_shape_map = reader.get_variable_to_shape_map()
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for key in var_to_shape_map:
            if "conv2d" in key and "Adam" not in key:
                for key_f in target_vars:
                    if key in key_f.name:
                        sess.run(key_f.assign(reader.get_tensor(key)))
                        break

#        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('logs', sess.graph)
    summary_merged = tf.summary.merge_all()

    # 타겟 네트웍을 초기화합니다.
    brain.update_target_network()

    # 다음에 취할 액션을 DQN 을 이용해 결정할 시기를 결정합니다.
    epsilon = 1.0
    # 프레임 횟수
    time_step = 0
    total_reward_list = []

    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        state = game.reset()
        brain.init_state(state)

        if episode > OBSERVE:
            epsilon = 0.01

        while not terminal:
            if np.random.rand() < epsilon:
                action = random.randrange(NUM_ACTION)
            else:
                action = brain.get_action()
            epsilon += 0.00001

            state, reward, terminal = game.step(action)
            total_reward += reward

            brain.remember(state, action, reward, terminal)

            if time_step > OBSERVE and time_step % TRAIN_INTERVAL == 0:
                brain.train()

            if time_step % TARGET_UPDATE_INTERVAL == 0:
                brain.update_target_network()

            time_step += 1

        if episode % 10 == 0:
            print('Games: %d Score: %d' % (episode + 1, total_reward))

        total_reward_list.append(total_reward)

        if episode % 10 == 0:
            summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
            writer.add_summary(summary, time_step)
            total_reward_list = []

        if episode % 10000 == 0:
            saver.save(sess, 'model/dqn.ckpt', global_step=episode)


def replay():
    sess = tf.Session()

    game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, OBS_NUM, BUN_NUM, show_game=True)
    brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, CHANNEL, NUM_ACTION)

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

            time.sleep(0.3)

        print('Games: %d Score: %d' % (episode + 1, total_reward))

def test():
    sess = tf.Session()

    game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, OBS_NUM, BUN_NUM, show_game=False)
    brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, CHANNEL, NUM_ACTION)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    total_succ = 0
    for episode in range(10000):
        terminal = False
        total_reward = 0

        state = game.reset()
        brain.init_state(state)

        step = 0
        while not terminal and step <= 200:
            action = brain.get_action()
            state, reward, terminal, succ = game.step(action)
            if terminal and succ:
                total_succ += 1
            step += 1

    print(total_succ)

def main(_):
    if FLAGS.test:
        test()
    if FLAGS.cont:
        train(True)
    elif FLAGS.train:
        train(False)
    else:
        replay()


if __name__ == '__main__':
    tf.app.run()
