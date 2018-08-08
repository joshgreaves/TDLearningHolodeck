import holodeck
from holodeck.sensors import Sensors
from holodeck.holodeck import GL_VERSION
from tqdm import tqdm
import numpy as np
import math
import random

ACTIONS = [0, 2, 3]
FORWARD = np.array([1., 0, 0])
STEP_SIZE = 0.8
GAMMA = 0.99
EPSILON = 0.2
THRESHOLD_DIST = 20


def get_angle(forward_vector):
    cos_theta = np.dot(FORWARD, forward_vector)
    cos_theta = max(float(cos_theta), -1.0)
    cos_theta = min(float(cos_theta), 1.0)
    theta = math.acos(cos_theta)
    if forward_vector[1] > 0:
        return round(theta, 1)
    else:
        return round(2 * math.pi - theta, 1)


def simplify_state(state):
    location = state[Sensors.LOCATION_SENSOR] * 10  # Each unit is 10 cm
    orientation = state[Sensors.ORIENTATION_SENSOR]
    return int(location[0]), int(location[1]), get_angle(orientation[0, :])


def softmax(w):
    exps = np.exp(w * 10)
    return exps / np.sum(exps)


def get_action_probabilities(state, action_values):
    weights = np.zeros(len(ACTIONS), dtype=np.float32)
    for i, action in enumerate(ACTIONS):
        key = (state, action)
        if key not in action_values:
            action_values[key] = 2.0
        weights[i] = action_values[key]
    return softmax(weights)


def choose_action(state, action_values):
    probs = get_action_probabilities(state, action_values)
    return np.random.choice(ACTIONS, p=probs)


def get_expected_action_value(simple_state, action_values):
    probs = get_action_probabilities(simple_state, action_values)
    vals = np.zeros(len(ACTIONS), dtype=np.float32)
    for i, action in enumerate(ACTIONS):
        vals[i] = action_values[(simple_state, action)]
    return float(np.sum(probs * vals))


# def get_max_action_value(state, action_values):
#     weights = np.zeros(len(ACTIONS), dtype=np.float32)
#     for i, action in enumerate(ACTIONS):
#         key = (state, action)
#         if key not in action_values:
#             action_values[key] = 0
#         weights[i] = action_values[key]
#     return action_values[(state, np.argmax(weights))]
#

def main():
    env = holodeck.make("MazeWorld")

    action_values = dict()

    for episode in range(100):
        # print("Episode", episode)
        state, reward, terminal, _ = env.reset()
        prev_state = simplify_state(state)
        avg_expected = 0.0
        total_reward = 0
        next_threshold = THRESHOLD_DIST

        for _ in tqdm(range(600)):
        # for _ in range(600):
            action = choose_action(prev_state, action_values)
            state, reward, terminal, _ = env.step(action)
            simple_state = simplify_state(state)

            reward -= 1
            if simple_state[0] > next_threshold:
                next_threshold += THRESHOLD_DIST
                reward += 1
            total_reward += reward

            # Update action values
            key = (prev_state, action)
            if key not in action_values:
                action_values[key] = 10.0
            expected_value = get_expected_action_value(simple_state, action_values)
            avg_expected += expected_value
            action_values[key] = ((action_values[key] * (1 - STEP_SIZE)) +
                                  STEP_SIZE * (reward + GAMMA * (expected_value - action_values[key])))

            prev_state = simple_state

        print("Episode", episode, "\tavg expected:", avg_expected / 600, "\tTotal reward:", total_reward)
    # print(all_states)


if __name__=="__main__":
    main()
