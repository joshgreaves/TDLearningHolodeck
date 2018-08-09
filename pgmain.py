import holodeck
from holodeck.sensors import Sensors
from holodeck.holodeck import GL_VERSION
from tqdm import tqdm
import numpy as np
import math
import random
import torch
import copy

ACTION_DIM = 4

HIDDEN1 = 20
HIDDEN2 = 20

FORWARD = np.array([1., 0, 0])
STEP_SIZE = 0.8
GAMMA = 0.99
EPSILON = 0.2
THRESHOLD_DIST = 20


# Receives state data, outputs an action
class ActorNetwork(torch.nn.Module):
    def __init__(self, conv1, conv2, conv3):
        # Expects images of (256 x 256)
        super(ActorNetwork, self).__init__()

        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3

        self.img_fc = torch.nn.Linear(128 * 128 * 64, 100)
        self.loc_fc = torch.nn.Linear(3, 10)
        self.orientation_fc = torch.nn.Linear(9, 20)

        self.fc1 = torch.nn.Linear(130, 130)
        self.fc2 = torch.nn.Linear(130, ACTION_DIM)

    def forward(self, x):
        img, loc, orientation = x
        img = torch.from_numpy(img)
        loc = torch.from_numpy(loc)
        orientation = torch.from_numpy(orientation)

        h1 = self.conv1(img).clamp(min=0)
        h2 = self.conv2(h1).clamp(min=0)
        h3 = self.conv3(h2).clamp(min=0)
        flattened = h3.view(-1, 128 * 128 * 64)

        img_h = self.img_fc(flattened).clamp(min=0)
        loc_h = self.loc_fc(loc).clamp(min=0)
        orientation_h = self.orientation_fc(orientation).clamp(min=0)

        concatenated = torch.cat((img_h, loc_h, orientation_h), 1)
        result_h = self.fc1(concatenated).clamp(min=0)
        result = self.fc2(result_h)
        return result


# Receives a (state, action), estimates the value of that (state, action)
class CriticNetwork(torch.nn.Module):
    def __init__(self, conv1, conv2, conv3):
        super(CriticNetwork, self).__init__()

        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3

        self.img_fc = torch.nn.Linear(128 * 128 * 64, 100)
        self.loc_fc = torch.nn.Linear(3, 10)
        self.orientation_fc = torch.nn.Linear(9, 20)
        self.action_fc = torch.nn.Linear(ACTION_DIM, 20)

        self.fc1 = torch.nn.Linear(150, 150)
        self.fc2 = torch.nn.Linear(150, 1)

    def forward(self, x):
        img, loc, orientation, action = x
        img = torch.from_numpy(img)
        loc = torch.from_numpy(loc)
        orientation = torch.from_numpy(orientation)

        h1 = self.conv1(img).clamp(min=0)
        h2 = self.conv2(h1).clamp(min=0)
        h3 = self.conv3(h2).clamp(min=0)
        flattened = h3.view(-1, 128 * 128 * 64)

        img_h = self.img_fc(flattened).clamp(min=0)
        loc_h = self.loc_fc(loc).clamp(min=0)
        orientation_h = self.orientation_fc(orientation).clamp(min=0)
        action_h = self.action_fc(action).clamp(min=0)

        concatenated = torch.cat((img_h, loc_h, orientation_h, action_h), 1)
        result_h = self.fc1(concatenated).clamp(min=0)
        result = self.fc2(result_h)
        return result


def get_state(state):
    camera = state[Sensors.PIXEL_CAMERA][:, :, :-1]
    camera = np.swapaxes(camera, 2, 0)
    camera = np.swapaxes(camera, 2, 1)
    camera = camera.reshape([1, 3, 256, 256]).astype(np.float32) / 255.0  # Get without alpha
    return camera, state[Sensors.LOCATION_SENSOR].reshape([1, 3]), state[Sensors.ORIENTATION_SENSOR].reshape([1, 9])


def main():
    env = holodeck.make("UrbanCity", resolution=(500, 800))

    conv1_in_channels = 3
    conv1_out_channels = 32
    conv1_size = 3
    conv1_stride = 1
    conv1_padding = 1

    conv2_in_channels = 32
    conv2_out_channels = 64
    conv2_size = 4
    conv2_stride = 2
    conv2_padding = 1

    conv3_in_channels = 64
    conv3_out_channels = 64
    conv3_size = 3
    conv3_stride = 1
    conv3_padding = 1

    conv1 = torch.nn.Conv2d(
        conv1_in_channels,
        conv1_out_channels,
        conv1_size,
        stride=conv1_stride,
        padding=conv1_padding
    )
    conv2 = torch.nn.Conv2d(
        conv2_in_channels,
        conv2_out_channels,
        conv2_size,
        stride=conv2_stride,
        padding=conv2_padding
    )
    conv3 = torch.nn.Conv2d(
        conv3_in_channels,
        conv3_out_channels,
        conv3_size,
        stride=conv3_stride,
        padding=conv3_padding
    )

    actor = ActorNetwork(conv1, conv2, conv3)
    critic = CriticNetwork(conv1, conv2, conv3)
    criterion = torch.nn.L1Loss(reduction='sum')
    critic_optimizer = torch.optim.SGD(critic.parameters(), lr=0.01)
    actor_optimizer = torch.optim.SGD(actor.parameters(), lr=0.01)

    distance_increment = 5

    for episode in range(100):
        target_critic = copy.deepcopy(critic)
        for param in target_critic.parameters():
            param.require_grad = False
        s, r, _, _ = env.reset()

        distance_to_travel = distance_increment

        state = get_state(s)
        sarsa = [None, None, r, state, None]

        for _ in tqdm(range(300)):
            action = actor(state)
            sarsa[4] = action

            if all(map(lambda x: x is not None, sarsa)):
                predicted_q = critic((sarsa[0][0], sarsa[0][1], sarsa[0][2], sarsa[1]))
                predicted_q_prime = target_critic((sarsa[3][0], sarsa[3][1], sarsa[3][2], sarsa[4]))
                loss = criterion(sarsa[2] + GAMMA * predicted_q_prime, predicted_q)
                critic_optimizer.zero_grad()
                loss.backward()
                critic_optimizer.step()

            s, r, terminal, _ = env.step(action.detach().numpy())

            # Calculate the true reward
            if s[Sensors.LOCATION_SENSOR][0] > distance_to_travel:
                r = 1
                distance_to_travel += distance_increment
            else:
                r = -1

            # Q(s, a) = r + gamma * Q(s', a')
            sarsa[0] = sarsa[3]
            sarsa[1] = sarsa[4].detach()
            sarsa[2] = r
            sarsa[3] = get_state(s)

        s, _, _, _ = env.reset()
        state = get_state(s)
        for _ in range(300):
            action = actor(state)
            predicted_q = critic((state[0], state[1], state[2], action))
            loss = -predicted_q
            critic_optimizer.zero_grad()
            actor_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()
            s, _, _, _ = env.step(action.numpy())
            state = get_state(s)


if __name__=="__main__":
    main()
