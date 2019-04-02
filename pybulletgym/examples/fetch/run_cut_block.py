import math
import pybullet as p
import random
from itertools import count

import pybulletgym.envs
import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image

import utils
# Load the OpenAI gym env
from MoveAgent import DQN, ReplayMemory, Transition

env = gym.make('FetchCutBlockEnv-v1')  # type: gym.Env

# Render the display and perform reset operations that set up the state
env.render(mode="human")
env.reset()

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
episode_durations = []


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # state_action_values = policy_net(state_batch).gather(25, action_batch)
    state_action_values = policy_net(state_batch)#action_batch.double()  # policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros((BATCH_SIZE, 25), device=device)
    # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    next_state_values[non_final_mask] = target_net(non_final_next_states)
    # Compute the expected Q values
    # expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.unsqueeze(1).expand(128, 25)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.unsqueeze(1), expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.shape[0]


policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # return policy_net(state).max(1)[1].view(1, 1)
            return torch.round(policy_net(state)).long()
    else:
        # return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        return torch.tensor([[random.randrange(-1, 2) for i in range(n_actions)]], device=device, dtype=torch.long)

# Find the robot's base
baseId = -1
for i in range(p.getNumBodies()):
    print(p.getBodyInfo(i))
    if p.getBodyInfo(i)[0].decode() == "base_link":
        baseId = i
        print("found base")


plotter = utils.Plotter()

num_episodes = 150
for i_episode in range(num_episodes):
    print(f'Starting episode {i_episode}')
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        fetchPos, fetchOrn = p.getBasePositionAndOrientation(baseId)
        distance = 2
        yaw = 90
        p.resetDebugVisualizerCamera(distance, yaw, -45, fetchPos)
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.numpy()[0])
        plotter.live_plotter(reward)

        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()