import gym
import retro
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import random
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from pythonosc import dispatcher
from pythonosc import osc_server

from time import sleep
from server import controlArray




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



X_train = np.array([
#   [x, y]
# Jump/crouch
    [.5, 0],
    [.5, .2],
    [.5, .4],
    [.5, .5],
    [.5, .6],
    [.5, .8],
    [.5, 1],

# Left/Right
    [0, .5],
    [.2, .5],
    [.4, .5],
    [.5, .5],
    [.6, .5],
    [.8, .5],
    [1, .5],

# Corners of the box
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
])

y_train = np.array([
#   [?,?,?,?,?,?,l,r,j]
# Jump/crouch
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,1],

# Left/Right
    [0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,1,0],

# Corners
    [0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,0,1,1]
])




class Controls:
    def __init__(self):
        self.env = retro.make('SuperMarioBros-Nes')
        self.tree = DecisionTreeClassifier()
        self.tree.fit(X_train, y_train)
        init_screen = self.get_screen()
        _, _, screen_height, screen_width = init_screen.shape
        self.policy_net = DQN(screen_height, screen_width).to(device)
        self.target_net = DQN(screen_height, screen_width).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(100000)

    def select_action(self,state):
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            ind = policy_net(state).max(1)[1].view(-1, 1)
            action = torch.zeros(9, dtype=torch.long)
            action[ind] = 1
            return action

    def get_user_input(self,x=None,y=None):
        # Get the x/y from microbit

        try:
            x, y = controlArray.remove(0)

        except:
            x,y = .5,.5


        return x, y

    def get_control_from_user_input(self, x, y):
        input = np.array([ x, y ])
        input = input.reshape(1,-1)
        control = self.tree.predict(input)
        return control

    def optimize_model(self,BATCH_SIZE=128,GAMMA=0.999):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
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
        state_action_values = self.policy_net(state_batch).reshape(-1,)[action_batch==1]

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        expected_state_action_values = expected_state_action_values[state_action_values.long()]
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1));
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.3):]
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0).to(device)

    def go(self):
        self.env.reset()
        last_screen = self.get_screen()
        current_screen = self.get_screen()
        state = current_screen - last_screen



        for t in count():
            # Select and perform an action

            self.env.render()

            action = self.get_control_from_user_input(*self.get_user_input())

            if False:
                action = self.select_action(state)
            _, reward, done, _ = self.env.step(np.array(action[0]))
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = self.get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            self.memory.push(state, torch.tensor(action[0]), next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            if t%250 == 0:
                self.optimize_model()
            if t%500 == 0:
                torch.save(self.policy_net.state_dict(), "policy_net.pt")
                torch.save(self.target_net.state_dict(), "target_net.pt")
                self.target_net.load_state_dict(self.policy_net.state_dict())


            if done:
                env.reset()


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

class DQN(nn.Module):

    def __init__(self, h, w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, 9) # 448 or 512

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


c = Controls()
c.go()
