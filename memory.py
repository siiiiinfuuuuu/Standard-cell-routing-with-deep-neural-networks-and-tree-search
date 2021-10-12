from collections import deque, namedtuple
from config import CAPACITY
import random
import numpy as np

Data = namedtuple('Data',('state', 'pi', 'value'))

class Memory:
    def __init__(self) -> None:
        self.memory = deque([],maxlen = CAPACITY)

    def push(self, *args):
        self.memory.append(Data(*args))

    def random_sample(self, batch_size):
        if(len(self.memory) < batch_size):
            batch_size = len(self.memory)
        mini_batch = random.sample(self.memory, batch_size)
        return np.array([x.state for x in mini_batch]), np.array([x.pi for x in mini_batch]), np.array([x.value for x in mini_batch]).reshape((-1, 1))

    def __len__(self):
        return len(self.memory)
