### environment ###


LAYOUT_SIZE = [15, 15, 1]
NUM_LAYOUT_TRAIN = 24
NUM_LAYOUT_TEST = 24
OBS_SIZE = [[1, 3], [1, 3]]
NUM_PIN = [5, 5]
NUM_NET = 1
NUM_OBS = 5
NUM_PROCESSES = 1

PRINT_MCTS = False
PRINT_ENV_STEP = True
PRINT_ENV = True

### MCTS ###
C_PUCT = 1
KEEP_CHILD = True
NUM_SIMULATION = 100
DIRICHLET = False
DIRICHLET_ALPHA = 0.3

## MEMORY ###
CAPACITY = 25000

### TRAIN ###
FILE_NAME = 'MCTS_15x15'
TRAIN = False
TRAINING_LOOP = 200
BATCH_SIZE = 512


class color:
   PURPLE = '\033[1;35;48m'
   CYAN = '\033[1;36;48m'
   BOLD = '\033[1;37;48m'
   BLUE = '\033[1;34;48m'
   GREEN = '\033[1;32;48m'
   YELLOW = '\033[1;33;48m'
   RED = '\033[1;31;48m'
   BLACK = '\033[1;30;48m'
   UNDERLINE = '\033[4;37;48m'
   END = '\033[1;37;0m'

if __name__ == '__main__':
   import torch as th
   device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
   print('device', device)
   a = th.Tensor(5,3)
   a = a.cuda()







