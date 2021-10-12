from network import ActorCriticNetwork
from env import *
from layout import *
from mcts import mctsZero
import torch.multiprocessing as mp
import torch as th

def sample_data(env: RoutingEnv, model, device):
    th.cuda.set_device(device)
    mcts = mctsZero(env, model, False, device)
    state, pi, value = mcts.run()
    return state, pi, value

if __name__ == '__main__':
    
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    layouts_train = [Layout(LAYOUT_SIZE) for _ in range(NUM_LAYOUT_TRAIN)]
    env = RoutingEnv(layouts_train)
    model= ActorCriticNetwork(num_input_channels=4).cuda('cuda:1')
    num_processes = 4
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=sample_data, args=(copy.deepcopy(env), model, 'cuda:1'))
        env.reset()
        p.start()
        processes.append(p)
    for p in processes:
        p.join()