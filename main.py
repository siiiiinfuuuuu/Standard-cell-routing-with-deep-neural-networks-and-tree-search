# import multiprocessing as mp
from numpy.core.fromnumeric import size
from mcts import*
from layout import*
from config import*
from env import*
from memory import*
from network import*
from copy import deepcopy
import torch.multiprocessing as mp
import torch.nn as nn
import time
import numpy as np
import random


def sample_data(env: RoutingEnv, model, device, test: bool = False):
    th.cuda.set_device(device)
    mcts = mctsZero(env, model, test, device)
    state, pi, value = mcts.run()
    return state, pi, value

def time_test(num: int):
    
    layouts_train = [Layout(LAYOUT_SIZE) for _ in range(NUM_LAYOUT_TRAIN)]
    env = RoutingEnv(layouts_train)
    device = ['cuda:0', 'cuda:1']
    model = ActorCriticNetwork(num_input_channels = LAYOUT_SIZE[2] * (2 * NUM_NET + 2))
    model = [deepcopy(model).cuda('cuda:0'), model.cuda('cuda:1')]
    # model =  nn.DataParallel(model).share_memory()
    # model =  nn.DataParallel(model).cuda().share_memory()
    # print("test = ", next(model.parameters()).is_cuda)
    # sample_data(env, model)
    result_objs = []
    pool = mp.Pool(num)
    memory = Memory()

    for i in range(num):
        result = pool.apply_async(sample_data, (copy.deepcopy(env), model[i%2], device[i%2]))
        result_objs.append(result)
        env.reset()
    
    for result in result_objs:
        # print(result.get())
        state, pi, value = result.get()
        for s, p in zip(state, pi):
            memory.push(s, p, value)
    # start  = time.time()
    # end = time.time()
    # print("num_pool = ",num," time = ", end - start)
    # end = time.time()
    # print("num_pool = ",num," time = ", end - start)

def ParallelDataCollection(env: RoutingEnv, model, memory: Memory, num_case: int, num_processes: int) -> float:
    device = ['cuda:0', 'cuda:1']
    model = [model, deepcopy(model).cuda('cuda:1')]
    # model = [model]
    c_idx = 0
    pool = mp.Pool(num_processes)
    total_value = 0
    while c_idx < num_case:
        p_idx = 0
        result_objs = []
        while p_idx < num_processes and c_idx < num_case:
            # print(c_idx, p_idx)
            result = pool.apply_async(sample_data, (copy.deepcopy(env), model[p_idx%2], device[p_idx%2]))
            # result = pool.apply_async(sample_data, (copy.deepcopy(env), model.cpu(), 'cpu'))
            result_objs.append(result)
            env.reset()
            p_idx += 1
            c_idx += 1
        for result in result_objs:
            # print(result.get())
            state, pi, value = result.get()
            total_value += value
            # print(c_idx, value)
            for s, p in zip(state, pi):
                memory.push(s, p, value)
        pool = mp.Pool(num_processes)
        # print("sleeping..")
        # time.sleep(20)
        # print("end")
    print("avg_value = ", total_value / num_case)
    return total_value / num_case

def SoftCrossEntropyLoss(input, target):
    logprobs = th.nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]

def Optimization(model, memory: Memory, num_epoch: int, device):
    optimizer = th.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    policy_loss = SoftCrossEntropyLoss
    value_loss = nn.MSELoss()
    for i in range(num_epoch):
        State, Pi, Value = memory.random_sample(BATCH_SIZE)
        State, Pi, Value = th.from_numpy(State).type(th.float).to(device), th.from_numpy(Pi).type(th.float).to(device),th.from_numpy(Value).type(th.float).to(device)
        State, Pi, Value = th.tensor(State, dtype=th.float).to(device), th.tensor(Pi, dtype=th.float).to(device), th.tensor(Value, dtype=th.float).to(device)
        # print(State.size(), Pi.size(), Value.size())
        pred_pi, _, pred_value = model(State)
        # print(pred_pi.type(), pred_value.type())
        # loss = policy_loss(pred_pi, Pi) + 0.1*value_loss(Value, pred_value)
        # print(Value, pred_value)
        # loss = value_loss(Value, pred_value)
        loss = policy_loss(pred_pi, Pi)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def ParallelEvaluate(model, layouts, num_processes: int) -> float:
    num_case = len(layouts)
    env = RoutingEnv(layouts)
    device = ['cuda:0', 'cuda:1']
    model = [model, deepcopy(model).cuda('cuda:1')]
    c_idx = 0
    pool = mp.Pool(num_processes)
    total_value = 0
    memory = Memory()
    while c_idx < num_case:
        p_idx = 0
        result_objs = []
        while p_idx < num_processes and c_idx < num_case:
            result = pool.apply_async(sample_data, (copy.deepcopy(env), model[p_idx%2], device[p_idx%2]))
            result_objs.append(result)
            env.reset()
            p_idx += 1
            c_idx += 1
        for result in result_objs:
            state, pi, value = result.get()
            # print(value)
            total_value += value
            for s, p in zip(state, pi):
                memory.push(s, p, value)
        pool = mp.Pool(num_processes)
    
    # calculate loss
    with th.no_grad():
        device = 'cuda:0'
        value_loss = nn.MSELoss()
        State, Pi, Value = memory.random_sample(len(memory))
        State, Pi, Value = th.from_numpy(State).type(th.float).to(device), th.from_numpy(Pi).type(th.float).to(device),th.from_numpy(Value).type(th.float).to(device)
        State, Pi, Value = th.tensor(State, dtype=th.float).to(device), th.tensor(Pi, dtype=th.float).to(device), th.tensor(Value, dtype=th.float).to(device)
        pred_pi, _, pred_value = model[0](State)
        policy_loss = SoftCrossEntropyLoss
        loss = policy_loss(pred_pi, Pi) + 0.1 * value_loss(Value, pred_value)
        # print("policy_loss = ", policy_loss(pred_pi, Pi).item(), "value_loss = ",  0.1 * value_loss(Value, pred_value))
        print("policy_loss = ", policy_loss(pred_pi, Pi).item())
    print("avg_value = ", total_value / num_case)
    return total_value / num_case


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn') 
    except RuntimeError:
        pass

    layouts_train = [Layout(LAYOUT_SIZE) for _ in range(NUM_LAYOUT_TRAIN)]
    layouts_test = [Layout(LAYOUT_SIZE) for _ in range(NUM_LAYOUT_TEST)]
    env = RoutingEnv(layouts_train)
    
    if TRAIN:
        model = ActorCriticNetwork(num_input_channels = LAYOUT_SIZE[2] * (2 * NUM_NET + 2)).cuda('cuda:0')
        memory = Memory()
        max_value = -100
        for i in range(1):
            print("EPOCHS = ",i)
            # print("DataCollection...")
            avg_value = ParallelDataCollection(env, model, memory, num_case = 24, num_processes = NUM_PROCESSES)
            if avg_value > max_value:
                max_value = avg_value
                th.save(model, FILE_NAME)
                print("save model")
            # print("Memory len = ", len(memory))
            # print("Optimization...")
            Optimization(model, memory, TRAINING_LOOP, 'cuda:0')
            # print("Evaluate...")
            # if i%5==0:
            #     value = ParallelEvaluate(model, layouts_train, num_processes = NUM_PROCESSES)
            #     # value = ParallelEvaluate(model, random.sample(layouts_train, 24), num_processes = NUM_PROCESSES)
            #     if value > max_value:
            #         max_value = value
            #         th.save(model, FILE_NAME)
            #         print("save model, avg_wire = ",max_value)
    else:                
        # test
        model = th.load(FILE_NAME).cuda('cuda:0')
        ParallelEvaluate(model, layouts_train[8:24], num_processes = NUM_PROCESSES)
        # ParallelEvaluate(model, random.sample(layouts_train, 24), num_processes = NUM_PROCESSES)


