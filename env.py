from typing import Set, Union
import numpy as np
import gym
from gym import spaces
import random
import copy
from stable_baselines3 import *
from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
import os
import time
from layout import *
from config import *
import queue

# class VecExtractDictObs(VecEnvWrapper):
#     """
#     A vectorized wrapper for filtering a specific key from dictionary observations.
#     Similar to Gym's FilterObservation wrapper:
#         https://github.com/openai/gym/blob/master/gym/wrappers/filter_observation.py

#     :param venv: The vectorized environment
#     :param key: The key of the dictionary observation
#     """

#     def __init__(self, venv: VecEnv, key: str):
#         self.key = key
#         super().__init__(venv=venv, observation_space=venv.observation_space.spaces[self.key])

#     def reset(self) -> np.ndarray:
#         obs = self.venv.reset()
#         return obs[self.key]

#     def step_async(self, actions: np.ndarray) -> None:
#         self.venv.step_async(actions)

#     def step_wait(self) -> VecEnvStepReturn:
#         obs, reward, done, info = self.venv.step_wait()
#         return obs[self.key], reward, done, info

class RoutingEnv(gym.Env):
    def __init__(self, layouts: List[Layout]):
        super(RoutingEnv, self).__init__()
        #init layout
        self.layouts = layouts
        self.layout_idx = -1
        self.reset()
        #init env
        self.action_space = spaces.Discrete(self.state.size[0] * self.state.size[1] * self.state.size[2])
        # self.observation_space = spaces.Box(low=0, high=255, shape=(self.state.size[0], self.state.size[1], self.state.size[2] * ( 1)), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.state.size[0], self.state.size[1], self.state.size[2] * ( 2 * len(self.state.nets) + 2)), dtype=np.uint8)

    def __deepcopy__(self, memodict={}):
        copy_object = RoutingEnv(self.layouts)
        copy_object.layout_idx = self.layout_idx
        copy_object.state = copy.deepcopy(self.state)
        copy_object.iter = self.iter
        copy_object.num_step =  self.num_step
        copy_object.done = self.done
        return copy_object

    def step(self, action: Union[int, Point]) -> Union[np.ndarray, float, bool, dict]:
        # self.render()
        if type(action) != Point:
            action = Int2Point(action)
        self.iter += 1
        # print("action  = ", action)
        # print("action point = ", action)
        legal_actions = self.state.legalActions()
        if len(legal_actions) == 0 or self.iter >= 500:
            # self.render()
            self.done = True
            return self.state.envState(), 0, True, {}
        elif action not in legal_actions:
            # print(action," is iligal actions")
            # self.state.renderLigalActions()
            self.done = True
            return self.state.envState(), -500 + self.num_step, True, {}
        else:
            self.num_step += 1
            next_state, done = self.state.step(action)
            # self.render()
            if done:
                # self.render()
                self.done = True
                return next_state.envState(), -1, True, {}
            return next_state.envState(), -1, False, {}  
    
    def reset(self) -> np.ndarray:
        self.layout_idx = ( self.layout_idx + 1 ) % len(self.layouts)
        self.state = State(self.layouts[self.layout_idx])
        self.iter = 0
        self.num_step = 0
        self.state.reset()
        self.done = False
        return self.state.envState()
    
    def render(self) -> None:
        self.state.render()
        # self.state.renderLigalActions()
        # self.state.renderEmptyState()

def Point2Int(action: Point) -> int:
    return action.z * LAYOUT_SIZE[0] * LAYOUT_SIZE[1] + action.y * LAYOUT_SIZE[0] + action.x

def Int2Point(action: int) -> Point:
    z = int(action / (LAYOUT_SIZE[0] * LAYOUT_SIZE[1]))
    y = int((action % (LAYOUT_SIZE[0] * LAYOUT_SIZE[1])) / LAYOUT_SIZE[0])
    x = action % (LAYOUT_SIZE[0] * LAYOUT_SIZE[1]) % LAYOUT_SIZE[0]
    return Point(x , y, z)

class EvalEnv(RoutingEnv):
    def step(self, action) -> Union[np.ndarray, float, bool, dict]:
        self.iter += 1
        z = int(action / (self.state.size[0] * self.state.size[1]))
        y = int((action % (self.state.size[0] * self.state.size[1])) / self.state.size[0])
        x = action % (self.state.size[0] * self.state.size[1]) % self.state.size[0]
        action = Point(x , y, z)
        # print("action = ", action)
        legal_actions = self.state.legalActions()
        if len(legal_actions)== 0 or self.iter >= 500:
            # self.render()
            return self.state.envState(), -100, True, {}
        elif action not in legal_actions:
            return self.state.envState(), 0, False, {}
        else:
            next_state, done = self.state.step(action)
            # self.render()
            if done:
                # self.render()
                return next_state.envState(), -1, done, {}
            return next_state.envState(), -1, done, {}
    
    def reset(self) -> np.ndarray:
        self.layout_idx = ( self.layout_idx + 1 ) % NUM_LAYOUT_TEST
        self.state = State(self.layouts[self.layout_idx])
        self.iter = 0
        self.state.reset()
        return self.state.envState()


class State:
    def __init__(self, layout: Layout):
        self.initLayout = layout
        self.reset()

    def __deepcopy__(self, memodict={}):
        copy_object = State(self.initLayout)
        copy_object.nets = copy.deepcopy(self.nets)
        copy_object.currNetIdx = self.currNetIdx
        copy_object.empty_state = np.copy(self.empty_state)
        return copy_object

    def envState(self) -> np.ndarray:
        # We assume HxW images
        env_state = self.obstacles_state
        env_state = np.concatenate((env_state, self.nets[self.currNetIdx].actions_state), axis = 2)
        for net in self.nets:
            env_state = np.concatenate((env_state, net.pins_state, net.wires_state), axis = 2)
        return np.transpose(env_state * 255, (2, 1, 0)) 

    def step(self, action) -> Union["State" , bool]:
        done = False
        self.nets[self.currNetIdx].addWire(action)
        self.nets[self.currNetIdx].updataActions(action, self.empty_state)
        self.empty_state[action.x][action.y][action.z] = 0
        self.__deleteActions(action)
        if self.nets[self.currNetIdx].isPin(action):
            self.nets[self.currNetIdx].deletePin(action)
        self.currNetIdx = self.__nextNet()
        if self.currNetIdx == -1:
            done = True
        return self, done

    def reset(self) -> None:
        self.size = self.initLayout.size
        self.nets = copy.deepcopy(self.initLayout.nets)
        self.currNetIdx = 0
        self.obstacles = self.initLayout.obstacles
        self.obstacles_state = self.initLayout.obstacles_state
        self.empty_state = np.logical_not(self.obstacles_state)
        #init empty_state
        for net in self.nets:
            self.empty_state = self.empty_state & np.logical_not(net.pins_state)
        #random select start pin
        for net in self.nets:
            start_pin = random.choice(net.pins)
            net.deletePin(start_pin)
            net.addWire(start_pin)
            net.updataActions(start_pin, self.empty_state)
            self.__deleteActions(start_pin)

    def distance_point2wire(self, p, net) -> int:
        if net.isWire(p):
            return 0
        q = queue.Queue()
        q.put(p)
        distance = 1
        visited = np.zeros((self.size[0], self.size[1], self.size[2]), dtype=np.bool_)
        visited[p.x][p.y][p.z] = True
        while not q.empty():
            level_size = q.qsize()
            # print(distance, level_size)
            for _ in range(level_size):
                point = q.get()
                for dir in directions:
                    neighbor = point + dir
                    if net.inLayout(neighbor) and visited[neighbor.x][neighbor.y][neighbor.z] == False:
                        if net.isWire(neighbor):
                            # print(distance)
                            return distance
                        elif self.empty_state[neighbor.x][neighbor.y][neighbor.z] == 1:
                            visited[neighbor.x][neighbor.y][neighbor.z] = True
                            q.put(neighbor)
            distance+=1
        return self.size[0] * self.size[1] * self.size[2]

    def distance_pins2wire(self) -> int:
        dis = 0
        for net in self.nets:
            if net.done == False:
                for pin in net.pins:
                    dis += self.distance_point2wire(pin, net)
        return dis

    def legalActions(self):
        return self.nets[self.currNetIdx].actions

    def render(self) -> None:
        for z in range(self.size[2]):
            # print("     =====Layer ",z,"=====")
            for y in reversed(range(self.size[1])):
                print(color.BLACK + str(y % 10) + color.END, end=" ")
                for x in range(self.size[0]): 
                    is_net = False
                    net_idx = 0
                    net: Net
                    for net in self.nets:
                        if Point(x, y, z) in net.routed_pins:
                            print(color.PURPLE + chr(ord('A') + net_idx) + color.END , end=" ")
                            is_net = True
                        elif net.pins_state[x][y][z] == 1:
                            print(color.GREEN + chr(ord('A') + net_idx) + color.END , end=" ")
                            is_net = True
                        elif net.wires_state[x][y][z] == 1:
                            print(color.BLUE + chr(ord('A') + net_idx) + color.END , end=" ")
                            is_net = True
                        net_idx = net_idx + 1
                    if not is_net:
                        if self.obstacles_state[x][y][z] == 1:
                            print(color.RED + "X" + color.END, end=" ")
                        elif self.nets[self.currNetIdx].actions_state[x][y][z] == 1:
                            print(color.YELLOW + "." + color.END, end=" ")
                        else:
                            print(".", end=" ")
                print("")
            print(" ", end=" ")
            for x in range(self.size[0]):
                print(color.BLACK + str(x % 10) + color.END, end=" ")
            print("")
            print("")
    
    def renderEmptyState(self) -> None:
        for z in range(self.size[2]):
            print("     =====Layer ",z,"=====")
            for y in reversed(range(self.size[1])):
                for x in range(self.size[0]):
                    # print("", end=" ")
                    if self.empty_state[x][y][z]:
                        print(".", end=" ")
                    else:
                        print("X", end=" ")
                print("")
    
    def renderLegalActions(self) -> None:
        print("legal action = ", end="")
        for la in self.legalActions():
            print(la, end=" ")
        print("")
    
    def renderCurr(self) -> None:
        print("current Net = ", chr(ord('A') + self.currNetIdx))
    
    def __nextNet(self) -> int:
        for netIdx in range(self.currNetIdx + 1, len(self.nets)):
            if not self.nets[netIdx].done:
                return netIdx
        for netIdx in range(0, self.currNetIdx + 1):
            if not self.nets[netIdx].done:
                return netIdx
        return -1
    
    def __inLayout(self, p) -> bool:
        return p.x >= 0 and p.x < self.size[0] and p.y >= 0 and p.y < self.size[1] and p.z >=0 and p.z < self.size[2]
    
    def __isEmpty(self, p) -> bool:
        return self.empty_state[p.x][p.y][p.z]
    
    def __deleteActions(self, action) -> None:
        net: Net
        for net in self.nets:
            action: Point
            if action in net.actions:
                net.actions.remove(action)
                net.actions_state[action.x][action.y][action.z] = 0




if __name__ == '__main__':
    layouts = [Layout(LAYOUT_SIZE) for _ in range(NUM_LAYOUT_TRAIN)]
    

    # print("memory test")
    env = RoutingEnv(layouts)
    for i  in range(10):
        env.render()
        # env.state.renderEmptyState()
        # print(env.state.distance_point2wire(Point(7,7,0),env.state.nets[0]))
        env.reset()
    # env_list = []
    # for i in range(100000):
    #     if i % 100 == 0:
    #         print(i)
    #     e = copy.deepcopy(env)
    #     env_list.append(e)
    # a = []
    # s = time.time()
    # for i in range(10):
    #     # a.append(copy.deepcopy(env))
    #     env.render()
    #     env.reset()
    # e = time.time()
    # print("time = %5.2f" %(e-s))
    # print("training")
    # # model = DQN('CnnPolicy', env).learn(total_timesteps=100000*30*3)
    # model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="tensorboard").learn(total_timesteps=100000*30*5)
    # model.save("PPO")

    # del model # remove to demonstrate saving and loading
    # model = DQN.load("DQN")
    # print("testing")
    # obs = env.reset()
    # step = 0
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     if step % 100 == 0:
    #         print("step = ", step)
    #         env.render()
    #     if done:
    #         env.render()
    #         obs = env.reset()
    #         break
    #     step += 1

    # env.render()
    # done = False
    # while not done:
    #     x, y, z = int(input()), int(input()), int(input())
    #     action = z * env.state.size[0] * env.state.size[1] + y * env.state.size[0] + x
    #     _, _, done,_ = env.step(action)
    #     env.render()
    # env.render()
