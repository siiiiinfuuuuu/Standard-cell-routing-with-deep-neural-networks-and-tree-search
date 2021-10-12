from numpy.random import standard_exponential
from config import*
from typing import Callable, Dict
import math
import copy
from env import*
import numpy as np
import random
from agent import*
import time

class Node:
    def __init__(self, parent: "Node", prior: float, depth: int, env: RoutingEnv, Qmax: int = -LAYOUT_SIZE[0]*LAYOUT_SIZE[1]*LAYOUT_SIZE[2]) -> None:
        self.children = dict()
        self.Q_max : int = Qmax # init parent Q
        self.Q_min : int = 0 # init parent Q
        self.parent = parent
        self.env = env
        self.Q : float = 0
        self.N : int = 0
        self.P = prior
        self.W : float = 0
        self.depth = depth

    def addChild(self, action: Point, priors: float, env: RoutingEnv) -> None:
        self.children[action] = Node(self, priors, self.depth + 1, env)
    
    def select(self) -> "Node":
        return max(self.children.items(), key = lambda child: child[1].getValue())[1]

    def getValue(self) -> float:
        u = C_PUCT * self.P * math.sqrt(self.parent.N) / (1 + self.N)
        # u =  C_PUCT * math.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + u
    
    def update(self, leaf_value: float) -> None:
        self.N += 1
        self.W += leaf_value
        self.Q = self.W / self.N
        if self.parent != None:
            if self.Q > self.parent.Q_max:
                self.parent.Q_max = self.Q
            if self.Q < self.parent.Q_min:
                self.parent.Q_min = self.Q
    
    def max_select(self) -> "Node":
        return max(self.children.items(), key = lambda child: child[1].max_getValue())[1]

    def max_update(self, leaf_value: float) -> None:
        self.N += 1
        if leaf_value > self.Q or self.Q == 0:
            self.Q = leaf_value
            if self.parent != None:
                if self.Q > self.parent.Q_max:
                    self.parent.Q_max = self.Q
                if self.Q < self.parent.Q_min:
                    self.parent.Q_min = self.Q
    
    def max_getValue(self) -> float:
        u = self.max_getU()
        qn = self.max_getQn()
        return qn + u
    
    def max_getU(self) -> float:
        return C_PUCT * self.P * math.sqrt(self.parent.N) / (1 + self.N)

    def max_getQn(self) -> float:
        if self.parent != None and self.parent.Q_max!=self.parent.Q_min:
            return (self.Q - self.parent.Q_min) / (self.parent.Q_max - self.parent.Q_min)
        else:
            return 0.5

    def isLeaf(self) ->bool:
        return self.children == {}

    def state(self) -> np.ndarray:
        return self.env.state.envState()
    
    def legalActions(self):
        return self.env.state.legalActions()
    

class mcts:
    def __init__(self, env: RoutingEnv, model) -> None:
        self.root = Node(None, 1.0, 0, env)
        self.env = env
        self.model = model

    def select(self) -> Node:
        current = self.root
        while  not current.isLeaf():
            current = current.select()
        return current

    def expand(self, leaf: Node) -> None:
        # leaf.env.render()
        # distribution, _ = self.model.network(leaf.state())
        # print("distribution = ", distribution)
        # print("legalActions = ", leaf.env.state.renderLegalActions())
        # print("expand")
        for action in leaf.legalActions():
            env = copy.deepcopy(leaf.env)
            env.step(action)
            # leaf.addChild(action, distribution[0][Point2Int(action)].item(), env)
            leaf.addChild(action, 1/len(leaf.legalActions()), env)
            # print("action ", action, " nn_distribution = ", distribution[0][Point2Int(action)])
    
    def evaluate(self, leaf: Node) -> float:
        if leaf.env.done == True:
            print("done")
            return -leaf.depth
        else:
            return - leaf.depth - self.randomRollout(copy.deepcopy(leaf.env))
            # return - leaf.depth - self.networkRollout(copy.deepcopy(leaf.env))

    def backup(self, leaf: Node, leaf_value: float):
        current = leaf
        while current != None:
            current.update(leaf_value)
            current = current.parent

    def randomRollout(self, env: RoutingEnv) -> int:
        num_step = 0
        while True:
            num_step += 1
            action = random.choice(tuple(env.state.legalActions())) 
            _, _, done, _ = env.step(action)
            if done:
                return num_step

    def networkRollout(self, env: RoutingEnv) -> int:
        num_step = 0
        obs = env.state.envState()
        while True:
            num_step += 1
            action, _ = self.model.predict(obs, deterministic=True)
            _, _, done, _ = env.step(action)
            if done:
                return num_step
    
    def step(self, action) -> bool:
        _, _, done, _ = self.env.step(action)
        # self.root = self.root.children[action]
        # self.root.parent = None
        self.root = Node(None, 1.0, self.root.depth + 1, self.env)
        return done

    def simulation(self, num_sim) -> None:
        select, expand, evaluate, backup = 0, 0, 0, 0
        for i in range(num_sim):
            # print(i)
            t0 = time.time()
            leaf = self.select()
            t1 = time.time()
            self.expand(leaf)
            t2 = time.time()
            leaf_value = self.evaluate(leaf)
            t3 = time.time()
            self.backup(leaf, leaf_value)
            t4 = time.time()
            select += t1 - t0
            expand += t2 - t1
            evaluate += t3 - t2
            backup += t4 - t3
        # print("time: select =  %5.2f expand = %5.2f evaluate = %5.2f backup = %5.2f" % (select, expand, evaluate, backup))

    def predict(self) -> dict():
        probabilities = {}
        # print("root ",self.root.N)
        # print("Q = ", -self.root.Q)
        # print(self.root.children.items())
        max_n = max(self.root.children.items(), key = lambda item: item[1].N)[1].N
        sum_exp = 0
        for item in self.root.children.items():
            sum_exp += np.exp(item[1].N - max_n)
        for item in self.root.children.items(): 
            probabilities[item[0]] = np.exp(item[1].N - max_n) / sum_exp
            # print("action = ", item[0]," N = %2d Q = %6.2f P = %5.2f" % (item[1].N, item[1].Q, item[1].P))
        return probabilities
        
    def render(self) -> None:
        self.env.render()

class mctsZero(mcts):
    def __init__(self, env: RoutingEnv, model, test: bool, device: str) -> None:
        self.root = Node(None, 1.0, 0, env)
        self.env = env
        self.test = test
        self.device = device
        self.model = model
    
    def run(self):
        state = []
        pi = []
        value = 0
        while True:
            # print(self.env.state.envState().shape)
            if PRINT_ENV_STEP:
                self.render()
                print(self.env.state.distance_pins2wire())
            # print(np.expand_dims(self.env.state.envState(), axis=0).shape)
            state.append(self.env.state.envState())
            self.simulation(NUM_SIMULATION)
            if not TRAIN:
                mcts_pi = self.predict(tau=0.05)
            elif self.env.iter < 10:
                mcts_pi = self.predict(tau=1)
            else:
                mcts_pi = self.predict(tau=0.3)
            pi.append(self.pi2NNpi(mcts_pi))
            action = np.random.choice([*mcts_pi.keys()], p = [*mcts_pi.values()])
            done = self.step(action)
            value -= 1
            if done:
                if PRINT_ENV:
                    self.render()
                return state, pi, value
    
    def pi2NNpi(self, pi:dict) -> np.ndarray:
        NNpi = np.zeros(self.env.action_space.n, dtype=np.float16)
        for key, value in pi.items():
            NNpi[Point2Int(key)] = value
        return NNpi

    def select(self) -> Node:
        current = self.root
        while  not current.isLeaf():
            current = current.max_select()
        return current
    
    def expandAndEvaluate(self, leaf: Node) -> float:
        with th.no_grad():
            state = th.as_tensor(np.expand_dims(leaf.state(), axis=0), device=th.device(self.device), dtype =th.float)
            _, pi, value = self.model(state)
            if leaf.env.done == True:
                value = - leaf.depth
            else:
                value = - leaf.depth - leaf.env.state.distance_pins2wire()
                # value = - leaf.depth - self.randomRollout(copy.deepcopy(leaf.env))
            dirichlet = np.random.dirichlet(DIRICHLET_ALPHA * np.ones(len(leaf.legalActions())))
            for idx, action in enumerate(leaf.legalActions()):
                env = copy.deepcopy(leaf.env)
                env.step(action)
                leaf.addChild(action, 1/len(leaf.legalActions()), env)
                # if DIRICHLET and leaf.parent is None:
                #     leaf.addChild(action, 0.75*pi[0][Point2Int(action)].item() + 0.25*dirichlet[idx], env)
                # else:
                #     leaf.addChild(action, pi[0][Point2Int(action)].item(), env)
            if leaf.env.done == True:
                # print(-leaf.depth)
                return -leaf.depth
            else:
                # print(value.item())
                return value
                # return value.item()
    
    def backup(self, leaf: Node, leaf_value: float):
        current = leaf
        while current != None:
            current.max_update(leaf_value)
            current = current.parent

    def simulation(self, num_sim) -> None:
        select, expandAndEvaluate, backup = 0, 0, 0
        for i in range(num_sim):
            t0 = time.time()
            leaf = self.select()
            t1 = time.time()
            leaf_value = self.expandAndEvaluate(leaf)
            t2 = time.time()
            self.backup(leaf, leaf_value)
            t3 = time.time()
            select += t1 - t0
            expandAndEvaluate += t2 - t1
            backup += t3 - t2
        # print("time: select =  %5.2f expand & evaluate = %5.2f backup = %5.2f" % (select, expandAndEvaluate, backup))

    def predict(self, tau: float = 1) -> dict():
        probabilities = {}
        if PRINT_MCTS:
            print("root ",self.root.N)
            print("Q = ", -self.root.Q)
        sum_exp = 0
        for item in self.root.children.items():
            sum_exp += item[1].N ** (1 / tau)
        for item in self.root.children.items(): 
            probabilities[item[0]] = item[1].N ** (1 / tau) / sum_exp
            if PRINT_MCTS:
                print("action = ", item[0]," N = %4d Q = %6.2f Qn = %6.2f U = %6.2f P = %5.2f pi = %5.2f" % (item[1].N, item[1].Q,item[1].max_getQn(), item[1].max_getU(), item[1].P, probabilities[item[0]]))
        return probabilities
    
    def step(self, action) -> bool:
        _, _, done, _ = self.env.step(action)
        if KEEP_CHILD:
            # Qmax = self.root.Q_max
            self.root = self.root.children[action]
            self.root.parent = None
            # self.root.Q_max = Qmax
            # self.root.Q_min = 0
            if DIRICHLET:
                dirichlet = np.random.dirichlet(DIRICHLET_ALPHA * np.ones(len(self.root.children.items())))
                for idx, item in enumerate(self.root.children.items()):
                    item[1].P = 0.75*item[1].P + 0.25*dirichlet[idx]
        else:
            self.root = Node(None, 1.0, self.root.depth + 1, self.env)
        
        return done
    


if __name__ == '__main__':

    print("Generate layouts ......", end=" ")
    layouts_train = [Layout(LAYOUT_SIZE) for _ in range(NUM_LAYOUT_TRAIN)]
    print("Finish!")
    env = RoutingEnv(layouts_train)
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    # model = CustomPPO.load("PPO_10x10_1")
    model = 0

    count = 0
    for _ in range(10000):
        MCTS = mcts(copy.deepcopy(env), 0)
        count +=MCTS.randomRollout(copy.deepcopy(env))
        env.reset()

    # for _ in range(10000):
    #     MCTS = mcts(copy.deepcopy(env), 0)
    #     count += MCTS.env.state.distance_pins2wire()
    #     env.reset()
    
    print(count/10000)

    # total_step = 0
    # for _ in range(10):
    #     MCTS = mcts(copy.deepcopy(env), model)
    #     step = 0
    #     while True:
    #         # mcts.render()
    #         start = time.time()
    #         MCTS.simulation(NUM_SIMULATION)
    #         end = time.time()
    #         print("simulation time = ",end - start)
    #         mcts_pi = MCTS.predict()
    #         action = max(mcts_pi.items(), key = lambda a: a[1])[0]
    #         done = MCTS.step(action)
    #         step += 1
    #         if done:
    #             MCTS.render()
    #             total_step += step
    #             env.reset()
    #             break
    # print("avg_return = ", total_step/10)


    


    


    

