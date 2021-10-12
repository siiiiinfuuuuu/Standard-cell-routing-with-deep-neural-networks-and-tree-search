import random

from numpy.lib import select
random.seed(2)

import numpy as np
import queue
from config import *
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import copy

class Point:
    def __init__(self, x: int , y: int, z: int):
        self.x = x
        self.y = y
        self.z = z
    def __deepcopy__(self, memodict={}):
        copy_object = Point(self.x, self.y, self.z)
        return copy_object
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Point(x, y, z)
    def __eq__(self, other: "Point"):
        return self.x == other.x and self.y == other.y and self.z == other.z
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    def __str__(self):
        return "(" + format(self.x,'2d') + ", " + format(self.y,'2d') + ", " + format(self.z,'2d') + ")"

directions = [Point(0, -1, 0), Point(-1, 0, 0), Point(1, 0, 0), Point(0, 1, 0), Point(0, 0, 1), Point(0, 0, -1)]
neighbors = [Point(-1, -1, 0), Point(0, -1, 0), Point(1, -1, 0), Point(-1, 0, 0), Point(1, 0, 0), Point(-1, 1, 0), Point(0, 1, 0), Point(1, 1, 0)]

class Net:
    def __init__(self, layout_size: List[int]):
        self.size = layout_size
        self.pins = [] #pins coor
        self.routed_pins = []
        self.pins_state = np.zeros((layout_size[0], layout_size[1], layout_size[2]), dtype=np.bool_)
        self.wires = [] # wires coor
        self.wires_state = np.zeros((layout_size[0], layout_size[1], layout_size[2]), dtype=np.bool_)
        self.actions = set()
        self.actions_state = np.zeros((layout_size[0], layout_size[1], layout_size[2]), dtype=np.bool_)
        self.done = False

    def __deepcopy__(self, memodict={}):
        copy_object = Net(self.size)
        copy_object.pins = []
        for item in self.pins:
            copy_object.pins.append(copy.deepcopy(item))
        copy_object.routed_pins =  []
        for item in self.routed_pins:
            copy_object.routed_pins.append(copy.deepcopy(item))
        copy_object.pins_state =  np.copy(self.pins_state)
        copy_object.wires =  []
        for item in self.wires:
            copy_object.wires.append(copy.deepcopy(item))
        copy_object.wires_state =  np.copy(self.wires_state)
        copy_object.actions = []
        for item in self.actions:
            copy_object.actions.append(copy.deepcopy(item))
        copy_object.actions = set(copy_object.actions)
        copy_object.actions_state =  np.copy(self.actions_state)
        copy_object.done =  self.done
        return copy_object

    def deletePin(self, p: "Point"):
        self.pins.remove(p)
        self.routed_pins.append(p)
        self.pins_state[p.x][p.y][p.z] = 0
        if len(self.pins) == 0:
            self.actions = set()
            self.actions_state = np.zeros((self.size[0], self.size[1], self.size[2]), dtype=np.bool_)
            self.done = True
    def addWire(self, p: "Point"):
        self.wires.append(p)
        self.wires_state[p.x][p.y][p.z] = 1
    def updataActions(self, new_wire: "Point", empty_state: np.ndarray):
        for dir in directions:
            new_action = new_wire + dir
            if self.inLayout(new_action):
                if self.isPin(new_action) or empty_state[new_action.x][new_action.y][new_action.z] == 1:
                    self.actions.add(new_action)
                    self.actions_state[new_action.x][new_action.y][new_action.z] = 1
    def isPin(self, p: "Point"):
        return self.pins_state[p.x][p.y][p.z] == 1
    def isWire(self, p: "Point"):
        return self.wires_state[p.x][p.y][p.z] == 1
    def inLayout(self, p: "Point"):
        return p.x >= 0 and p.x < self.size[0] and p.y >= 0 and p.y < self.size[1] and p.z >=0 and p.z < self.size[2]

class Layout:  
    def __init__(self, layout_size: List[int]):
        self.nets = []
        self.obstacles = []
        self.obstacles_state = np.zeros((layout_size[0], layout_size[1], layout_size[2]), dtype=np.bool_)
        self.size = layout_size
        self.ligalPos = np.zeros((self.size[0], self.size[1], self.size[2]), dtype=int) # 1 for obstacle ,2 for legal 
        self.randomInit()

    def randomInit(self):
        self.__generateObstacles()
        self.__checkLigalPosition()
        self.__generateNets()
        
    def __generateObstacles(self):
        # random generate rectangle obstacles
        for _ in range(NUM_OBS):
            width = random.randint(OBS_SIZE[0][0], OBS_SIZE[0][1])
            height = random.randint(OBS_SIZE[1][0], OBS_SIZE[1][1])
            # don't generate on boundary(to fit MLOARST)
            pos = Point(random.randint(1, self.size[0] - 1 - width),random.randint(1, self.size[1] - 1 - height),random.randint(0, self.size[2] - 1))
            pos_ = Point(pos.x + width -1, pos.y + height -1, pos.z)
            self.__addObstacle(pos, pos_)
            for x in range(pos.x, pos_.x + 1):
                for y in range(pos.y, pos_.y + 1):
                    self.ligalPos[x][y][pos.z] = 1
    def __checkLigalPosition(self):
        # check available pin position
        q = queue.Queue()
        q.put(Point(0, 0, 0))
        self.ligalPos[0][0][0] = 2
        while not q.empty():
            point = q.get()
            # rewrite to for loop
            if point.x - 1 >= 0 and self.ligalPos[point.x - 1][point.y][point.z] == 0:
                self.ligalPos[point.x - 1][point.y][point.z] = 2
                q.put(Point(point.x - 1, point.y, point.z))
            if point.x + 1 < self.size[0] and self.ligalPos[point.x + 1][point.y][point.z] == 0:
                self.ligalPos[point.x + 1][point.y][point.z] = 2
                q.put(Point(point.x + 1, point.y, point.z))
            if point.y - 1 >= 0 and self.ligalPos[point.x][point.y - 1][point.z] == 0:
                self.ligalPos[point.x][point.y - 1][point.z] = 2
                q.put(Point(point.x, point.y - 1, point.z))
            if point.y + 1 < self.size[1] and self.ligalPos[point.x][point.y + 1][point.z] == 0:
                self.ligalPos[point.x][point.y + 1][point.z] = 2
                q.put(Point(point.x, point.y + 1, point.z))
            if point.z - 1 >= 0 and self.ligalPos[point.x][point.y][point.z - 1] == 0:
                self.ligalPos[point.x][point.y][point.z - 1] = 2
                q.put(Point(point.x, point.y, point.z - 1))
            if point.z + 1 < self.size[2] and self.ligalPos[point.x][point.y][point.z + 1] == 0:
                self.ligalPos[point.x][point.y][point.z + 1] = 2
                q.put(Point(point.x, point.y, point.z + 1))
    def __generateNets(self):
        for _ in range(NUM_NET):
            n = Net(self.size)
            num_pins = random.randint(NUM_PIN[0], NUM_PIN[1])
            for _ in range(num_pins):
                pin = Point(random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1), random.randint(0, self.size[2] - 1))
                while(self.ligalPos[pin.x][pin.y][pin.z] != 2):
                    pin = Point(random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1), random.randint(0, self.size[2] - 1))
                self.ligalPos[pin.x][pin.y][pin.z] = -1
                n.pins.append(pin)
                n.pins_state[pin.x][pin.y][pin.z] = 1
                for action in neighbors:
                    neighbor = pin + action
                    if neighbor.x >= 0 and neighbor.x < LAYOUT_SIZE[0] and neighbor.y >= 0 and neighbor.y < LAYOUT_SIZE[1] and self.ligalPos[neighbor.x][neighbor.y][neighbor.z] == 2:
                        self.ligalPos[neighbor.x][neighbor.y][neighbor.z] = 0
            self.nets.append(n)
    def __addObstacle(self, p: "Point", p_: "Point"):
        self.obstacles.append([p, p_])
        for x in range(p.x, p_.x + 1):
                for y in range(p.y, p_.y + 1):
                    self.obstacles_state[x][y][p.z] = 1


            
if __name__ == '__main__':
    l = Layout(LAYOUT_SIZE)
    for z in range(l.size[2]):
        print("     =====Layer ",z,"=====")
        for y in reversed(range(l.size[1])):
            for x in range(l.size[0]):
                if(l.ligalPos[x][y][z] == -1):
                    print("P", end=" ")
                    # print(color.BLUE + "P" + color.END, end=" ")
                if(l.ligalPos[x][y][z] == 0):
                    print(color.RED + "." + color.END , end=" ")
                if(l.ligalPos[x][y][z] == 1):
                    print("X", end=" ")
                if(l.ligalPos[x][y][z] == 2):
                    print(color.RED + "." + color.END , end=" ")
            print("")


