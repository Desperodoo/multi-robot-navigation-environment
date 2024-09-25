import hydra
import time
import numpy as np
import random
from numba import njit, prange
from utils.Occupied_Grid_Map import OccupiedGridMap
from abc import ABCMeta,abstractmethod
from utils.agent import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from omegaconf import DictConfig, OmegaConf


@njit()
def bresenham_line(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    err = dx - dy
    line = []
    while True:
        line.append((x0, y0))
        if ((x0 - x1) < 0.001 and (x0 - x1) > -0.001) and ((y0 - y1) < 0.001 and (y0 - y1) > -0.001):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return line


@njit()
def _communicate(states, active, grid_map, comm_range, width, height):
    num_defender = len(states)
    adj_mat = np.zeros((num_defender, num_defender))
    
    for i in prange(num_defender):
        for j in range(i, num_defender):  # 只处理i<=j的情况，减少重复计算
            if active[i] and active[j] and (np.sqrt((states[i][0] - states[j][0]) ** 2 + (states[i][1] - states[j][1]) ** 2) <= comm_range):
                block = False
                ray_indices = bresenham_line(states[i][0], states[i][1], states[j][0], states[j][1])
                for k in range(len(ray_indices)):
                    index = ray_indices[k]
                    if (0 < index[0] < width and 0 < index[1] < height) and grid_map[index[0], index[1]] == 1:
                        block = True
                        break
                if not block:
                    adj_mat[i, j] = adj_mat[j, i] = 1
    return adj_mat


# @njit()
# def _communicate_with_protector(states_defender, states_protector, active, grid_map, sen_range, width, height):
#     num_defender = len(states_defender)
#     num_protector = len(states_protector)
#     adj_mat = np.zeros((num_defender, num_protector))
    
#     for i in prange(num_defender):
#         for j in range(num_protector):  # 只处理i<=j的情况，减少重复计算
#             if active[i] and (np.sqrt((states_defender[i][0] - states_protector[j][0]) ** 2 + (states_defender[i][1] - states_protector[j][1]) ** 2) <= sen_range):
#                 block = False
#                 ray_indices = bresenham_line(states_defender[i][0], states_defender[i][1], states_protector[j][0], states_protector[j][1])
#                 for k in range(len(ray_indices)):
#                     index = ray_indices[k]
#                     if (0 < index[0] < width and 0 < index[1] < height) and grid_map[index[0], index[1]] == 1:
#                         block = True
#                         break
#                 if not block:
#                     adj_mat[i, j] = 1
#     return adj_mat



class BaseEnv(metaclass=ABCMeta):
    def __init__(self, map_config, env_config, defender_config, attacker_config, sensor_config):
        # Agent config
        self.defender_class = eval(env_config.defender_class)
        self.attacker_class = eval(env_config.attacker_class)
        # Simulation config
        self.time_step = 0
        self.n_episode = 0
        self.max_steps = env_config.max_steps
        self.step_size = env_config.step_size

        self.num_target = env_config.num_target
        self.num_defender = env_config.num_defender
        self.num_attacker = env_config.num_attacker

        self.defender_config = defender_config
        self.attacker_config = attacker_config
        self.sensor_config = sensor_config
        self.map_config = map_config
        self.env_config = env_config
        
        if self.env_config.task == 'Navigation':
            self.init_attacker = self.init_attacker_navigation
        elif self.env_config.task == 'PE':
            self.init_attacker = self.init_attacker_PE_game
        
    def init_map(self, map_info=None):
        self.occupied_map = OccupiedGridMap(map_config=self.map_config)
        if map_info is not None:
            [grid_map, obstacles] = map_info
            self.occupied_map.grid_map = grid_map
            self.occupied_map.obstacles = obstacles
        else:
            self.occupied_map.initailize_obstacle()
        self.occupied_map.get_boundary_map(max_num=self.map_config.max_num_obstacle)
        self.occupied_map.extend_obstacles(extend_dis=1)
        self.occupied_map.get_raser_map(num_beams=self.sensor_config.num_beams, radius=self.defender_config.sen_range)
        
    def init_target(self, idx=None, target=None):
        """initialize the target position
        """
        (width, height) = self.occupied_map.boundaries
        if idx is not None:
            while True:
                pos = (
                    random.randint(0, width - 1),
                    random.randint(0, height - 1)
                )
                if self.occupied_map.is_unoccupied(pos=pos, map=self.occupied_map.ex_grid_map):
                    self.target[idx] = pos
                    break
        else:
            if target is not None:
                self.target = target
            else:
                self.target = list()
                while len(self.target) < self.num_target:
                    pos = (
                        random.randint(0, width - 1),
                        random.randint(0, height - 1)
                    )
                    if self.occupied_map.is_unoccupied(pos=pos, map=self.occupied_map.ex_grid_map):
                        self.target.append(pos)
                    
    # def init_random_target(self, idx=None):
    #     """initialize the target for protector
    #     """
    #     (width, height) = self.occupied_map.boundaries
    #     if idx is not None:
    #         while True:
    #             pos = (
    #                 random.randint(0, width - 1),
    #                 random.randint(0, height - 1)
    #             )
    #             if self.occupied_map.is_unoccupied(pos=pos, map=self.occupied_map.ex_grid_map):
    #                 self.random_target[idx] = pos
    #                 break
    #     else:
    #         self.random_target = list()
    #         while len(self.random_target) < self.num_protector:
    #             pos = (
    #                 random.randint(0, width - 1),
    #                 random.randint(0, height - 1)
    #             )
    #             if self.occupied_map.is_unoccupied(pos=pos, map=self.occupied_map.ex_grid_map):
    #                 self.random_target.append(pos)
    
    def init_defender(self, position_list=None):
        if position_list is not None:
            self.defender_list = []
            self.position_list = position_list
            for pos in self.position_list:
                self.defender_list.append(self.defender_class(x=pos[0], y=pos[1], defender_cfg=self.defender_config, map_size=self.map_config.map_size))
        else:
            (width, height) = self.occupied_map.boundaries
            self.defender_list = []
            self.position_list = []
            min_dist = self.defender_config.min_dist
            start_time = time.time()
            while len(self.position_list) < self.num_defender:
                pos = tuple(np.random.rand(2) * np.array([width - 1, height - 1]))

                if self.occupied_map.is_unoccupied(pos=pos, map=self.occupied_map.ex_grid_map):
                    collision = False
                    connectivity = len(self.defender_list) == 0  # True if first defender
                    dist_list = list()
                    for p in self.position_list:
                        dist = np.linalg.norm((pos[0] - p[0], pos[1] - p[1]))
                        if dist < min_dist:
                            collision = True
                            break  # 提前终止
                        dist_list.append(dist)

                    if not connectivity:
                        for d in dist_list:
                            # 检查连通性
                            block = any(self.occupied_map.grid_map[index] == 1 for index in bresenham_line(round(pos[0]), round(pos[1]), round(p[0]), round(p[1])))
                            connectivity = (not block) and (d < self.defender_config.comm_range)
                            if connectivity: break  # 提前终止

                    if (connectivity and not collision) or (time.time() - start_time > 1):
                        if (time.time() - start_time > 1) and (min_dist > 1):
                            min_dist -= 1
                        self.position_list.append(pos)
                        self.defender_list.append(self.defender_class(x=pos[0], y=pos[1], defender_cfg=self.defender_config, map_size=self.map_config.map_size))
                        start_time = time.time()

    def init_attacker_navigation(self):
        self.attacker_list = list()
        for pos in self.target:
            self.attacker_list.append(self.attacker_class(x=pos[0], y=pos[1], attacker_config=self.attacker_config))

    # def init_protector(self):
    #     (width, height) = self.occupied_map.boundaries
    #     self.protector_list = list()
    #     while len(self.protector_list) < self.num_protector:
    #         pos = (
    #             random.randint(0, width - 1),
    #             random.randint(0, height - 1)
    #         )
    #         if self.occupied_map.is_unoccupied(pos=pos, map=self.occupied_map.ex_grid_map):
    #             self.protector_list.append(self.protector_class(x=pos[0], y=pos[1], protector_config=self.protector_config))

    def get_reward(self):
        reward = []
        for defender in self.defender_list:
            reward.append(self.defender_reward(defender))
        return reward

    def get_state(self, agent_type: str):
        """get states of the collective
        
        """
        agent_list = getattr(self, agent_type + '_list')
        state = np.array([self.get_agent_state(agent) for agent in agent_list], dtype=np.float32)
        return state

    def get_agent_state(self, agent):
        return [agent.x, agent.y, agent.vx, agent.vy]

    def init_attacker_PE_game(self, should_be_percepted: bool=False):
        """initialize the attacker.

        Args:
            should_be_percepted (bool): whether the attacker should be percepted in the initial state, true in pursuit_env, false in navigation and coverage_env.
        """
        (width, height) = self.occupied_map.boundaries
        agent_block = self.get_state(agent_type='defender')        
        position_list = list()
        self.attacker_list = list()

        while len(position_list) < self.num_attacker:
            pos = (random.randint(0, width - 1), random.randint(0, height - 1))
            if self.occupied_map.is_unoccupied(pos=pos, map=self.occupied_map.ex_grid_map):
                if should_be_percepted:
                    for block in agent_block:
                        dist = np.linalg.norm([block[0] - pos[0], block[1] - pos[1]])
                        if dist < self.defender_config.sen_range:
                            position_list.append(pos)
                            break
                else:
                    position_list.append(pos)
        for pos in position_list:
            self.attacker_list.append(self.attacker_class(x=pos[0], y=pos[1], attacker_config=self.attacker_config))

    def communicate(self, next_state=None):
        active = self.get_active()
        if next_state is None:
            states = np.round(self.get_state(agent_type='defender')).astype(np.int64)
        else:
            states = np.round(np.array(next_state)).astype(np.int64)
        (width, height) = self.occupied_map.boundaries
        adj_mat = _communicate(states, active, self.occupied_map.grid_map, self.defender_config.comm_range, width, height)
        return adj_mat.astype(np.float32)
    
    # def communicate_with_protector(self):
    #     active = self.get_active()
    #     defender_states = np.round(self.get_state(agent_type='defender')).astype(np.int64)
    #     protector_states = np.round(self.get_state(agent_type='protector')).astype(np.int64)
    #     (width, height) = self.occupied_map.boundaries
    #     adj_mat = _communicate_with_protector(defender_states, protector_states, active, self.occupied_map.grid_map, self.defender_config.sen_range, width, height)
    #     return adj_mat.astype(np.float32)
    
    def get_active(self):
        active = []
        for defender in self.defender_list:
            active.append(1 if defender.active else 0)
        return active
    
    def get_target(self):
        return np.array([[*target, 0, 0] for target in self.target], dtype=np.float32)

    # @abstractmethod
    # def reset(self):
    #     pass

    # @abstractmethod
    # def step(self, action):
    #     pass

    # @abstractmethod
    # def defender_reward(self, agent_idx, is_pursuer=True):
    #     pass

    # @abstractmethod
    # def communicate(self):
    #     pass


############################################################## Unit test #########################################################
def draw_obstacle(x, y, box_width, color):
    left, bottom, wid, hei = (x - box_width / 2, y - box_width / 2, box_width, box_width)
    rect = patches.Rectangle(xy=(left, bottom), width=wid, height=hei, color=color)
    return rect


def map_plotting(width, height, obstacles, obstacle_agent, ex_obstacles, defender_pos, attacker_pos, raser_map):
    """
    :param step: time steps, type=int
    :param height: height of the map, type=int
    :param width: width of the map, type=int
    :param obstacles: list of obstacles, shape=(n, 2)
    :param boundary obstacles: list of boundary of the obstacles
    :param exteneded obstacles : list of extended obstacles
    :param box_width: describe the mesh density, type=int

    :return:
    """
    fig3 = plt.figure(3, figsize=(5, 5 * 0.9))
    ax3 = fig3.add_axes(rect=[0.12, 0.1, 0.8, 0.82])

    ax3.set_title('Map', size=12)
    ax3.set_xlabel('x/(m)', size=12)
    ax3.set_ylabel('y/(m)', size=12)
    # ax3.grid(color='black', linestyle='--', linewidth=0.8)
    ax3.axis([0, height - 1, 0, width - 1])
    ax3.set_aspect('equal')
    
    for obstacle in obstacle_agent:
        rect = draw_obstacle(x=obstacle[0], y=obstacle[1], box_width=0.8, color='black')
        ax3.add_patch(rect)

    # draw obstacles
    for obstacle in obstacles:
        rect = draw_obstacle(x=obstacle[0], y=obstacle[1], box_width=0.4, color='red')
        ax3.add_patch(rect)

    for obstacle in ex_obstacles:
        rect = draw_obstacle(x=obstacle[0], y=obstacle[1], box_width=0.4, color='grey')
        ax3.add_patch(rect)
    
    ax3.scatter(x=defender_pos[:, 0], y=defender_pos[:, 1], color='green', s=0.5)
    ax3.scatter(x=attacker_pos[:, 0], y=attacker_pos[:, 1], color='red', s=0.5)
    # ax3.scatter(x=protector_pos[:, 0], y=protector_pos[:, 1], color='blue', s=0.5)
    # for i, obs in enumerate(obstacle_agent):
    #     if obstacle_adj[i] == 1:
    #         ax3.plot([pos[0], obs[0]], [pos[1], obs[1]], color='green', linestyle='--', alpha=0.5)

    fig3.savefig('map.png')


def resolution_scaling(cfg):
    resolution = cfg.map.resolution
    cfg_dict = OmegaConf.to_container(cfg)
    cfg_dict['defender']['collision_radius'] = cfg_dict['defender']['collision_radius'] / resolution
    cfg_dict['defender']['comm_range'] = cfg_dict['defender']['comm_range'] / resolution
    cfg_dict['defender']['sen_range'] = cfg_dict['defender']['sen_range'] / resolution
    cfg_dict['defender']['vmax'] = cfg_dict['defender']['vmax'] / resolution
    cfg_dict['defender']['tau'] = cfg_dict['defender']['tau'] / resolution

    cfg_dict['attacker']['collision_radius'] = cfg_dict['attacker']['collision_radius'] / resolution
    cfg_dict['attacker']['extend_dis'] = cfg_dict['attacker']['extend_dis'] / resolution
    cfg_dict['attacker']['comm_range'] = cfg_dict['attacker']['comm_range'] / resolution
    cfg_dict['attacker']['sen_range'] = cfg_dict['attacker']['sen_range'] / resolution
    cfg_dict['attacker']['vmax'] = cfg_dict['attacker']['vmax'] / resolution
    cfg_dict['attacker']['tau'] = cfg_dict['attacker']['tau'] / resolution

    # cfg_dict['protector']['collision_radius'] = cfg_dict['protector']['collision_radius'] / resolution
    # cfg_dict['protector']['extend_dis'] = cfg_dict['protector']['extend_dis'] / resolution
    # cfg_dict['protector']['comm_range'] = cfg_dict['protector']['comm_range'] / resolution
    # cfg_dict['protector']['sen_range'] = cfg_dict['protector']['sen_range'] / resolution
    # cfg_dict['protector']['vmax'] = cfg_dict['protector']['vmax'] / resolution
    # cfg_dict['protector']['tau'] = cfg_dict['protector']['tau'] / resolution

    cfg_dict['env']['goal_range'] = cfg_dict['env']['goal_range'] / resolution

    cfg_dict['map']['center'] = [cfg_dict['map']['center'][0] / resolution, cfg_dict['map']['center'][1] / resolution]
    cfg_dict['map']['variance_u_shaped'] = cfg_dict['map']['variance_u_shaped'] / resolution
    cfg_dict['map']['map_size'] = [int(cfg_dict['map']['map_size'][0] / resolution), int(cfg_dict['map']['map_size'][1] / resolution)]
    cfg_dict['map']['u_shape'] = [int(cfg_dict['map']['u_shape'][0] / resolution), int(cfg_dict['map']['u_shape'][1] / resolution)]
    cfg_dict['map']['block_shape'] = [int(cfg_dict['map']['block_shape'][0] / resolution), int(cfg_dict['map']['block_shape'][1] / resolution)]
    block_shape = cfg_dict['map']['block_shape']
    u_shape = cfg_dict['map']['u_shape']
    boundary = block_shape[0] * block_shape[1] - (block_shape[0] - 2) * (block_shape[1] -2)
    u_boundary = u_shape[0] + 2 * u_shape[1]
    cfg_dict['map']['max_num_obstacle'] = 2 * sum(cfg_dict['map']['map_size']) + boundary * cfg_dict['map']['num_blocks'] + u_boundary * cfg_dict['map']['num_u_shaped'] * 2
    cfg = OmegaConf.create(cfg_dict)
    return cfg


@hydra.main(config_path='./', config_name='navigation.yaml', version_base=None)
def main(cfg):
    cfg = resolution_scaling(cfg=cfg)
    env = BaseEnv(cfg.map, cfg.env, cfg.defender, cfg.attacker, cfg.sensor)
    t1 = time.time()
    env.init_map()
    print(time.time() - t1)
    grid_map = env.occupied_map

    t1 = time.time()
    env.init_target()
    print(time.time() - t1)
    t1 = time.time()
    env.init_defender()
    print(time.time() - t1)
    t1 = time.time()
    env.init_attacker()
    print(time.time() - t1)
    # t1 = time.time()
    # env.init_protector()
    # print(time.time() - t1)
    # t1 = time.time()
    # env.init_random_target()
    # print(time.time() - t1)
    t1 = time.time()
    env.communicate()
    print(time.time() - t1)

    defender_pos = env.get_state(agent_type='defender')
    attacker_pos = env.get_state(agent_type='attacker')
    # protector_pos = env.get_state(agent_type='protector')
    map_plotting(
        width=grid_map.boundaries[0], 
        height=grid_map.boundaries[1], 
        obstacles=grid_map.obstacles, 
        obstacle_agent=grid_map.obstacle_agent,
        ex_obstacles=grid_map.ex_obstacles,
        defender_pos=defender_pos,
        attacker_pos=attacker_pos,
        # protector_pos=protector_pos,
        raser_map=grid_map.raser_map
    )


if __name__ == "__main__":
    main()