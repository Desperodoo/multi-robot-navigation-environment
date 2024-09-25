import time, hydra
import warnings
import random
import numpy as np
from numba import njit, int32
from copy import deepcopy
from base_env import BaseEnv, bresenham_line, resolution_scaling
from utils.astar import parallel_path_planning
from numba.core.errors import NumbaDeprecationWarning, NumbaWarning, NumbaPendingDeprecationWarning
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation


warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)   


@njit()    
def get_local_sensed_map(num_agent: int32, width: int32, height: int32, occupied_map: np.ndarray, radius: int32, pos: np.ndarray, max_num_obstacle: int32, num_obstacles: int32, obstacles: np.ndarray, num_beams: int32): # type: ignore
    obstacle_adj = np.zeros((num_agent, max_num_obstacle))
    times = num_beams * num_agent
    for i in range(times):
        agent = i // num_beams
        beam = i % num_beams
        beam_angle = beam * 2 * np.pi / num_beams
        beam_dir_x = np.cos(beam_angle)
        beam_dir_y = np.sin(beam_angle)
        x = pos[agent, 0]
        y = pos[agent, 1]
        for beam_range in range(1, radius):
            beam_current_x = x + beam_range * beam_dir_x
            beam_current_y = y + beam_range * beam_dir_y
            if x < 0 or beam_current_x < 0 or beam_current_x >= width or beam_current_y < 0 or beam_current_y >= height :
                break
            beam_current_x = round(beam_current_x)
            beam_current_y = round(beam_current_y)
            if occupied_map[beam_current_x, beam_current_y] == 1:
                for idx in range(num_obstacles):
                    if obstacles[idx, 0] == beam_current_x and obstacles[idx, 1] == beam_current_y:
                        obstacle_adj[agent, idx] = 1
                        break
                break
    return obstacle_adj


class PE_Env(BaseEnv):
    def __init__(self, cfg):
        super().__init__(cfg.map, cfg.env, cfg.defender, cfg.attacker, cfg.sensor)
        self.difficulty = cfg.env.difficulty
        self.extend_dis = int(cfg.attacker.extend_dis)
        self.max_num_obstacle = cfg.map.max_num_obstacle
        self.goal_range = cfg.env.goal_range
        
    def store_map(self):
        map_copy = [self.occupied_map.grid_map, self.occupied_map.obstacles, self.target, self.position_list]
        return map_copy
        
    def reset(self):
        self.group_dis = np.zeros(shape=(self.num_defender, self.num_defender))
        self.time_step = 0
        self.n_episode += 1
        self.collision = False
        self.num_component = None
        self.diameter_reciprocal = None

        self.init_map()  # grid_map, obstacles, boundary_map, boundary_obstacles, obstacle_agents, ex_grid_map, ex_obstacles
        self.init_target()
        self.init_defender()
        
        self.num_obstacle = len(self.occupied_map.boundary_obstacles)
        self.init_attacker()
        self.caught = False
        
    def step(self, action):
        next_state = list()
        rewards = list()
        self.time_step += 1
        adj = self.communicate()
        for idx, defender in enumerate(self.defender_list):
            next_state.append(defender.step(action[idx]))

        # reward 1: Collision and Arriving Reward
        for idx, defender in enumerate(self.defender_list):
            state = next_state[idx]
            if defender.active:
                reward, collision, caught = self.defender_reward(idx, state, next_state)  # self.target[idx]
                rewards.append(reward)
                if caught:
                    self.caught = caught
                    agent_idx = idx
                if collision:
                    defender.dead()
                else:
                    defender.apply_update(state)
            else:
                rewards.append(0)
        # reward 2: Cooperative Reward 
        if self.caught:
            neighbor_set = adj[agent_idx]
            for neigh, conn in enumerate(neighbor_set):
                if (neigh != agent_idx) and conn:
                    rewards[neigh] = rewards[neigh] + 1

        active = self.get_active()
        done = True if (self.time_step >= self.max_steps) or (sum(active) == 0) else False
        info = []
        return rewards, done, info
    
    def func(self, x):
        return -0.0022 * x**3 + 0.0499 * x ** 2 + -0.3840 * x + 1.0000

    def defender_reward(self, idx, state, next_state):
        reward = 0
        collision = False
        caught = False
        inner_collision = self.collision_detection(state, obstacle_type='defender', next_state=next_state, idx=idx)
        reward -= (sum(inner_collision) - 1) * 100
        if reward < 0:
            collision = True
            return reward, collision, caught
        obstacle_collision = self.collision_detection(state, obstacle_type='obstacle')
        reward -= obstacle_collision * 100
        if reward < 0:
            collision = True
            return reward, collision, caught
        
        is_collision = self.collision_detection(state, obstacle_type='attacker')
        reward = is_collision * 0.1
        if is_collision:
            caught = True
        return reward, collision, caught

    def collision_detection(self, state, obstacle_type: str = 'obstacle', next_state: list = None, target: tuple = None, idx=None):
        if obstacle_type == 'obstacle':
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    inflated_pos = (state[0] + i * self.defender_config.collision_radius, state[1] + j * self.defender_config.collision_radius)
                    if not self.occupied_map.is_unoccupied(inflated_pos, self.occupied_map.grid_map):
                        return True
            return False
        elif obstacle_type == 'defender':
            # 使用向量化的方式计算与每个防御者的距离，然后判断是否小于碰撞半径
            distances = np.linalg.norm(np.array(next_state)[:, :2] - state[:2], axis=1)
            self.group_dis[idx, :] = distances
            return distances <= self.defender_config.collision_radius
        elif obstacle_type =='attacker':
            distances = np.linalg.norm(self.get_state(agent_type=obstacle_type)[:, :2] - state[:2], axis=1)
            return distances <= self.defender_config.collision_radius

    def update_attacker(self, idx):
        (width, height) = self.occupied_map.boundaries
        while True:
            pos = (
                random.randint(0, width - 1),
                random.randint(0, height - 1)
            )
            if self.occupied_map.is_unoccupied(pos, map=self.occupied_map.ex_grid_map):
                break
        self.target[idx] = pos
        (self.attacker_list[idx].x, self.attacker_list[idx].y) = pos
    
    def attacker_replan(self):
        attacker_state = np.round(self.get_state(agent_type='attacker')).astype(np.int64)
        target = np.array(self.target, dtype=np.int64)
        defender_state = self.get_state(agent_type='defender').tolist()
        (width, height) = self.occupied_map.boundaries
        attacker_indices = list()
        for i in range(self.num_attacker):
            if self.attacker_list[i].active:
                attacker_indices.append(i)
        if self.time_step % self.difficulty == 0: 
            for dis in range(self.extend_dis, -1, -1):
                self.occupied_map.set_moving_obstacle(pos=defender_state)
                self.occupied_map.extend_moving_obstacles(extend_dis=dis)
                random_indices, paths = parallel_path_planning(attacker_state[attacker_indices], target[attacker_indices], width, height, obs=np.array(self.occupied_map.ex_moving_obstacles + self.occupied_map.ex_obstacles))
                sorted_indices = [attacker_indices[random_indices[i]] for i in range(len(random_indices))]
                attacker_indices = list()
                for i, ids in enumerate(sorted_indices):
                    attacker = self.attacker_list[ids]
                    path = paths[i]
                    if len(path) >= 2:
                        attacker.path = path
                    else:
                        if dis == 0:
                            attacker.path = path
                        else:
                            attacker_indices.append(i)
            
    def attacker_step(self):
        for idx, attacker in enumerate(self.attacker_list):
            if attacker.active:
                if len(attacker.path) >= 2:
                    if np.linalg.norm((attacker.x - attacker.path[-1][0], attacker.y - attacker.path[-1][1])) < self.map_config.resolution * 0.4:
                        attacker.path.pop()
                way_point = attacker.path[-1]
                [x, y, vx, vy, theta] = attacker.step(way_point)
                if self.occupied_map.is_unoccupied((x, y), map=self.occupied_map.grid_map):
                    attacker.apply_update([x, y, vx, vy, theta])
                if np.linalg.norm((self.target[idx][0] - x, self.target[idx][1] - y)) <= self.attacker_config.collision_radius:
                    self.init_target(idx=idx)

    def merge_static_dynamic_obstacles(self, protector_state=None, defender_state=None):
        static_obstacles = deepcopy(self.occupied_map.grid_map)
        if protector_state is not None:
            for state in protector_state:
                static_obstacles[state[0], state[1]] = 1
        if defender_state is not None:
            for state in defender_state:
                static_obstacles[state[0], state[1]] = 1
        return static_obstacles

    def sensor(self):
        defender_state = np.round(self.get_state(agent_type='defender')).astype(np.int64)
        merged_obstacle_grid = self.merge_static_dynamic_obstacles(defender_state=defender_state)
        
        obstacle_adj_list = get_local_sensed_map(
            num_agent=self.num_defender,
            width=self.occupied_map.boundaries[0],
            height=self.occupied_map.boundaries[1],
            occupied_map=merged_obstacle_grid,
            radius=int(self.defender_config.sen_range),
            pos=defender_state,
            max_num_obstacle=self.map_config.max_num_obstacle,
            num_obstacles=self.num_obstacle,
            obstacles=self.occupied_map.obstacle_agent,
            num_beams=self.sensor_config.num_beams,
        )
        obstacle_adj_list = obstacle_adj_list.astype(np.float32)

        target_adj = np.ones(shape=(self.num_defender, 1), dtype=np.float32)
        return obstacle_adj_list, target_adj

    def demon(self):
        next_state = list()
        path_list = list()
        # parallel astar
        defender_state = self.get_state(agent_type='defender').astype(np.int64)
        attacker_state = self.get_state(agent_type='attacker')
        attacker_state = np.stack([attacker_state[0] for _ in range(self.num_defender)], axis=0).astype(np.int64)
        (width, height) = self.occupied_map.boundaries
        self.occupied_map.set_moving_obstacle(pos=defender_state)
        random_indices, paths = parallel_path_planning(defender_state, attacker_state, width, height, obs=np.array(self.occupied_map.ex_obstacles), remove_starts=1)
        for i, idx in enumerate(random_indices):
            defender = self.defender_list[idx]
            defender.path = paths[i]
            if len(defender.path) >= 2:
                if np.linalg.norm((defender.x - defender.path[-1][0], defender.y - defender.path[-1][1])) < self.map_config.resolution * 0.4:
                    defender.path.pop()
            way_point = defender.path[-1]
            next_state.append(defender.demon(way_point))
        for defender in self.defender_list:
            path_list.append(deepcopy(defender.path))
                
        rewards = list()
        adj = self.communicate()
        self.time_step += 1
        # reward 1: Collision and Arriving Reward
        for idx, defender in enumerate(self.defender_list):
            state = next_state[idx]
            if defender.active:
                reward, collision, caught = self.defender_reward(idx, state, next_state)
                rewards.append(reward)
                if caught:
                    self.caught = caught
                    agent_idx = idx
                if collision:
                    defender.dead()
                else:
                    defender.apply_update(state)
            else:
                rewards.append(0)
        # cooperative reward
        if self.caught:
            neighbor_set = adj[agent_idx]
            for neigh, conn in enumerate(neighbor_set):
                if (neigh != agent_idx) and conn:
                    rewards[neigh] = rewards[neigh] + 1
                    
        active = self.get_active()
        done = True if (self.time_step >= self.max_steps) or (sum(active) == 0) or self.caught else False
        info = None
        return rewards, done, info, path_list

def draw_obstacle(ax, x, y, box_width, color):
    left, bottom, wid, hei = (x - box_width / 2, y - box_width / 2, box_width, box_width)
    rect = patches.Rectangle(xy=(left, bottom), width=wid, height=hei, color=color)
    ax.add_patch(rect)


def draw_waypoint(ax, x, y, color):
    path = Path(vertices=np.array([[x - 0.05, y], [x, y - 0.05], [x + 0.05, y], [x, y + 0.05]]))
    waypoint = patches.PathPatch(path, color=color, alpha=0.5)
    ax.add_patch(waypoint)


def update(frame, width, height, obstacles, obstacle_agent, ex_obstacles, defender_pos, attacker_pos, target, path,
           epi_d_e_adj, epi_d_o_adj, epi_d_d_adj, ax):
    """更新每一帧的函数"""
    ax.cla()  # 清除之前的绘图
    ax.set_xlim(0, width - 1)
    ax.set_ylim(0, height - 1)
    ax.set_aspect('equal')
    
    # 绘制障碍物、攻击者、防御者等
    for obstacle in obstacle_agent:
        draw_obstacle(ax, obstacle[0], obstacle[1], 0.8, 'black')
    for obstacle in obstacles:
        draw_obstacle(ax, obstacle[0], obstacle[1], 0.4, 'red')
    for obstacle in ex_obstacles:
        draw_obstacle(ax, obstacle[0], obstacle[1], 0.4, 'grey')

    # 绘制其他实体的位置
    ax.scatter(defender_pos[frame][:, 0], defender_pos[frame][:, 1], color='green', s=0.5)
    ax.scatter(attacker_pos[frame][:, 0], attacker_pos[frame][:, 1], color='red', s=2)
    ax.scatter(target[frame][:, 0], target[frame][:, 1], color='black', s=1)

    # 假设path是一个每一步包含所有路径点的列表
    # for p in path[frame]:
    #     p = np.array(p)
    #     ax.scatter(p[:, 0], p[:, 1], color='cyan', s=0.5, marker='d')
    #     for waypoint in p:
    #         draw_waypoint(ax, waypoint[0], waypoint[1], color='black')  # 绘制路径
    # ax.scatter(target[frame][:, 0], target[frame][:, 1], color='black', s=1)

    # draw communication
    for p1, a in enumerate(epi_d_d_adj[frame]):
        for p2, connected in enumerate(a):
            if (p1 > p2) and connected:
                ax.plot([defender_pos[frame][p1, 0], defender_pos[frame][p2, 0]], [defender_pos[frame][p1, 1], defender_pos[frame][p2, 1]], color='green', linestyle='--', alpha=0.1)

    for p1, a in enumerate(epi_d_o_adj[frame]):
        for p2, connected in enumerate(a):
            if connected:
                ax.plot([defender_pos[frame][p1, 0], obstacle_agent[p2, 0]], [defender_pos[frame][p1, 1], obstacle_agent[p2, 1]], color='grey', linestyle='--', alpha=0.3)

def map_plotting(step, width, height, obstacles, obstacle_agent, ex_obstacles, moving_obstacles, defender_pos, attacker_pos, target, path, 
                 epi_d_e_adj, epi_d_o_adj, epi_d_d_adj, dir):
    defender_pos = np.array(defender_pos)
    attacker_pos = np.array(attacker_pos)

    target = np.array(target)
    fig, ax = plt.subplots(figsize=(5, 5 * 0.9))
    ani = FuncAnimation(fig, update, frames=range(step), fargs=(width, height, obstacles, obstacle_agent, ex_obstacles, defender_pos, attacker_pos, target, path,
                                                                epi_d_e_adj, epi_d_o_adj, epi_d_d_adj, ax), repeat=False)
    ani.save(dir + '.gif', writer='pillow', fps=10, dpi=150)
    

@hydra.main(config_path='./', config_name='pegame.yaml', version_base=None)
def main(cfg):
    cfg = resolution_scaling(cfg=cfg)
    env = PE_Env(cfg)
    t1 = time.time()
    env.reset()
    # Evaluate Matrix
    epi_obs_d = list()
    epi_obs_a = list()
    epi_target = list()
    epi_r = list()
    epi_path = list()
    epi_d_e_adj = list()
    epi_d_o_adj = list()
    epi_d_d_adj = list()
    epi_moving_obstacle = list()
    for i in range(cfg.env.max_steps):
        env.attacker_replan()
        env.attacker_step()
        attacker_pos = env.get_state(agent_type='attacker').tolist()
        epi_obs_a.append(attacker_pos)
        epi_moving_obstacle.append(deepcopy(env.occupied_map.moving_obstacles))

        d_d_adj = env.communicate()
        d_o_adj, d_e_adj = env.sensor()
        
        epi_d_e_adj.append(d_e_adj)
        epi_d_o_adj.append(d_o_adj)
        epi_d_d_adj.append(d_d_adj)

        rewards, done, info, path_list = env.demon()
        defender_pos = env.get_state(agent_type='defender').tolist()
        epi_obs_d.append(defender_pos)
        
        target = env.get_target().tolist()
        epi_target.append(deepcopy(target))
        
        # path = list()
        # for i in range(env.num_attacker):
        #     p = np.array(deepcopy(env.attacker_list[i].path))
        #     path.append(p)
        # epi_path.append(path)
        
    print('finish')
    print(time.time() - t1)
    map_plotting(
        step=cfg.env.max_steps,
        width=env.occupied_map.boundaries[0],
        height=env.occupied_map.boundaries[1],
        obstacles=env.occupied_map.obstacles,
        obstacle_agent=env.occupied_map.obstacle_agent,
        ex_obstacles=env.occupied_map.ex_obstacles,
        moving_obstacles=epi_moving_obstacle,
        defender_pos=epi_obs_d,
        attacker_pos=epi_obs_a,
        target=epi_target,
        path=epi_path,
        epi_d_d_adj=epi_d_d_adj,
        epi_d_e_adj=epi_d_e_adj,
        epi_d_o_adj=epi_d_o_adj,
        dir='unit_test_adj' + str(time.time())
    )

    
if __name__ == '__main__':
    main()
