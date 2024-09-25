from navigation import Navigation_Env, resolution_scaling, map_plotting
import numpy as np
import time, hydra
from copy import deepcopy
from navigation import get_local_sensed_map
from astar import parallel_path_planning


class NonCooperative(Navigation_Env):
    def __init__(self, cfg):
        super().__init__(cfg)

    def store_map(self):
        map_copy = [self.occupied_map.grid_map, self.occupied_map.obstacles, self.target, self.position_list]
        return map_copy
    
    def reset(self, hard_map=None):
        self.group_dis = np.zeros(shape=(self.num_defender, self.num_defender))
        self.time_step = 0
        self.n_episode += 1
        self.collision = False
        self.num_component = None
        self.diameter_reciprocal = None
        # load map
        if hard_map is not None:
            [grid_map, obstacles, target, position_list] = hard_map
            self.init_map(map_info=[grid_map, obstacles])
            self.init_target(target=target)
            self.init_defender(position_list=position_list)
        else:  # generate new map
            self.init_map()  # grid_map, obstacles, boundary_map, boundary_obstacles, obstacle_agents, ex_grid_map, ex_obstacles
            self.init_target()
            self.init_defender()
        self.num_obstacle = len(self.occupied_map.boundary_obstacles)
        self.init_attacker()
        if self.num_protector > 0:
            self.init_protector()
            self.init_random_target()
            self.protector_replan()
        
    def step(self, action):
        next_state = list()
        Total_conn_reward = 0
        Total_wonder_reward = 0
        Total_time_punish = 0
        rewards = list()
        arrival_in_this_round = list()
        collision_in_this_round = list()
        self.time_step += 1
        adj = self.communicate()
        for idx, defender in enumerate(self.defender_list):
            next_state.append(defender.step(action[idx]))
        # reward -1: Wonder Punishment
        wonder_reward = list()
        for idx, defender in enumerate(self.defender_list):
            wonder = defender.traj_map[round(next_state[idx][0]), round(next_state[idx][1])]
            wonder_r = wonder if wonder > 3 else 0
            wonder_reward.append(-0.05 * wonder_r)
        # reward 1: Collision and Arriving Reward
        for idx, defender in enumerate(self.defender_list):
            state = next_state[idx]
            if defender.active:
                reward, collision, caught = self.defender_reward(idx, state, next_state, self.target[0])
                rewards.append(reward)
                if caught:
                    arrival_in_this_round.append(idx)
                    defender.arrived = True
                else:
                    defender.arrived = False
                if collision:
                    collision_in_this_round.append(idx)
                    defender.dead()
                else:
                    defender.apply_update(state)
            else:
                rewards.append(0)
        # reward 2: Cooperative Reward 
        if len(arrival_in_this_round) > 0:
            for idx in arrival_in_this_round:
                neighbor_set = adj[idx]
                for neigh, conn in enumerate(neighbor_set):
                    if (neigh != idx) and conn:
                        rewards[neigh] = rewards[neigh] + 0.05
        # reward 2.5: Cooperative Collision Reward
        if len(collision_in_this_round) > 0:
            for idx in collision_in_this_round:
                neighbor_set = adj[idx]
                for neigh, conn in enumerate(neighbor_set):
                    if (neigh != idx) and conn:
                        rewards[neigh] = rewards[neigh] - 100 * np.clip(self.func(self.group_dis[idx, neigh]), 0, 1)
        # reward 4: Wonder Punishment
        for idx, defender in enumerate(self.defender_list):
            if defender.active and rewards[idx] <= 0:
                rewards[idx] = rewards[idx] + wonder_reward[idx]
                Total_wonder_reward += wonder_reward[idx]

        active = self.get_active()
        done = True if (self.time_step >= self.max_steps) or (sum(active) == 0) else False
        info = [Total_conn_reward, Total_wonder_reward, Total_time_punish]
        return rewards, done, info

    def protector_replan(self, idx=None):
        if idx is None:
            idx_list = [i for i in range(self.num_protector)]
        else:
            idx_list = [idx]
        protector_state = np.round(self.get_state(agent_type='protector')).astype(np.int64)
        random_target = np.array(self.random_target, dtype=np.int64)
        (width, height) = self.occupied_map.boundaries
        random_indices, paths = parallel_path_planning(protector_state[idx_list], random_target[idx_list], width, height, obs=np.array(self.occupied_map.ex_obstacles))
        
        if idx is None:
            for i, path in enumerate(paths):
                self.protector_list[random_indices[i]].path = path
        else:
            self.protector_list[idx].path = paths[0]

    def protector_step(self):
        for idx, protector in enumerate(self.protector_list):
            if len(protector.path) >= 2:
                if np.linalg.norm((protector.x - protector.path[-1][0], protector.y - protector.path[-1][1])) < self.map_config.resolution:
                    protector.path.pop()
            way_point = protector.path[-1]
            [x, y, vx, vy, theta] = protector.step(way_point)
            if self.occupied_map.is_unoccupied((x, y), map=self.occupied_map.grid_map):
                protector.apply_update([x, y, vx, vy, theta])
            if np.linalg.norm((self.random_target[idx][0] - x, self.random_target[idx][1] - y)) <= self.protector_config.collision_radius:
                self.init_random_target(idx=idx)
                self.protector_replan(idx=idx)
    
    def defender_reward(self, idx, state, next_state, target):
        reward = 0
        collision = False
        caught = False
        inner_collision = self.collision_detection(state, obstacle_type='defender', next_state=next_state, idx=idx)
        reward -= (sum(inner_collision) - 1) * 100
        if reward < 0:
            collision = True
            return reward, collision, caught
        
        protector_collision = self.collision_detection(state, obstacle_type='protector')
        reward -= (sum(protector_collision)) * 100
        if reward < 0:
            collision = True
            return reward, collision, caught

        obstacle_collision = self.collision_detection(state, obstacle_type='obstacle')
        reward -= obstacle_collision * 100
        if reward < 0:
            collision = True
            return reward, collision, caught
        
        is_collision = self.collision_detection(state, obstacle_type='target', target=target)
        reward = is_collision * 0.1
        if is_collision:
            caught = True
        return reward, collision, caught
    
    def all_arrived(self):
        arrived = list()
        for defender in self.defender_list:
            if defender.active:
                arrived.append(0 if defender.arrived else 1)
            all_arrived = not bool(sum(arrived))
        return all_arrived
    
    def sensor(self):
        protector_state = np.round(self.get_state(agent_type='protector')).astype(np.int64).tolist()
        defender_state = np.round(self.get_state(agent_type='defender')).astype(np.int64)
        merged_obstacle_grid = self.merge_static_dynamic_obstacles(protector_state=protector_state, defender_state=defender_state)
        
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

        for i, defender in enumerate(self.defender_list):
            if defender.obstacle_adj is not None:
                defender.obstacle_adj = [1 if x == 1 or y == 1 else 0 for x, y in zip(obstacle_adj_list[i], defender.obstacle_adj)]
            else:
                defender.obstacle_adj = obstacle_adj_list[i]
            
            obstacle_adj_list[i] = defender.obstacle_adj

        obstacle_adj_list = obstacle_adj_list.astype(np.float32)
        target_adj = np.ones(shape=(self.num_defender, 1), dtype=np.float32)
        return obstacle_adj_list, target_adj
                        
          
@hydra.main(config_path='./', config_name='config.yaml', version_base=None)          
def main(cfg):
    cfg = resolution_scaling(cfg=cfg)
    env = NonCooperative(cfg)
    start_time = time.time()
    env.reset()
    # Evaluate Matrix
    epi_obs_d = list()
    epi_obs_a = list()
    epi_obs_p = list()
    epi_target = list()
    epi_random_target = list()
    epi_r = list()
    epi_path = list()
    epi_d_e_adj = list()
    epi_d_o_adj = list()
    epi_d_p_adj = list()
    epi_d_d_adj = list()
    epi_moving_obstacle = list()
    for i in range(cfg.env.max_steps):
        env.protector_step()
        protector_pos = env.get_state(agent_type='protector').tolist()
        epi_obs_p.append(protector_pos)
        epi_random_target.append(deepcopy(env.random_target))

        # env.attacker_replan()
        # env.attacker_step()
        attacker_pos = env.get_state(agent_type='attacker').tolist()
        epi_obs_a.append(attacker_pos)
        epi_moving_obstacle.append(deepcopy(env.occupied_map.moving_obstacles))

        d_d_adj = env.communicate()
        d_p_adj = env.communicate_with_protector()
        d_o_adj, d_e_adj = env.sensor()
        
        epi_d_e_adj.append(d_e_adj)
        epi_d_o_adj.append(d_o_adj)
        epi_d_d_adj.append(d_d_adj)
        epi_d_p_adj.append(d_p_adj)

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
    print(time.time() - start_time)
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
        protector_pos=epi_obs_p,
        target=epi_target,
        path=epi_path,
        epi_d_d_adj=epi_d_d_adj,
        epi_d_e_adj=epi_d_e_adj,
        epi_d_p_adj=epi_d_p_adj,
        epi_d_o_adj=epi_d_o_adj,
        dir='unit_test_non_cooperative_env' + str(time.time())
    )

    
if __name__ == '__main__':
    main()
