import numpy as np
import math
from omegaconf import DictConfig


class Agent:
    def __init__(
        self, 
        step_size: float, tau: float,
        x: float, y: float,
        vmax: float, 
        sen_range: int, comm_range: int,
        action_dim: int = 4,
        discrete: bool = True
    ):
        """_summary_

        Args:
            idx (int): agent id
            step_size (float): simulation time step
            tau (float): time-delay in first-order dynamic
            DOF: the dimension of freedom
            x (float): x position
            y (float): y position
            vx (float): velocity in x-axis
            vy (float): velocity in y-axis
            theta (float): the angle between velocity vector and x-axis
            v_max (float): the max velocity
            sen_range (int): sensor range
            comm_range (int): communication range
        """
        self.step_size = step_size
        self.tau = tau
        
        self.x = x
        self.y = y
        
        self.vmax = vmax
        self.theta = np.random.rand() * 2 * np.pi
        v = self.vmax * np.random.rand()
        # self.vx = v * np.cos(self.theta)
        # self.vy = v * np.sin(self.theta)
        self.vx = 0
        self.vy = 0
        self.sen_range = sen_range
        self.comm_range = comm_range

        self.active = True
        self.arrived = False
        
        self.discrete = discrete
        self.action_dim = action_dim
        if self.discrete:
            self.actions_mat = []
            self.actions_mat.append((0, 0))  # (dx, dy) = (0, 0) 表示静止不动
            for i in range(1, action_dim + 1):
                angle = 360 * (i - 1) / action_dim  # 计算角度
                dx = math.cos(math.radians(angle))
                dy = math.sin(math.radians(angle))
                self.actions_mat.append((dx * self.vmax, dy * self.vmax))

    def step(self, action: int):
        """Transform discrete action to desired velocity
        Args:
            action (int): an int belong to [0, 1, 2, ..., 8] - 2D
        """
        if self.discrete:
            desired_velocity = self.actions_mat[action]
        else:
            desired_velocity = action * self.vmax

        next_state = self.dynamic(u=desired_velocity)
        return next_state
    
    def demon(self, theta: float):
        next_state = self.dynamic(u=(math.cos(theta) * self.vmax, math.sin(theta) * self.vmax))
        return next_state
        
    def apply_update(self, next_state):
        if self.active:
            self.x, self.y, self.vx, self.vy, self.theta = next_state

    def dynamic(self, u: float = 0):
        """The dynamic of the agent is considered as a 1-order system with 2/3 DOF.
        The input dimension is the same as the state dimension.

        Args:
            u (float): The desired velocity.
            # DOF (int, optional): Degree of freedom. Defaults to 2.
        """
        tau_x = self.tau * np.abs(np.cos(self.theta)) + 0.5
        tau_y = self.tau * np.abs(np.sin(self.theta)) + 0.5
        # tau_x = 0.2
        # tau_y = 0.2
        k1vx = (u[0] - self.vx) / tau_x
        k2vx = (u[0] - (self.vx + self.step_size * k1vx / 2)) / tau_x
        k3vx = (u[0] - (self.vx + self.step_size * k2vx / 2)) / tau_x
        k4vx = (u[0] - (self.vx + self.step_size * k3vx)) / tau_x
        vx = self.vx + (k1vx + 2 * k2vx + 2 * k3vx + k4vx) * self.step_size / 6
        k1vy = (u[1] - self.vy) / tau_y
        k2vy = (u[1] - (self.vy + self.step_size * k1vy / 2)) / tau_y
        k3vy = (u[1] - (self.vy + self.step_size * k2vy / 2)) / tau_y
        k4vy = (u[1] - (self.vy + self.step_size * k3vy)) / tau_y
        vy = self.vy + (k1vy + 2 * k2vy + 2 * k3vy + k4vy) * self.step_size / 6

        v = np.linalg.norm([vx, vy])
        v = np.clip(v, 0, self.vmax)
        
        if math.isclose(v, 0, rel_tol=1e-3):
            theta = self.theta
        else:
            theta = math.atan2(vy, vx)
        
        x = self.x + vx * self.step_size
        y = self.y + vy * self.step_size
        
        return [x, y, vx, vy, theta]

    def dead(self):
        self.active = False
        self.x = -1
        self.y = -1
        self.vx = 0
        self.vy = 0
        self.theta = 0
        
    def stop(self):
        self.vx = 0
        self.vy = 0


class Pursuer(Agent):
    def __init__(
        self, 
        x: float, y: float,
        defender_cfg: DictConfig,
        map_size=None
    ):
        super().__init__(
            x=x, y=y, 
            vmax=defender_cfg.vmax, step_size=defender_cfg.step_size, tau=defender_cfg.tau,
            sen_range=defender_cfg.sen_range, comm_range=defender_cfg.comm_range,
            action_dim=defender_cfg.action_dim,
            discrete=defender_cfg.discrete
        )
        
        self.obstacle_adj = None
        self.arrived = False
        if map_size is not None:
            self.traj_map = np.zeros(shape=map_size)
    
    def apply_update(self, next_state):
        if self.active:
            self.x, self.y, self.vx, self.vy, self.theta = next_state
        # for x in range(max(round(self.x) - 1, 0), min(round(self.x) + 2, self.traj_map.shape[0])):
        #     for y in range(max(round(self.y) - 1, 0), min(round(self.y) + 2, self.traj_map.shape[1])):
        #         self.traj_map[x, y] += 1
        self.traj_map[round(self.x), round(self.y)] += 1

    def demon(self, waypoint):
        radius = np.linalg.norm([waypoint[0] - self.x, waypoint[1] - self.y])
        if math.isclose(radius, 0.0, abs_tol=0.01):
            u = [0, 0]
        else:
            phi = self.waypoint2phi(waypoint)
            u = [np.cos(phi) * self.vmax, np.sin(phi) * self.vmax]
        next_state = self.dynamic(u=u)
        return next_state
    
    def waypoint2phi(self, way_point):
        """
        :param way_point:
        :return phi: angle belong to (-pi, pi)
        """
        radius = np.linalg.norm([way_point[0] - self.x, way_point[1] - self.y])
        if math.isclose(radius, 0.0, abs_tol=0.01):
            phi = 0
        else:
            if np.sign(way_point[1] - self.y) == 0:
                phi = 0 if way_point[0] - self.x >= 0 else np.pi
            else:
                phi = np.sign(way_point[1] - self.y) * np.arccos((way_point[0] - self.x) / (radius + 1e-3))
        return phi


class Evader(Agent):
    def __init__(
        self, 
        x: float, y: float, 
        attacker_config,
    ):
        super().__init__(
            x=x, y=y, 
            vmax=attacker_config.vmax, step_size=attacker_config.step_size, tau=attacker_config.tau,
            sen_range=attacker_config.sen_range, comm_range=attacker_config.comm_range
        )
        self.is_pursuer = False
        self.path = list()
        
    def step(self, waypoint):
        radius = np.linalg.norm([waypoint[0] - self.x, waypoint[1] - self.y])
        if math.isclose(radius, 0.0, abs_tol=0.01):
            u = [0, 0]
        else:
            phi = self.waypoint2phi(waypoint)
            u = [np.cos(phi) * self.vmax, np.sin(phi) * self.vmax]
        next_state = self.dynamic(u=u)
        return next_state
        
    def waypoint2phi(self, way_point):
        """
        :param way_point:
        :return phi: angle belong to (-pi, pi)
        """
        radius = np.linalg.norm([way_point[0] - self.x, way_point[1] - self.y])
        if math.isclose(radius, 0.0, abs_tol=0.01):
            phi = 0
        else:
            if np.sign(way_point[1] - self.y) == 0:
                phi = 0 if way_point[0] - self.x >= 0 else np.pi
            else:
                phi = np.sign(way_point[1] - self.y) * np.arccos((way_point[0] - self.x) / (radius + 1e-3))
        return phi


class Protector(Agent):
    def __init__(
        self, 
        x: float, y: float, 
        protector_config,
    ):
        super().__init__(
            x=x, y=y, 
            vmax=protector_config.vmax, step_size=protector_config.step_size, tau=protector_config.tau,
            sen_range=protector_config.sen_range, comm_range=protector_config.comm_range
        )
        self.is_pursuer = False
        self.path = list()
        
    def step(self, waypoint):
        radius = np.linalg.norm([waypoint[0] - self.x, waypoint[1] - self.y])
        if math.isclose(radius, 0.0, abs_tol=0.01):
            u = [0, 0]
        else:
            phi = self.waypoint2phi(waypoint)
            u = [np.cos(phi) * self.vmax, np.sin(phi) * self.vmax]
        next_state = self.dynamic(u=u)
        return next_state
        
    def waypoint2phi(self, way_point):
        """
        :param way_point:
        :return phi: angle belong to (-pi, pi)
        """
        radius = np.linalg.norm([way_point[0] - self.x, way_point[1] - self.y])
        if math.isclose(radius, 0.0, abs_tol=0.01):
            phi = 0
        else:
            if np.sign(way_point[1] - self.y) == 0:
                phi = 0 if way_point[0] - self.x >= 0 else np.pi
            else:
                phi = np.sign(way_point[1] - self.y) * np.arccos((way_point[0] - self.x) / (radius + 1e-3))
        return phi

