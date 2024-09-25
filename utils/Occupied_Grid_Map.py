import numpy as np
from skimage.segmentation import find_boundaries
import random
from numba import njit, prange
from copy import deepcopy
import time


OBSTACLE = 1
UNOCCUPIED = 0
@njit()
def _extend_obstacle(ex_grid_map, boundary_obstacles, width, height, extend_dis):
    ex_obstacles = list()
    for obstacle in boundary_obstacles:  # 只遍历边界即可
        (x, y) = obstacle
        for x_ in range(x - extend_dis, x + extend_dis + 1):
            for y_ in range(y - extend_dis, y + extend_dis + 1):
                if 0 < x_ < width and 0 < y_ < height and ex_grid_map[x_, y_] == UNOCCUPIED:  # 判断是否在范围内，并且没有被占据
                    ex_grid_map[x_, y_] = OBSTACLE  # 得到一个拓展的地图，并且避免重复添加
                    ex_obstacles.append((x_, y_))
    return ex_grid_map, ex_obstacles


@njit()
def _extend_moving_obstacles(ex_grid_map, moving_obstacles, width, height, extend_dis):
    ex_moving_obstacles = list()
    for obstacle in moving_obstacles:  # 只遍历边界即可
        (x, y) = obstacle
        for x_ in range(x - extend_dis, x + extend_dis + 1):  # extend
            for y_ in range(y - extend_dis, y + extend_dis + 1):  # extend
                if 0 < x_ < width and 0 < y_ < height and ex_grid_map[x_, y_] == UNOCCUPIED:  # 判断是否在范围内，并且没有被占据
                    point = (x_, y_)
                    if point not in moving_obstacles:
                        if point not in ex_moving_obstacles:
                            ex_moving_obstacles.append(point)
    return ex_moving_obstacles


@njit()    
def _get_raser_map(boundary_map, obstacle_agent, num_beams, radius, width, height): # type: ignore
    num_obstacles = len(obstacle_agent)
    hash_map = np.zeros((width, height, num_obstacles))
    blocks = width * height
    beam_angles = np.array([beam * 2 * np.pi / num_beams for beam in range(num_beams)])
    beam_dirs_x = np.cos(beam_angles)
    beam_dirs_y = np.sin(beam_angles)
    for idx_1 in prange(blocks):
        x = idx_1 // height
        y = idx_1 % height
        for beam in range(num_beams):
            for beam_range in range(radius):
                beam_current_x = x + beam_range * beam_dirs_x[beam]
                beam_current_y = y + beam_range * beam_dirs_y[beam]
                if beam_current_x < 0 or beam_current_x >= width or beam_current_y < 0 or beam_current_y >= height:
                    break
                beam_current_x = round(beam_current_x)
                beam_current_y = round(beam_current_y)
                if boundary_map[beam_current_x, beam_current_y] == 1:
                    for idx_2 in range(num_obstacles):
                        if obstacle_agent[idx_2][0] == beam_current_x and obstacle_agent[idx_2][1] == beam_current_y:
                            hash_map[x, y, idx_2] = 1
                            break
                    break
    return hash_map


class OccupiedGridMap:
    def __init__(self, map_config, new_grid=None, obstacles = None, exploration_setting='8N' ) -> None:
        self.resolution = map_config.resolution
        self.boundaries = map_config.map_size
        if new_grid == None:
            self.grid_map = np.zeros(shape=self.boundaries)
        else:
            self.grid_map = new_grid
        if obstacles is None:
            self.obstacles = list()
        else:
            self.obstacles = obstacles
        # obstacles
        self.exploration_setting = exploration_setting
        self.map_config = map_config
        self.moving_obstacles = list()
    # def add_blocker_type(self, center_point: tuple, data: tuple):
    #     '''
    #     param:
    #     center_point -> (int, int) the center point of the obstacle
    #     in 2D case: 
    #        rectangle, data ->(x, y)
    #     '''
    #     x_bound = (int(-data[0] / 2), int(data[0] / 2))
    #     y_bound = (int(-data[1] / 2), int(data[1] / 2))
    #     for x in range(x_bound[0], x_bound[1]):
    #         for y in range(y_bound[0], y_bound[1]):
    #             (x_, y_) = self.round_up((x + center_point[0], y + center_point[1]))
    #             if self.in_bound((x_, y_)):
    #                 self.set_obstacle((x_, y_))

    def add_blocker_type(self, center_point: tuple, size: tuple, blocker_type: str = "rectangle", orientation: str = "up"):
        if blocker_type == "rectangle":
            x_bound = (int(-size[0] / 2), int(size[0] / 2))
            y_bound = (int(-size[1] / 2), int(size[1] / 2))
            for x in range(x_bound[0], x_bound[1]):
                for y in range(y_bound[0], y_bound[1]):
                    (x_, y_) = self.round_up((x + center_point[0], y + center_point[1]))
                    if self.in_bound((x_, y_)):
                        self.set_obstacle((x_, y_))
        elif blocker_type == "u_shaped":
            # Define U-shaped obstacle based on its orientation
            outer_bounds = size  # Assuming data contains outer dimensions of U shape
            width = outer_bounds[0] // 2
            height = outer_bounds[1] // 2
            if orientation == "up" or orientation == "down":
                up_center_point = [center_point[0], center_point[1] + height / 2]
                down_center_point = [center_point[0], center_point[1] - height / 2]
                for dy in range(0, height):
                    self.set_obstacle_in_grid(up_center_point, -width, dy)
                    self.set_obstacle_in_grid(up_center_point, width, dy)
                    
                for dy in range(-height, 0):
                    self.set_obstacle_in_grid(down_center_point, -width, dy)
                    self.set_obstacle_in_grid(down_center_point, width, dy)
                    
                for dx in range(-width, width + 1):
                    self.set_obstacle_in_grid(up_center_point, dx, -height)
                    
                for dx in range(-width, width + 1):
                    self.set_obstacle_in_grid(down_center_point, dx, height)
                    
            elif orientation == "left" or orientation == "right":
                left_center_point = [center_point[0] - height / 2, center_point[1]]
                right_center_point = [center_point[0] + height / 2, center_point[1]]
                for dx in range(-height, 0):
                    self.set_obstacle_in_grid(left_center_point, dx, -width)
                    self.set_obstacle_in_grid(left_center_point, dx, width)
                    
                for dx in range(0, height):
                    self.set_obstacle_in_grid(right_center_point, dx, -width)
                    self.set_obstacle_in_grid(right_center_point, dx, width)
                    
                for dy in range(-width, width + 1):
                    self.set_obstacle_in_grid(left_center_point, height, dy)

                for dy in range(-width, width + 1):
                    self.set_obstacle_in_grid(right_center_point, -height, dy)

    def set_obstacle_in_grid(self, center_point, dx, dy):
        # Adjust obstacle placement based on orientation if needed
        x_, y_ = self.round_up((center_point[0] + dx, center_point[1] + dy))
        if self.in_bound((x_, y_)):
            self.set_obstacle((x_, y_))

    # convert the float point into the int point
    def initailize_obstacle(self):
        # for _ in range(num):
        #     center_point = np.random.normal(center, variance, 2)
        #     self.add_blocker_type(center_point=center_point,data=size)
        center = self.map_config.center
        num_u_shaped = self.map_config.num_u_shaped
        variance_u_shaped = self.map_config.variance_u_shaped
        u_shape = self.map_config.u_shape
        num_blocks = self.map_config.num_blocks
        block_shape = self.map_config.block_shape
        for _ in range(num_u_shaped):
            center_point = np.random.normal(center, variance_u_shaped, 2)
            orientation = random.choice(["up", "down", "left", "right"])  # Random orientation for U-shaped obstacles
            self.add_blocker_type(center_point=center_point, size=u_shape, blocker_type='u_shaped', orientation=orientation)
        for _ in range(num_blocks):
            center_point = np.random.uniform((1, 1), self.boundaries, 2)
            self.add_blocker_type(center_point=center_point, size=block_shape, blocker_type="rectangle")

        for x in range(self.boundaries[0]):
            self.set_obstacle([x, 0])
            self.set_obstacle([x, self.boundaries[1] - 1])
            
        for y in range(self.boundaries[1]):
            self.set_obstacle([0, y])
            self.set_obstacle([self.boundaries[0] - 1, y])
            
    def round_up(self, pos: tuple) -> tuple:
        return (round(pos[0]), round(pos[1]))
        
    def in_bound(self, pos: tuple):
        (x, y) = pos
        return x < self.boundaries[0] and x > 0 and y < self.boundaries[1] and y > 0

    def is_unoccupied(self, pos, map: np.ndarray) -> bool:
        """
        :param pos: cell position we wish to check
        :param map: 
        :return: True if cell is occupied with obstacle, False else
        """
        (x, y) = self.round_up(pos)
        if self.in_bound((x, y)):
            if map[x][y] == UNOCCUPIED:
                return True
        return False

    def set_moving_obstacle(self, pos: list):
        self.moving_obstacles = list()
        for (x, y, *_) in pos:
            point = self.round_up((x, y))
            if self.in_bound(pos=point):
                if point not in self.moving_obstacles:
                    self.moving_obstacles.append(point)

    # def extend_moving_obstacles(self, extend_dis):
    #     self.ex_moving_obstacles = list()
    #     for (x, y) in self.moving_obstacles:
    #         for x_ in range(x - extend_dis, x + extend_dis + 1):  # extend
    #             for y_ in range(y - extend_dis, y + extend_dis + 1):  # extend
    #                 if self.in_bound(pos=(x_, y_)):  # whether is bounded
    #                     point = (x_, y_)
    #                     if point not in self.moving_obstacles:
    #                         if point not in self.ex_moving_obstacles:
    #                             self.ex_moving_obstacles.append(point)

    def extend_moving_obstacles(self, extend_dis):
        self.ex_moving_obstacles = _extend_moving_obstacles(self.ex_grid_map, self.moving_obstacles, self.boundaries[0], self.boundaries[1], extend_dis)
        
    def set_obstacle(self, pos):
        """
        :param pos: cell position we wish to set obstacle
        :return: None
        """
        (x, y) = pos
        self.grid_map[x][y] = OBSTACLE
        if (x, y) not in self.obstacles:
            self.obstacles.append((x, y))

    # def set_extended_obstacle(self, pos):
    #     """
    #     :param pos: cell position we wish to set obstacle
    #     :return: None
    #     """
    #     point = self.round_up(pos)
    #     if point not in self.obstacles:
    #         if point not in self.ex_obstacles:
    #             self.ex_obstacles.append(point)

    def get_boundary_map(self, max_num):
        self.boundary_map = find_boundaries(self.grid_map, mode='inner')
        self.boundary_obstacles = np.argwhere(self.boundary_map == 1)

        # 直接使用 NumPy 的操作来添加列，而不是使用 deepcopy 和循环
        zeros = np.zeros((self.boundary_obstacles.shape[0], 2), dtype=np.float32)
        obstacle_agent = np.hstack((self.boundary_obstacles, zeros)).astype(np.float32)

        # 预分配内存并设置值，避免使用循环和列表追加
        if max_num > len(obstacle_agent):
            padding = np.zeros((max_num - len(obstacle_agent), obstacle_agent.shape[1]), dtype=np.float32)
            obstacle_agent = np.vstack((obstacle_agent, padding))
        
        self.obstacle_agent = obstacle_agent
        self.boundary_obstacles = [tuple(row) for row in self.boundary_obstacles]

    def extend_obstacles(self, extend_dis: int):
        ex_grid_map, ex_obstacles = _extend_obstacle(
            ex_grid_map=deepcopy(self.grid_map), 
            boundary_obstacles=self.boundary_obstacles, 
            width=self.boundaries[0],
            height=self.boundaries[1],
            extend_dis=extend_dis
        )
        self.ex_obstacles = ex_obstacles
        self.ex_grid_map = ex_grid_map

    def get_raser_map(self, num_beams, radius):
        hash_map = _get_raser_map(
            boundary_map=self.boundary_map,
            obstacle_agent=self.obstacle_agent,
            num_beams=num_beams,
            radius=radius,
            width=self.boundaries[0],
            height=self.boundaries[1],
        )
        self.raser_map = hash_map.tolist()

    def local_observation(self, global_position: tuple, view_range: int) -> dict:
        """
        :param global_position: position of robot in the global map frame
        :param view_range: how far ahead we should look
        :return: dictionary of new observations
        """
        pos_obstacle = {}
        point = self.round_up(global_position)
        for x_ in range(point[0] - view_range, point[0] + view_range):
            for y_ in range(point[1] - view_range, point[1] + view_range):
                if (self.in_bound((x_, y_))) and (np.linalg.norm([point[0] - x_, point[1] - y_]) <= view_range):
                    pos_obstacle[(x_, y_)] = OBSTACLE if not self.is_unoccupied(pos=(x_, y_), map=self.grid_map) else UNOCCUPIED
                            
        return pos_obstacle
    

if __name__ == '__main__':
    def generate_points(num_points, obs, max_range=(100, 100)):
        points = []
        while len(points) < num_points:
            point = (np.random.randint(0, max_range[0]), np.random.randint(0, max_range[1]))
            if point not in obs and point not in points:  # 确保点不在障碍物中且没有重复
                points.append(point)
        return points
    num = 5
    center = (50, 50)
    variance = 25
    occupied_map = OccupiedGridMap(resolution=0.1, boundaries=(100, 100))
    occupied_map.initailize_obstacle(num=num, center=center, variance=variance, shape=(10, 10))
    # grid_map_backup = deepcopy(occupied_map.grid_map)
    boundary_map = occupied_map.get_boundary_map(50000)
    occupied_map.extend_obstacles(extend_dis=1)
    # occupied_map.extended_obstacles2(grid_map_backup, boundary_map, extend_dis=1)
    occupied_map.get_raser_map(num_beams=36, radius=8)
    defender_state = generate_points(20, obs=occupied_map.ex_obstacles + occupied_map.obstacles, max_range=(100, 100))

    occupied_map.set_moving_obstacle(pos=defender_state)
    t1 = time.time()
    occupied_map.extend_moving_obstacles(extend_dis=2)
    print(time.time() - t1)
    t1 = time.time()
    occupied_map.extend_moving_obstacles(extend_dis=2)
    print(time.time() - t1)