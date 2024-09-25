import math
import heapq
import time
import numba
from numba import typed
import numpy as np


@numba.njit
def remove_element_from_array(arr, key):
    mask = np.ones(len(arr), dtype=np.bool_)
    for i in range(len(arr)):
        if arr[i][0] == key[0] and arr[i][1] == key[1]:
            mask[i] = False
    return arr[mask]


@numba.njit()
def element_in_array(arr, key):
    for x in arr:
        if x[0] == key[0] and x[1] == key[1]:
            return True
    return False


@numba.njit()
def heuristic(s, goal, heuristic_type="manhattan"):
    if heuristic_type == "manhattan":
        return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
    else:  # Euclidean distance as fallback
        return math.sqrt((goal[0] - s[0])**2 + (goal[1] - s[1])**2)


@numba.njit()
def is_collision(s, width, height, obs):
    # Check if a point is out of bounds or is an obstacle
    if s[0] < 0 or s[0] >= width or s[1] < 0 or s[1] >= height:
        return True
    for ob in obs:  # Linear search; for large obstacle sets, consider spatial indexing
        if s[0] == ob[0] and s[1] == ob[1]:
            return True
    return False


@numba.njit()
def astar_pathfinding_numba(start, goal, width, height, obs, remove_start=0, heuristic_type="manhattan"):
    u_set = np.array([(-1, 0), (-1, 1), (0, 1), (1, 1),
                      (1, 0), (1, -1), (0, -1), (-1, -1)])
    parent = dict()
    open_set = [(0.0, start)]  # 初始化时使用0.0作为浮点数
    g = dict()
    g[start] = 0.0  # 明确指定为浮点数
    if remove_start:
        if element_in_array(obs, start):
            obs = remove_element_from_array(obs, start)
        if element_in_array(obs, goal):
            obs = remove_element_from_array(obs, goal)
        # if start in obs:
        #     obs.remove(start)
        # if goal in obs:
        #     obs.remove(goal)
    while open_set:
        _, current = open_set.pop(0)

        if current == goal:
            path = []
            count = 0
            while current != start:
                path.append(current)
                count += 1
                current = parent[current]  # 假设current始终在parent中
            if count == 0:
                path.append(start)
            # path.reverse()
            return np.array(path, dtype=np.int64)  # 返回路径

        for i in range(u_set.shape[0]):
            neighbor = (current[0] + u_set[i, 0], current[1] + u_set[i, 1])
            if is_collision(neighbor, width, height, obs):
                continue
            new_cost = g[current] + heuristic(current, neighbor, "euclidean")
            # 这里不再需要修改，因为g和new_cost现在都是浮点数
            if neighbor not in g or new_cost < g.get(neighbor, math.inf):
                g[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal, heuristic_type)
                open_set.append((priority, neighbor))
                parent[neighbor] = current
        open_set.sort(key=lambda x: x[0])

    return np.array([start], dtype=np.int64)  # 如果没有找到路径，返回起点


@numba.njit()
def parallel_path_planning(starts, goals, width, height, obs, remove_starts=0, heuristic_type="manhattan"):
    num_robots = len(starts)
    paths = typed.List()
    ids = typed.List()
    for i in range(num_robots):
        start = (starts[i][0], starts[i][1])
        goal = (goals[i][0], goals[i][1])
        path = astar_pathfinding_numba(start, goal, width, height, obs, remove_starts, heuristic_type)
        
        path_tuples = [(path[j, 0], path[j, 1]) for j in range(path.shape[0])]
        paths.append(path_tuples)
        ids.append(i)  # 存储编号以便之后排序

    return ids, paths


class AStar_2D:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, width, height, heuristic_type="manhattan"):
        self.heuristic_type = heuristic_type

        self.u_set = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                      (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.action_set = {(-1,0) : 0, (-1,1) : 1, (0, 1) : 2, (1,1) : 3,
                           (1,0) : 4, (1,-1) : 5, (0, -1) : 6, (-1, -1) : 7}
        self.s_start = None
        self.s_goal = None
        self.obs = None  # position of obstacles
        self.OPEN = None  # priority queue / OPEN set
        self.CLOSED = None  # CLOSED set / VISITED order
        self.PARENT = None  # recorded parent
        self.g = None  # cost to come
        # self.path = None
        self.width = width
        self.height = height

    def searching(self, s_start: tuple, s_goal: tuple, obs):
        """
        A_star Searching.
        :return: path, visited order
        """
        self.s_start = s_start

        self.s_goal = s_goal
        self.obs = obs  # position of obstacles
        if self.s_start in self.obs:
            self.obs.remove(self.s_start)
        if self.s_goal in self.obs:
            self.obs.remove(self.s_goal)
        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        if s_goal in obs:
            path = [s_start]

        else:
            while self.OPEN:
                _, s = heapq.heappop(self.OPEN)
                self.CLOSED.append(s)
                if s == self.s_goal:  # stop condition
                    break

                for s_n in self.get_neighbor(s):
                    new_cost = self.g[s] + self.cost(s, s_n)

                    if s_n not in self.g:
                        self.g[s_n] = math.inf

                    if new_cost < self.g[s_n]:  # conditions for updating Cost
                        self.g[s_n] = new_cost
                        self.PARENT[s_n] = s
                        heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

            try:
                path = self.extract_path(self.PARENT)
            except:
                # print('No path found')
                path = [s_start]

        return path

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """
        if s_start in self.obs:
            return True

        if s_start[0] < 0 or s_start[0] > self.width or s_start[1] < 0 or s_start[1] > self.height:
            return True
            
        if s_end[0] < 0 or s_end[0] > self.width or s_end[1] < 0 or s_end[1] > self.height:
            return True
            
        if s_end in self.obs:
            return True
        return False

    def f_value(self, s, e=2.5):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :param e: hyperparameter
        :return: f
        """

        return self.g[s] + e * self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        
        s = self.s_goal
        path = [s]
        while True:
            p = PARENT[s]
            key = (p[0] - s[0], p[1] - s[1])
            path.append(p)
            s = p
            if s == self.s_start:
                break
        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


if __name__ == '__main__':
    # 示例用法
    width = 50
    height = 50
    original_obs = [((3, 3), (3, 4), (4, 3), (4, 4), (25, 25), (26, 26), (27, 27), (28, 28), (31, 31))]

    # 将原始障碍物转换为列表形式以便添加更多障碍物
    obs_list = list(original_obs[0])

    # 为了增加障碍物，我们在50x50的范围内随机选择位置
    # 确保总障碍物数量合适，这里我们尝试添加至总数达到100
    np.random.seed(0)  # 为了可复现性设置随机种子
    # while len(obs_list) < 1500:
    #     new_obs = (np.random.randint(0, 50), np.random.randint(0, 50))
    #     if new_obs not in obs_list:  # 避免重复位置
    #         obs_list.append(new_obs)

    # 转换回numpy数组格式
    extended_obs = np.array([tuple(obs_list)])[0]
    # print(type(extended_obs))
    # 假设我们有num_points个机器人需要规划路径
    def generate_points(num_points, obs, max_range=(50, 50)):
        points = []
        while len(points) < num_points:
            point = (np.random.randint(0, max_range[0]), np.random.randint(0, max_range[1]))
            if point not in obs and point not in points:  # 确保点不在障碍物中且没有重复
                points.append(point)
        return points

    # 起点和终点的数量
    num_starts_goals = 50

    # 生成起点和终点
    starts = generate_points(num_starts_goals, obs_list)
    goals = generate_points(num_starts_goals, obs_list + starts)  # 避免与起点和障碍物重合

    t0 = time.time()
    ids, paths = parallel_path_planning(np.array(starts), np.array(goals), width, height, extended_obs, heuristic_type="manhattan")
    print(time.time() - t0)

    # 调用并行路径规划函数
    t1 = time.time()
    ids, paths = parallel_path_planning(np.array(starts), np.array(goals), width, height, extended_obs, heuristic_type="manhattan")
    print(time.time() - t1)
    print(ids)
    

    astar = AStar_2D(width=width, height=height)
    t2 = time.time()
    for i in range(num_starts_goals):
        path_list = astar.searching(s_start=starts[i], s_goal=goals[i], obs=extended_obs.tolist())
    print(time.time() - t2)