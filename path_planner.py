"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""

import numpy as np
import heapq


class AStarPlanner:
    """
    A* path planning algorithm for 3D environment with cylindrical obstacles.
    """
    
    def __init__(self, env, resolution=0.5):
        """
        Initialize the A* planner.
        
        Parameters:
            env: FlightEnvironment object
            resolution: Grid resolution for discretization (smaller = finer grid)
        """
        self.env = env
        self.resolution = resolution
        
    def plan(self, start, goal):
        """
        Plan a collision-free path from start to goal using A* algorithm.
        
        Parameters:
            start: tuple (x, y, z) - starting position
            goal: tuple (x, y, z) - goal position
            
        Returns:
            path: N×3 numpy array containing waypoints from start to goal
        """
        # Convert continuous coordinates to grid coordinates
        start_node = self._to_grid(start)
        goal_node = self._to_grid(goal)
        
        # Check if start or goal is invalid
        if self.env.is_outside(start) or self.env.is_collide(start):
            raise ValueError("Start position is invalid (outside or collision)")
        if self.env.is_outside(goal) or self.env.is_collide(goal):
            raise ValueError("Goal position is invalid (outside or collision)")
        
        # Initialize open and closed sets
        open_set = []  # Priority queue: (f_score, counter, node)
        closed_set = set()
        
        # Track g_scores and parent nodes
        g_score = {start_node: 0}
        f_score = {start_node: self._heuristic(start_node, goal_node)}
        parent = {}
        
        # Counter for tie-breaking in priority queue
        counter = 0
        heapq.heappush(open_set, (f_score[start_node], counter, start_node))
        
        while open_set:
            # Get node with lowest f_score
            current_f, _, current = heapq.heappop(open_set)
            
            # Check if we reached the goal
            if current == goal_node:
                return self._reconstruct_path(parent, current, start_node)
            
            # Skip if already processed
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Check if neighbor is valid
                neighbor_pos = self._to_continuous(neighbor)
                if self.env.is_outside(neighbor_pos) or self.env.is_collide(neighbor_pos):
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + self._distance(current, neighbor)
                
                # Update if this path is better
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    parent[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_node)
                    
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        # No path found
        raise RuntimeError("No collision-free path found from start to goal")
    
    def _to_grid(self, pos):
        """Convert continuous position to grid coordinates."""
        return (
            int(round(pos[0] / self.resolution)),
            int(round(pos[1] / self.resolution)),
            int(round(pos[2] / self.resolution))
        )
    
    def _to_continuous(self, grid_pos):
        """Convert grid coordinates to continuous position."""
        return (
            grid_pos[0] * self.resolution,
            grid_pos[1] * self.resolution,
            grid_pos[2] * self.resolution
        )
    
    def _heuristic(self, node1, node2):
        """
        Heuristic function (Euclidean distance).
        """
        return np.sqrt(
            (node1[0] - node2[0])**2 + 
            (node1[1] - node2[1])**2 + 
            (node1[2] - node2[2])**2
        )
    
    def _distance(self, node1, node2):
        """
        Actual distance between two nodes (Euclidean).
        """
        return self._heuristic(node1, node2)
    
    def _get_neighbors(self, node):
        """
        Get all valid neighboring nodes (26-connectivity in 3D).
        """
        neighbors = []
        x, y, z = node
        
        # Generate all 26 neighbors (including diagonals)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((x + dx, y + dy, z + dz))
        
        return neighbors
    
    def _reconstruct_path(self, parent, current, start):
        """
        Reconstruct path from start to goal using parent dictionary.
        """
        path = [self._to_continuous(current)]
        
        while current in parent:
            current = parent[current]
            path.append(self._to_continuous(current))
        
        path.reverse()
        
        # Smooth the path by removing redundant waypoints
        path = self._smooth_path(path)
        
        return np.array(path)
    
    def _smooth_path(self, path):
        """
        Smooth path by removing intermediate points on straight lines.
        Uses line-of-sight checking.
        """
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Try to connect to farthest visible point
            for next_idx in range(len(path) - 1, current_idx, -1):
                if self._is_line_collision_free(path[current_idx], path[next_idx]):
                    smoothed.append(path[next_idx])
                    current_idx = next_idx
                    break
            else:
                # If no point is visible, move to next point
                current_idx += 1
                if current_idx < len(path):
                    smoothed.append(path[current_idx])
        
        return smoothed
    
    def _is_line_collision_free(self, p1, p2, num_checks=20):
        """
        Check if the straight line between p1 and p2 is collision-free.
        """
        for i in range(num_checks + 1):
            t = i / num_checks
            point = (
                p1[0] + t * (p2[0] - p1[0]),
                p1[1] + t * (p2[1] - p1[1]),
                p1[2] + t * (p2[2] - p1[2])
            )
            if self.env.is_outside(point) or self.env.is_collide(point):
                return False
        return True
            











