from flight_environment import FlightEnvironment
from path_planner import AStarPlanner
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='UAV Path Planning and Trajectory Generation')
parser.add_argument('--remote', action='store_true', 
                    help='Save plots to files instead of showing them (for remote environments)')
args = parser.parse_args()

# Set random seed for reproducibility
np.random.seed(42)

env = FlightEnvironment(50)
start = (1,2,0)
goal = (18,18,3)

# --------------------------------------------------------------------------------------------------- #
# Call your path planning algorithm here. 
# The planner should return a collision-free path and store it in the variable `path`. 
# `path` must be an N×3 numpy array, where:
#   - column 1 contains the x-coordinates of all path points
#   - column 2 contains the y-coordinates of all path points
#   - column 3 contains the z-coordinates of all path points
# This `path` array will be provided to the `env` object for visualization.

# Create A* planner and find path
planner = AStarPlanner(env, resolution=0.5)
path = planner.plan(start, goal)

print(f"Path found with {len(path)} waypoints")

# --------------------------------------------------------------------------------------------------- #


env.plot_cylinders(path, remote=args.remote)


# --------------------------------------------------------------------------------------------------- #
#   Call your trajectory planning algorithm here. The algorithm should
#   generate a smooth trajectory that passes through all the previously
#   planned path points.
#
#   After generating the trajectory, plot it in a new figure.
#   The figure should contain three subplots showing the time histories of
#   x, y, and z respectively, where the horizontal axis represents time (in seconds).
#
#   Additionally, you must also plot the previously planned discrete path
#   points on the same figure to clearly show how the continuous trajectory
#   follows these path points.

from trajectory_generator import PolynomialTrajectory

# Generate smooth trajectory through waypoints
traj_gen = PolynomialTrajectory(path, velocity=2.0)
t, trajectory = traj_gen.generate()

print(f"Trajectory generated with {len(trajectory)} time steps over {t[-1]:.2f} seconds")

# Plot trajectory with waypoints (time domain)
traj_gen.plot_trajectory(show_waypoints=True, remote=args.remote)

# Plot 3D visualization with both path and trajectory
env.plot_trajectory_3d(path, trajectory, remote=args.remote)

# --------------------------------------------------------------------------------------------------- #



# You must manage this entire project using Git. 
# When submitting your assignment, upload the project to a code-hosting platform 
# such as GitHub or GitLab. The repository must be accessible and directly cloneable. 
#
# After cloning, running `python3 main.py` in the project root directory 
# should successfully execute your program and display:
#   1) the 3D path visualization, and
#   2) the trajectory plot.
#
# You must also include the link to your GitHub/GitLab repository in your written report.
