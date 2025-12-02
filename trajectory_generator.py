"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""

import numpy as np
import matplotlib.pyplot as plt


class PolynomialTrajectory:
    """
    Polynomial-based trajectory generator for smooth path following.
    
    Uses 5th-order polynomials for each segment to ensure continuity in 
    position, velocity, and acceleration. The 5th-order polynomial is chosen
    because it allows us to specify 6 boundary conditions:
    - Position at start and end
    - Velocity at start and end
    - Acceleration at start and end
    
    This ensures C² continuity (continuous up to second derivative), which is
    important for smooth robot motion.
    """
    
    def __init__(self, waypoints, velocity=2.0):
        """
        Initialize trajectory generator.
        
        Parameters:
            waypoints: N×3 numpy array of path waypoints
            velocity: Average velocity for trajectory (m/s)
        """
        self.waypoints = np.array(waypoints)
        self.velocity = velocity
        self.trajectory = None
        self.time_points = None
        
    def generate(self):
        """
        Generate smooth trajectory using 5th-order polynomial segments.
        
        Returns:
            t: Array of time samples
            trajectory: M×3 array of (x, y, z) positions at each time sample
        """
        n_waypoints = len(self.waypoints)
        if n_waypoints < 2:
            raise ValueError("Need at least 2 waypoints to generate trajectory")
        
        # Calculate time allocation for each segment based on distance
        segment_times = self._calculate_segment_times()
        total_time = np.sum(segment_times)
        
        # Generate time samples (100 Hz sampling)
        dt = 0.01
        # Calculate number of samples to cover exact duration
        n_samples = int(np.round(total_time / dt)) + 1
        t = np.linspace(0, total_time, n_samples)
        
        # Initialize trajectory array
        trajectory = np.zeros((len(t), 3))
        
        # Generate trajectory for each axis independently
        for axis in range(3):
            trajectory[:, axis] = self._generate_axis_trajectory(
                self.waypoints[:, axis], segment_times, t
            )
        
        self.trajectory = trajectory
        self.time_points = t
        
        return t, trajectory
    
    def _calculate_segment_times(self):
        """
        Calculate time duration for each segment based on distance and velocity.
        """
        segment_times = []
        for i in range(len(self.waypoints) - 1):
            distance = np.linalg.norm(self.waypoints[i+1] - self.waypoints[i])
            time = distance / self.velocity
            # No minimum time constraint - let natural velocity determine timing
            segment_times.append(time)
        return np.array(segment_times)
    
    def _generate_axis_trajectory(self, waypoint_axis, segment_times, t):
        """
        Generate trajectory for a single axis using 5th-order polynomials.
        
        Parameters:
            waypoint_axis: Array of waypoint positions for this axis
            segment_times: Array of time durations for each segment
            t: Time samples
            
        Returns:
            positions: Array of positions at each time sample
        """
        n_segments = len(segment_times)
        positions = np.zeros(len(t))
        
        # Cumulative time at each waypoint
        cumulative_times = np.zeros(n_segments + 1)
        for i in range(n_segments):
            cumulative_times[i+1] = cumulative_times[i] + segment_times[i]
        
        # Generate polynomial for each segment
        for seg in range(n_segments):
            t_start = cumulative_times[seg]
            t_end = cumulative_times[seg + 1]
            
            # Find time samples in this segment
            # For the last segment, use a slightly larger tolerance to catch the end point
            if seg == n_segments - 1:
                mask = (t >= t_start) & (t <= t_end + 1e-9)  # Small tolerance for floating point
            else:
                mask = (t >= t_start) & (t < t_end)
            
            t_seg = t[mask] - t_start
            
            # Boundary conditions for 5th-order polynomial
            p0 = waypoint_axis[seg]      # Start position
            p1 = waypoint_axis[seg + 1]  # End position
            
            # Calculate velocities for continuity
            # Use a smoother velocity transition by averaging directions
            
            # Start velocity of this segment
            if seg == 0:
                # First segment: start from rest
                v0 = 0.0
            else:
                # Average of previous and current segment directions for smoother transition
                prev_direction = (waypoint_axis[seg] - waypoint_axis[seg-1]) / segment_times[seg-1]
                curr_direction = (waypoint_axis[seg+1] - waypoint_axis[seg]) / segment_times[seg]
                v0 = (prev_direction + curr_direction) / 2.0
            
            # End velocity of this segment  
            if seg == n_segments - 1:
                # Last segment: end at rest
                v1 = 0.0
            else:
                # Average of current and next segment directions for smoother transition
                curr_direction = (waypoint_axis[seg+1] - waypoint_axis[seg]) / segment_times[seg]
                next_direction = (waypoint_axis[seg+2] - waypoint_axis[seg+1]) / segment_times[seg+1]
                v1 = (curr_direction + next_direction) / 2.0
            
            a0 = 0  # Start acceleration
            a1 = 0  # End acceleration
            
            # Duration of this segment
            T = segment_times[seg]
            
            # Solve for 5th-order polynomial coefficients
            # p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
            coeffs = self._solve_polynomial_coefficients(p0, p1, v0, v1, a0, a1, T)
            
            # Evaluate polynomial at time samples
            positions[mask] = self._evaluate_polynomial(coeffs, t_seg)
        
        return positions
    
    def _solve_polynomial_coefficients(self, p0, p1, v0, v1, a0, a1, T):
        """
        Solve for 5th-order polynomial coefficients given boundary conditions.
        
        Boundary conditions:
            p(0) = p0,   p(T) = p1
            v(0) = v0,   v(T) = v1
            a(0) = a0,   a(T) = a1
        """
        # Coefficient matrix for 5th-order polynomial
        # Using standard equations for position, velocity, and acceleration constraints
        c0 = p0
        c1 = v0
        c2 = a0 / 2.0
        
        # Solve for remaining coefficients using end conditions
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T
        
        # System of equations for c3, c4, c5
        A = np.array([
            [T3, T4, T5],
            [3*T2, 4*T3, 5*T4],
            [6*T, 12*T2, 20*T3]
        ])
        
        b = np.array([
            p1 - c0 - c1*T - c2*T2,
            v1 - c1 - 2*c2*T,
            a1 - 2*c2
        ])
        
        c3, c4, c5 = np.linalg.solve(A, b)
        
        return np.array([c0, c1, c2, c3, c4, c5])
    
    def _evaluate_polynomial(self, coeffs, t):
        """
        Evaluate 5th-order polynomial at given time points.
        """
        return (coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + 
                coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5)
    
    def plot_trajectory(self, show_waypoints=True, remote=False):
        """
        Plot the generated trajectory with three subplots for x, y, z vs time.
        
        Parameters:
            show_waypoints: Whether to show the original waypoints
            remote: If True, save to file; if False, show plot
        """
        if self.trajectory is None:
            raise ValueError("Must call generate() before plotting")
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)']
        colors = ['red', 'green', 'blue']
        
        # Calculate time points for waypoints
        segment_times = self._calculate_segment_times()
        waypoint_times = np.zeros(len(self.waypoints))
        for i in range(1, len(self.waypoints)):
            waypoint_times[i] = waypoint_times[i-1] + segment_times[i-1]
        
        for i in range(3):
            # Plot trajectory
            axes[i].plot(self.time_points, self.trajectory[:, i], 
                        color=colors[i], linewidth=2, label='Trajectory')
            
            # Plot waypoints if requested
            if show_waypoints:
                axes[i].scatter(waypoint_times, self.waypoints[:, i], 
                              color='black', s=50, marker='o', 
                              zorder=5, label='Waypoints')
            
            axes[i].set_ylabel(labels[i], fontsize=11)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='best')
            
            # Only show x-label on bottom plot
            if i == 2:
                axes[i].set_xlabel('Time (s)', fontsize=11)
        
        plt.suptitle('Polynomial Trajectory - Position vs Time', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if remote:
            plt.savefig('trajectory_time_series.png', dpi=300, bbox_inches='tight')
            print("Trajectory time series saved as 'trajectory_time_series.png'")
            plt.close()
        else:
            plt.show()
    
    def get_trajectory(self):
        """
        Get the generated trajectory.
        
        Returns:
            t: Time array
            trajectory: Position array (N×3)
        """
        if self.trajectory is None:
            raise ValueError("Must call generate() before getting trajectory")
        return self.time_points, self.trajectory

