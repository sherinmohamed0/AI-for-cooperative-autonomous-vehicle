import casadi as ca
import numpy as np

class Layer2_NMPC:
    def __init__(self, N=25, dt=0.2):
        """
        N: Prediction horizon (25 steps)
        dt: Time step (0.2 seconds)
        """
        self.N = N
        self.dt = dt

        # Balanced Optimization Weights (Eliminates tracking-induced phantom braking)
        self.W_stage_track = 10.0   # Soft guide to keep the car centered on the LSTM path
        self.W_vel_track   = 40.0   # Cruise control baseline: maintain smooth forward velocity
        self.W_term_track  = 60.0   # Pulls the vehicle forward toward the distant destination
        self.W_smooth_u    = 2.0    # Minimizes unnecessary acceleration spikes
        self.W_smooth_du   = 20.0   # High penalty on jerk for passenger comfort
        self.W_collision   = 150.0  # Massive penalty for obstacle proximity to trigger early braking

        # Actuator Bounds (Meters)
        self.a_x_min, self.a_x_max = -3.0, 3.0
        self.a_y_min, self.a_y_max = -6.0, 4.0
        self.v_max = 40.0

    def solve(self, current_state, final_goal, ego_data, neighbors_data):
        """
        current_state:   [x, y, vx, vy]
        final_goal:      [x_dest, y_dest]
        """
        opti = ca.Opti()

        # 1. State & Control Variables
        X = opti.variable(4, self.N + 1)
        x_pos, y_pos, vx, vy = X[0, :], X[1, :], X[2, :], X[3, :]

        U = opti.variable(2, self.N)
        ax, ay = U[0, :], U[1, :]

        # 2. Initial State Constraint
        opti.subject_to(X[:, 0] == current_state)

        # 3. Double-Integrator Kinematics
        for k in range(self.N):
            opti.subject_to(X[0, k+1] == X[0, k] + X[2, k] * self.dt + 0.5 * U[0, k] * self.dt**2)
            opti.subject_to(X[1, k+1] == X[1, k] + X[3, k] * self.dt + 0.5 * U[1, k] * self.dt**2)
            opti.subject_to(X[2, k+1] == X[2, k] + U[0, k] * self.dt)
            opti.subject_to(X[3, k+1] == X[3, k] + U[1, k] * self.dt)

        # 4. Physical Boundaries
        opti.subject_to(opti.bounded(self.a_x_min, ax, self.a_x_max))
        opti.subject_to(opti.bounded(self.a_y_min, ay, self.a_y_max))
        opti.subject_to(opti.bounded(0.0, vy, self.v_max))

        # 5. Dynamic Objective Cost Function
        cost = 0

        ego_x_ref = ego_data.get('Pred_Lateral_m', [0.0]*(self.N+1))
        ego_y_ref = ego_data.get('Pred_LocalY_m', [0.0]*(self.N+1))

        # Target cruise speed matches the initial entry velocity
        target_cruise_speed = current_state[3]

        for k in range(self.N):
            # A. Soft Path Tracking
            cost += self.W_stage_track * ((x_pos[k] - ego_x_ref[k])**2 + (y_pos[k] - ego_y_ref[k])**2)

            # B. Cruise Speed Maintenance (Prevents unprovoked brake slamming)
            cost += self.W_vel_track * (vy[k] - target_cruise_speed)**2

            # C. Comfort Control Smoothness & Jerk Control
            cost += self.W_smooth_u * (ax[k]**2 + ay[k]**2)
            if k > 0:
                cost += self.W_smooth_du * ((ax[k] - ax[k-1])**2 + (ay[k] - ay[k-1])**2)

            # D. Uncertainty-Aware Collision Avoidance
            for nbr in neighbors_data:
                nbr_x = nbr.get('Pred_Lateral_m', [0.0]*(self.N+1))
                nbr_y = nbr.get('Pred_LocalY_m', [0.0]*(self.N+1))
                sig_x = nbr.get('Sigma_Lat_m', [0.1]*(self.N+1))
                sig_y = nbr.get('Sigma_Long_m', [0.1]*(self.N+1))

                dist_sq = (x_pos[k] - nbr_x[k])**2 + (y_pos[k] - nbr_y[k])**2
                uncertainty_bubble = (sig_x[k]**2 + sig_y[k]**2)

                # Active cost penalty scaling up based on proximity and predictive variance
                cost += self.W_collision * (uncertainty_bubble / (dist_sq + 0.1))

        # E. Terminal Cost to Target Destination Point
        cost += self.W_term_track * ((x_pos[-1] - final_goal[0])**2 + (y_pos[-1] - final_goal[1])**2)

        opti.minimize(cost)

        # Solver Setup
        p_opts = {"expand": True, "print_time": False}
        s_opts = {"max_iter": 100, "print_level": 0, "acceptable_tol": 1e-4}
        opti.solver('ipopt', p_opts, s_opts)

        try:
            sol = opti.solve()
            return {
                'success': True,
                'a_x': sol.value(ax),
                'a_y': sol.value(ay)
            }
        except RuntimeError:
            return {
                'success': False,
                'a_x': np.zeros(self.N),
                'a_y': np.ones(self.N) * self.a_y_min
            }
