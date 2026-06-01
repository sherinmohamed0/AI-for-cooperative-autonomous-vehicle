import numpy as np
from scipy.optimize import minimize

class Layer3_CBF:
    def __init__(self, alpha1=1.5, alpha2=1.5):
        """
        alpha1, alpha2: Tuning parameters for safety aggressiveness.
        """
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        # Bounding Box Physical Margins
        self.d_long = 5.0  # Safe longitudinal buffer (meters)
        self.d_lat  = 2.0  # Safe lateral buffer (meters)

        # Strict physical actuator limits
        self.a_x_min, self.a_x_max = -3.0, 3.0
        self.a_y_min, self.a_y_max = -6.0, 4.0

    def filter_control(self, current_state, u_mpc, neighbors_current):
        """
        Filters NMPC commands through Linear Half-Space Barrier Functions.
        """
        x_ego, y_ego, vx_ego, vy_ego = current_state
        ax_mpc, ay_mpc = u_mpc

        # 1. Objective: Minimize deviation from MPC
        def objective(u):
            return 0.5 * ((u[0] - ax_mpc)**2 + (u[1] - ay_mpc)**2)

        constraints = []

        # 2. Build Linear Half-Space Constraints
        for nbr in neighbors_current:
            x_nbr, y_nbr, vx_nbr, vy_nbr = nbr

            dp_x = x_nbr - x_ego
            dp_y = y_nbr - y_ego
            dv_x = vx_nbr - vx_ego
            dv_y = vy_nbr - vy_ego

            # A. Longitudinal Shielding (Neighbor is in the same lane)
            if abs(dp_x) <= self.d_lat:
                if dp_y > 0: # Neighbor is directly AHEAD
                    def cbf_long_ahead(u, dp=dp_y, dv=dv_y):
                        # h = dp - d_long
                        b = - ((self.alpha1 + self.alpha2)*dv + self.alpha1*self.alpha2*(dp - self.d_long))
                        return -1.0 * u[1] - b  # -a_y - b >= 0
                    constraints.append({'type': 'ineq', 'fun': cbf_long_ahead})

                else: # Neighbor is directly BEHIND
                    def cbf_long_behind(u, dp=dp_y, dv=dv_y):
                        # h = -dp - d_long
                        b = - ((self.alpha1 + self.alpha2)*(-dv) + self.alpha1*self.alpha2*(-dp - self.d_long))
                        return 1.0 * u[1] - b   # a_y - b >= 0
                    constraints.append({'type': 'ineq', 'fun': cbf_long_behind})

            # B. Lateral Shielding (Neighbor is in an adjacent lane right beside us)
            elif abs(dp_y) <= self.d_long:
                if dp_x > 0: # Neighbor is to the RIGHT
                    def cbf_lat_right(u, dp=dp_x, dv=dv_x):
                        # h = dp - d_lat
                        b = - ((self.alpha1 + self.alpha2)*dv + self.alpha1*self.alpha2*(dp - self.d_lat))
                        return -1.0 * u[0] - b  # -a_x - b >= 0
                    constraints.append({'type': 'ineq', 'fun': cbf_lat_right})

                else: # Neighbor is to the LEFT
                    def cbf_lat_left(u, dp=dp_x, dv=dv_x):
                        # h = -dp - d_lat
                        b = - ((self.alpha1 + self.alpha2)*(-dv) + self.alpha1*self.alpha2*(-dp - self.d_lat))
                        return 1.0 * u[0] - b   # a_x - b >= 0
                    constraints.append({'type': 'ineq', 'fun': cbf_lat_left})

        # 3. Actuator Bounds
        bounds = [
            (self.a_x_min, self.a_x_max),
            (self.a_y_min, self.a_y_max)
        ]

        # 4. Solve QP
        initial_guess = [ax_mpc, ay_mpc]
        res = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        # 5. Failsafe Logic
        if res.success:
            return {
                "success": True,
                "u_safe": res.x,
                "mode": "Nominal_MPC"
            }
        else:
            return {
                "success": False,
                "u_safe": np.array([0.0, self.a_y_min]), # Maintain lane, max emergency brake
                "mode": "Emergency_Fallback"
            }
