from math import inf


def clip(value, min_val, max_val):
    """Clip a value between lower and upper bounds.

    Args:
        value (number): the value to clip
        min_val (number): the lower bound
        max_val (number): the upper bound

    Return:
        The clipped value
    """
    return min(max_val, max(min_val, value))


def exp_filter(alpha, raw_value, value_previous):
    """An exponential filter.

    y(k) = a * y(k-1) + (1-a) * x(k)
        x(k) raw input at time step k
        y(k) filtered output at time step k
        a is a constant between 0 and 1

    Args:
        alpha (float): a smoothing constant between 0 and 1 (typically 0.8 to 0.99)
        raw_value (float): raw input value at current time step
        value_previous (float): filtered value from previous time step

    Return:
        The updated filtered value
    """
    return alpha * value_previous + (1 - alpha) * raw_value


class PIDController:
    def __init__(self, kp=0, ki=0, kd=0, windup=20, alpha=1, u_bounds=[-inf, inf]):
        """
        Args:
            kp(float): proportional control factor
            ki(float): integrative control factor
            kd(float): derivative control factor
            windup(float): maximum integral error value
            alpha(float): derivative filter smoothing factor
            u_bounds([float, float]): upper and lower bounds on the control output(u)
        """

        # Control parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.windup = windup
        self.alpha = alpha
        self.u_min = u_bounds[0]
        self.u_max = u_bounds[1]

        # Dynamics data
        self.error_sum = 0
        self.state_previous = None
        self.delta_state_previous = 0

    def step(self, state_setpoint, state_actual):
        """Step the PID controller and get a new control output.

        This step function should be called at a regular interval, and since it
        is we do not need to use change in time for this implementation. The PID
        gain terms will need to account for the time step.

        Args:
            state_setpoint(float): the goal state for the system
            state_actual(float): the current state of the system
            time(float): current time

        Return:
            Control output signal based on current and desired states
        """

        if self.state_previous == None:
            self.state_previous = state_actual
            return 0

        # Proportional term
        error = state_setpoint - state_actual
        p = self.kp * error

        # Integral term
        self.error_sum += error
        self.error_sum = clip(self.error_sum, -self.windup, self.windup)
        i = self.ki * self.error_sum

        # Derivative term
        delta_state = state_actual - self.state_previous
        delta_state_filtered = exp_filter(
            self.alpha, delta_state, self.delta_state_previous
        )
        d = self.kd * delta_state_filtered
        self.state_previous = state_actual
        self.delta_state_previous = delta_state_filtered

        # Set the control effort
        u = p + i + d
        u = clip(u, self.u_min, self.u_max)
        return u
