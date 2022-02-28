from random import gauss


class MultiRotor:
    """Simple vertical dynamics for a multirotor vehicle."""

    GRAVITY = -9.81

    def __init__(
        self, altitude=10, velocity=0, mass=1.54, emc=10.0, dt=0.05, noise=0.1
    ):
        """

        Args:
            altitude (float): initial altitude of the vehicle
            velocity (float): initial velocity of the vehicle
            mass (float): mass of the vehicle
            emc (float): electromechanical constant for the vehicle
            dt (float): simulation time step
            noise (float): standard deviation of normally distributed simulation noise
        """

        self.y0 = altitude
        self.y1 = velocity
        self.mass = mass
        self.emc = emc
        self.dt = dt
        self.noise = noise

    def step(self, effort):
        """Advance the multirotor simulation and apply motor forces.

        Args:
            effort (float): related to the upward thrust of the vehicle,
                it must be >= 0

        Return:
            The current state (altitude, velocity) of the vehicle.
        """

        effort = max(0, effort)

        scaled_effort = self.emc / self.mass * effort
        net_acceleration = MultiRotor.GRAVITY - 0.75 * self.y1 + scaled_effort

        # Don't let the vehcicle fall through the ground
        if self.y0 <= 0 and net_acceleration < 0:
            y0dot = 0
            y1dot = 0
        else:
            y0dot = self.y1
            y1dot = net_acceleration

        self.y0 += y0dot * self.dt
        self.y1 += y1dot * self.dt

        self.y0 += gauss(0, self.noise)

        return self.y0, self.y1

    def get_altitude(self):
        """Return the current altitude."""
        return self.y0

    def get_delta_time(self):
        """Return the simulation time step."""
        return self.dt
