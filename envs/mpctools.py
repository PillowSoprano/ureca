"""
Minimal MPC Tools module for discrete-time simulation.

This module provides a simple discrete-time simulator wrapper
for ODE-based models used in the wastewater treatment system.
"""

import numpy as np


class DiscreteSimulator:
    """
    A simple discrete-time simulator that wraps an ODE function.

    This simulator integrates the ODE function over one timestep using
    a simple Euler integration method.

    Args:
        ode_func: A function that computes dx/dt = f(x, u)
        timestep: The integration timestep (Delta)
        dimensions: List of [state_dim, input_dim]
        var_names: List of variable names (e.g., ["x", "u"])
    """

    def __init__(self, ode_func, timestep, dimensions, var_names):
        self.ode_func = ode_func
        self.timestep = timestep
        self.state_dim = dimensions[0]
        self.input_dim = dimensions[1]
        self.var_names = var_names

    def sim(self, state, input_vec):
        """
        Simulate one timestep forward.

        Args:
            state: Current state vector (numpy array)
            input_vec: Input/control vector (numpy array)

        Returns:
            Next state vector after one timestep
        """
        # Compute the derivative
        dxdt = self.ode_func(state, input_vec)

        # Euler integration: x(t+1) = x(t) + dt * f(x(t), u(t))
        next_state = state + self.timestep * dxdt

        return next_state
