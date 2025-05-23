from constants import *

"""
state.py

This file contains a class representing a game environment state. You should make use of this class in your solver.

COMP3702 2024 Assignment 2 Support Code
"""


class State:
    """
    Instance of a HexBot environment state.

    See constructor docstring for information on instance variables.

    You may use this class and its functions. You may add your own code to this class (e.g. get_successors function,
    get_heuristic function, etc), but should avoid removing or renaming existing variables and functions to ensure
    Tester functions correctly.
    """

    def __init__(self, environment, BEE_posit, BEE_orient, widget_centres, widget_orients, force_valid=True):
        """
        Construct a HexRobot environment state.

        :param environment: an Environment instance
        :param BEE_posit: (row, col) tuple representing BEE position
        :param BEE_orient: element of BEE_ORIENTATIONS representing BEE orientation
        :param widget_centres: tuple of (row, col) tuples representing centre position of each widget
        :param widget_orients: tuple of elements of WIDGET_ORIENTATIONS representing orientation of each widget
        :param force_valid: If true, raise exception if the created State violates validity constraints
        """
        if force_valid:
            r, c = BEE_posit
            assert isinstance(r, int), '!!! tried to create State but BEE_posit row is not an integer !!!'
            assert isinstance(c, int), '!!! tried to create State but BEE_posit col is not an integer !!!'
            assert 0 <= r < environment.n_rows, '!!! tried to create State but BEE_posit row is out of range !!!'
            assert 0 <= c < environment.n_cols, '!!! tried to create State but BEE_posit col is out of range !!!'
            assert BEE_orient in BEE_ORIENTATIONS, \
                '!!! tried to create State but BEE_orient is not a valid orientation !!!'
            assert len(widget_centres) == environment.n_widgets, \
                '!!! tried to create State but number of widget positions does not match environment !!!'
            assert len(widget_orients) == environment.n_widgets, \
                '!!! tried to create State but number of widget orientations does not match environment !!!'
            for i in range(environment.n_widgets):
                assert widget_orients[i] in WIDGET_ORIENTS[environment.widget_types[i]], \
                    f'!!! tried to create State but widget {i} has invalid orientation for its type !!!'
            # does not check for widget collision or out of bounds
        self.environment = environment
        self.BEE_posit = BEE_posit
        self.BEE_orient = BEE_orient
        self.widget_centres = widget_centres
        self.widget_orients = widget_orients
        self.force_valid = force_valid

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return (self.BEE_posit == other.BEE_posit and
                self.BEE_orient == other.BEE_orient and
                self.widget_centres == other.widget_centres and
                self.widget_orients == other.widget_orients)

    def __hash__(self):
        return hash((self.BEE_posit, self.BEE_orient, self.widget_centres, self.widget_orients))

    def deepcopy(self):
        return State(self.environment, self.BEE_posit, self.BEE_orient, self.widget_centres, self.widget_orients,
                     force_valid=self.force_valid)




