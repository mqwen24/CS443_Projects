'''joint.py
Classes that defines an arm joint
CS443: Bio-Inspired Machne Learning
YOUR NAMES HERE
Project 3: Outstar learning
'''
import numpy as np


class Joint:
    '''A joint of the model arm (e.g. shoulder, elbow, wrist). This is a point on the arm where one arm segment rotates
    with respect to one another (lower arm rotates relate to upper arm at elbow joint).
    '''
    def __init__(self, name, initial_angle, dist2shoulder, angle_limits=(-np.pi/2, np.pi)):
        '''Joint constructor

        Parameters:
        -----------
        name: str. Name of the joint.
        initial_angle: float. Initial angle (in radians) of the joint when the joint is created / in the "default" arm
            posture.
        dist2shoulder: float. Total distance from the current joint to the shoulder joint (defined as (x, y) = (0, 0))
            along the arm segments. This is NOT Euclidean distance.
            For example: Distance from the wrist joint to the shoulder is the sum of the lengths of the upper (between
            shoulder and elbow) and lower (between elbow and wrist) arm segments.
        angle_limits: tuple of 2 floats. format=(min angle, max angle). Ergonomic range of angles for the current joint.
            The joint cannot take on an angle outside this range because it would be uncomfortable/painful/impossible.
            Angles are in radians.

        TODO: Make instance variables for all parameters
        '''
        pass

    def get_name(self):
        '''Returns the name (str) of the joint'''
        pass

    def get_limits(self):
        '''Returns the min/max ergonomic angle range of the joint'''
        pass

    def get_dist2shoulder(self):
        '''Returns the distance of the joint along the arm to the shoulder'''
        pass

    def get_angle(self):
        '''Returns the current joint angle'''
        pass

    def reset_angle(self):
        '''Resets the current joint angle back to its initial angle (regardless of current angle).'''
        pass

    def randomize_angle(self):
        '''Generates and sets the joint angle to a (uniform) random value within the ergonomic range (within bounds).'''
        # generate random angle within bounds.
        pass

    def update_angle(self, muscle_pair_acts, angle_step=0.05, oob_correction=(0.05, 0.1)):
        '''Updates the joint angle based on the DIFFERENCE in activation between the agonist/antagonist muscle pair
        (`muscle_pair_acts`) that controls the joint.

        Parameters:
        -----------
        muscle_pair_acts: ndarray. shape=(2,). Pair of nonnegative values representing the activations of the
            agonist/antagonist muscles (`muscle_pair_acts`) that control the joint. During training, one value will be 0,
            the other will be a positive float.
        angle_step: float. Positive value that specifies how many the joint angle changes per unit muscle activation.
        oob_correction: tuple of 2 floats. format=(min correction, max correction). If the joint angle goes outside the
            ergonomic range, a random correction is applied within the range specified by `oob_correction`. The correction
            always brings the joint angle closer to being within bounds / the ergonomic range.
            Examples with ergonomic range = (0, 1):
            - If joint angle updated to 1.1, the oob_correction would make the angle closer to 1 (e.g. 1.07)
            - If joint angle updated to -0.05, the oob_correction would make the angle closer to 0 (e.g. 0.01)
            Angles are in radians.

        TODO:
        - Update the current angle based on the difference in activation between the muscle pair, scaled by `angle_step`.
        Use the convention: 2nd muscle - 1st muscle.
        - Check if the angle goes out-of-bounds. If so, make a random correction (see `oob_correction`) toward the valid
        angle range.
        '''
        pass


class EndEffector:
    '''The part of the arm (i.e. hand) that makes contact with the target.
    '''
    def __init__(self, name, dist2shoulder):
        '''Constructor

        Parameters:
        -----------
        name: str. Name of the end effector.
        dist2shoulder: float. Total distance between the end effector and the shoulder joint along the arm segments.
            This is NOT Euclidean distance.
        '''
        pass

    def get_name(self):
        '''Returns the name of the EndEffector'''
        pass

    def get_dist2shoulder(self):
        '''Returns the distance of the EndEffector along the arm to the shoulder joint'''
        pass
