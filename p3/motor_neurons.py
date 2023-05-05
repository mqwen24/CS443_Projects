'''motor_neurons.py
Classes that defines set of motorneurons that sense joint angles and hand movement direction
CS443: Bio-Inspired Machine Learning
YOUR NAMES HERE
Project 3: Outstar learning
'''
import numpy as np
import matplotlib.pyplot as plt
from outstar import Source


class MotorNeurons(Source):
    def __init__(self, joints, n_joint_angle_prefs=7, n_dir_angle_prefs=30, n_winners=3):
        '''Neurons that sense the state of the arm. Each neuron is sensitive to a combination of all the joint angles
        and the direction in which the end effector is moving.

        Sources of the Outstar network

        Each motorneuron receives input about the angular state of ALL 3 of the joints.
        Each motorneuron sends a output signal to activate ALL 3 muscle groups (6 muscles total).
        Each motorneuron is tuned to specific combinations of:
        - Joint angles
        - Direction vectors (training: diff in pre/post babble end effector position,
                             test: diff between end effector vs target direction)

        Parameters:
        -----------
        joints: list of `Joint`. Joint objects that make up the arm.
        n_joint_angle_prefs: int. Number of evenly spaced angles within each joint's ergonomic range the motorneurons
            should prefer. `n_joint_angle_prefs` is the same for each joint (for simplicity).
            So if n_joint_angle_prefs=3, there are 3 preferred shoulder joint angles, 3 preferred elbow joint angles,
            and 3 preferred wrist joint angles (and each of these are likely different).
        n_dir_angle_prefs: int. Number of preferred movement angles of the end effector (hand) over 360°.
        n_winners: int. Number of neurons that win the competition and are allowed to have nonzero net_act values.

        TODO:
        - Make an instance variable for `n_winners` and any other parameters you need later.
        - Call self.initialize_pref_angles to set an instance variance for the array that contains each neuron's
        preferred combination of joint angles.
        '''
        self.joints = joints
        self.n_joint_angle_prefs = n_joint_angle_prefs
        self.n_dir_angle_prefs = n_dir_angle_prefs
        self.n_winners = n_winners

        self.pref_angles = self.initialize_pref_angles(self.joints, self.n_joint_angle_prefs, self.n_dir_angle_prefs)
        
    def get_num_units(self):
        '''Returns the number of motorneurons in the layer.

        Returns:
        -----------
        int. The number of motorneurons is the same as the total number of combinations of (3) preferred joint angles
            and preferred hand movement directions. In the default configuration with 7 preferred angles per joint and
            30 preferred hand movement directions, the total number of neurons is 10,290 (do NOT hardcode this!).
        '''
        return self.n_joint_angle_prefs**3*self.n_dir_angle_prefs

    def get_pref_angles(self):
        '''Returns the joint angle + hand movement angle preferences of each neuron in the layer

        Returns:
        -----------
        ndarray. (4=3 joints + 1 hand, num_motorneurons).
        '''
        return self.pref_angles

    def set_pref_angles(self, prefs):
        '''Sets the joint angle + hand movement angle preferences of each neuron in the layer

        Parameters:
        -----------
        prefs: ndarray. (4=3 joints + 1 hand, num_motorneurons).
        '''
        self.pref_angles = prefs

    def set_num_winners(self, n_winners):
        '''Sets the number of neurons that win the competition and send non-zero net_act values.

        Parameters:
        -----------
        n_winners: int. Number of winners.
        '''
        self.n_winners = n_winners

    def initialize_pref_angles(self, joints, n_joint_angle_prefs, n_dir_angle_prefs):
        '''Initializes the array that defines each neuron's preferred set of joint angles (3) and hand movement
        directions (1).

        Examples are [0°, 0°, 0°, 0°], [0°, 10°, 0°, 30°], [10°, 20°, 30°, 40°]. Any one of these
        arrays define the angles of the joints and hand that make one neuron "happiest" (i.e. produces the highest
        net_in). This method creates the array of preferences for ALL motorneurons.

        Angle preferences are evenly spaced within:
        - each joint's ergnomic range (each joint likely has a different valid range).
        - 360° for the end effector (hand), because the hand can move in any direction. We are working in radians so
        this range in our angle convention is (-π, -+π).

        Parameters:
        -----------
        joints: list of `Joint`. Joint objects that make up the arm.
        n_joint_angle_prefs: int. Number of evenly spaced angles within each joint's ergonomic range the motorneurons
            should prefer. `n_joint_angle_prefs` is the same for each joint (for simplicity).
            So if n_joint_angle_prefs=3, there are 3 preferred shoulder joint angles, 3 preferred elbow joint angles,
            and 3 preferred wrist joint angles (and each of these are likely different).
        n_dir_angle_prefs: int. Number of preferred movement angles of the end effector (hand) over 360°.

        Returns:
        -----------
        ndarray. (4, num_neurons). The angle preferences of all neurons. See notebook for an example array.

        HINT:
        - You solved a similar problem in the SOM project when creating the Gaussian row/col indices and Project 0 of
        CS343 with 2D coordinates when you plotted a heatmap. In each of these cases, you had to generate all possible
        combinations of values in two 1D arrays (rows and cols). The problem here is the same, except you want to
        generate all combinations of the values in four arrays.
        '''
        shoulder_angle_range = joints[0].get_limits()
        elbow_angle_range = joints[1].get_limits()
        wrist_angle_range = joints[2].get_limits()
        hand_angle_range = (-np.pi, np.pi)

        # numpy.linspace(start, stop, num=50)
        shoulder_pref_angles = np.linspace(shoulder_angle_range[0], shoulder_angle_range[1], n_joint_angle_prefs)
        elbow_pref_angles = np.linspace(elbow_angle_range[0], elbow_angle_range[1], n_joint_angle_prefs)
        wrist_pref_angles = np.linspace(wrist_angle_range[0], wrist_angle_range[1], n_joint_angle_prefs)
        hand_pref_angles = np.linspace(hand_angle_range[0], hand_angle_range[1], n_dir_angle_prefs)

        # num_units = # motor neurons = # different angle combinations;
        pref_angles = np.zeros((4, self.get_num_units()))

        neuron_i = 0
        for i in range(len(shoulder_pref_angles)):
            for j in range(len(elbow_pref_angles)):
                for k in range(len(wrist_pref_angles)):
                    for m in range(len(hand_pref_angles)):
                        pref_angles[0, neuron_i] = shoulder_pref_angles[i]
                        pref_angles[1, neuron_i] = elbow_pref_angles[j]
                        pref_angles[2, neuron_i] = wrist_pref_angles[k]
                        pref_angles[3, neuron_i] = hand_pref_angles[m]
                        neuron_i += 1

        return pref_angles

    def net_in(self, joint_angles, move_dir):
        '''Each cell compares the arm's current joint angles (`joint_angles`) and hand movement direction angle
        (`move_dir`) with its preferred set of 4 angles. Close matches between the current arm angles and the neuron's
        preference results in high activation. Dissimilar angles result in low activation, close to 0.

        See notebook for refresher on equation.

        Parameters:
        -----------
        joint_angles: ndarray. (num_joints,). The 3 joint angles the arm has in its current posture.
        move_dir: float. The current movement direction of hand. In training this is the direction angle of the hand
            before vs after the current babble (random movement). In testing, this is the direction between the hand
            and the target.

        Returns:
        -----------
        ndarray. shape=(num_neurons,)=(10290,). Total match between each neuron's preferred set of 4 angles and the
            current joint angles `joint_angles` and hand movement direction.

        '''
        num_neurons = self.get_num_units()
        net_in = np.zeros(shape=(num_neurons,))

        for i in range(num_neurons):
            pref_angle = self.pref_angles.T[i, :]
            net_in[i] = 4- (np.abs(move_dir- pref_angle[3])+ np.sum(np.abs(joint_angles- pref_angle[0:3])))/ np.pi

        return net_in

    def net_act(self, net_in, eps=1e-10):
        '''Neurons compete and only the net_in values of the top `n_winners` neurons have non-zero net_act values.

        Parameters:
        -----------
        net_in: ndarray. shape=(num_neurons,)=(10290,). Net input of motorneurons reflecting match between each neuron's
            angle preferences and the set of angles that characterize the current state of the arm.
        eps: float. Small number to prevent division by zero when normalizing.

        Returns:
        -----------
        ndarray. shape=(num_neurons,)=(10290,). Normalized activation of the motorneurons after a competitive process.

        TODO:
        - Neurons that survive the competition have their net_in value set as their net_act value (others set to 0).
        - The surviving net_act values are normalized relative to the top winner (with the max net_act) so that the top
        winner has a net_act of 1, the other survivors have nonnegative net_acts between 0 and 1.

        HINT:
        - This competition is very similar to that from the Hebbian Learning project!
        '''
        top_k_indices = np.argsort(net_in)[::-1][:self.n_winners]
        
        net_act = np.zeros_like(net_in)
        net_act[top_k_indices] = net_in[top_k_indices]

        net_act = net_act/(np.max(net_act)+eps)

        return net_act

    def forward(self, joint_angles, move_dir):
        '''Forward pass through the motorneuron layer.

        Parameters:
        -----------
        joint_angles: ndarray. (num_joints,). The 3 joint angles the arm has in its current posture.
        move_dir: float. The current movement direction of hand. In training this is the direction angle of the hand
            before vs after the current babble (random movement). In testing, this is the direction between the hand
            and the target.

        Returns:
        -----------
        ndarray. shape=(num_neurons,)=(10290,). Normalized activation of the motorneurons after a competitive process.
        '''
        net_in = self.net_in(joint_angles, move_dir)
        net_act = self.net_act(net_in)

        return net_act

    def plot(self, act, true_angles, titles=['Shoulder angle', 'Elbow angle', 'Wrist angle', 'Move dir'], num_x_labels=5):
        '''Create a vertical stack of plots showing the evidence for the current shoulder, elbow, wrist, and hand
            direction angle that characterize the state of the arm as coded by the motorneurons.

        Parameters:
        -----------
        act: ndarray. shape=(num_neurons,)=(10290,). Either motorneuron net_in or net_act.
        true_angles: ndarray. shape=(4,). True set of current joint angles and hand movement direction angle.
        titles: list of str. Titles for each subplot showing the evidence for a particular joint/hand angle.
        num_x_labels: int. Number of angles to show on the x axis of each subplot.

        (This is provided to you and should not require modification)
        '''

        angle_prefs = self.get_pref_angles()
        fig, axes = plt.subplots(nrows=len(angle_prefs), ncols=1, figsize=(5, len(angle_prefs)*3.5))

        for i in range(len(angle_prefs)):
            # Get set of preferred angles for current thing we are detecting (angle of joint i or hand move dir)
            curr_pref_angles = angle_prefs[i]
            curr_pref_angle_set = np.unique(curr_pref_angles)
            
            curr_resp = np.zeros(len(curr_pref_angle_set))
            # Step through each preferred angle, determine total evidence for it
            for a in range(len(curr_pref_angle_set)):
                curr_resp[a] = np.sum(act[curr_pref_angles == curr_pref_angle_set[a]])

            x = np.linspace(curr_pref_angle_set.min(), curr_pref_angle_set.max(), len(curr_pref_angle_set))
            axes[i].plot(x, curr_resp, 'o-')
            axes[i].axvline(x=true_angles[i], linestyle='--', color='k')

            x_ticks = np.linspace(curr_pref_angle_set.min(), curr_pref_angle_set.max(), num_x_labels)
            axes[i].set_xticks(x_ticks)
            axes[i].set_xticklabels([f'{tickmark/np.pi:.2f}π' for tickmark in x_ticks], fontdict={'fontsize': 12})

            axes[i].set_ylabel('Evidence')

            axes[i].set_title(titles[i])
        axes[i].set_xlabel('Angle')
        plt.show()