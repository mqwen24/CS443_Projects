'''model_arm.py
Classes that defines a multi-jointed model arm
CS443: Bio-Inspired Machne Learning
YOUR NAMES HERE
Project 3: Outstar learning
'''
import numpy as np
import matplotlib.pyplot as plt
from viz import ArmPlot


class Arm:
    '''Multi-jointed arm that is trained to reach for targets (meaning the end effector intercepts them).
    '''
    def __init__(self, joints, effector, outstar_net):
        '''Arm constructor

        Parameters:
        -----------
        joints: list of `Joint`. Joint objects that make up the arm.
        effector: EndEffector. The end effector at the end of the arm that makes contact with targets.
        outstar_net: Outstar. The Outstar neural network that learns the muscle activations (sink layer) needed for the
            arm to move closer to the target based on the motorneuron activations (source layer) that sense the current
            state of the arm.
        '''
        # Placeholder for ArmPlot object used to visualize arm position in workspace. Keep this here.
        self.curr_plot = None
        self.joints = joints
        self.effector = effector
        self.outstar_net = outstar_net

    def get_joints(self):
        '''Returns the list of `Joint` objects that make up the arm.'''
        return self.joints

    def get_effector(self):
        '''Returns the `EndEffector` object associated with the arm.'''
        return self.effector

    def get_wts(self):
        '''Returns the weights learned by the Outstar network.
        (This is provided to you and should not require any changes)
        '''
        return self.outstar.get_wts()

    def get_joint_positions(self):
        '''Computes the (x, y) positions of the arm joints (shoulder, elbow, wrist) AND the end effector (hand).
        The shoulder joint is assumed to always remain at (x, y) = (0, 0).

        See notebook for a refresher on the equation to compute the (x, y) components.

        Returns:
        -----------
        ndarray. shape=(4, 2). The (x, y) positions of the joints and end effector in the following order:
            shoulder (0), elbow (1), wrist (2), hand (3)

        Hint: Each joint/end effector maintains its own TOTAL distance along the arm to the shoulder. It is important to
        note, however, that each "l" variable in position equation is NOT the total length. It is the length of the arm
        SEGMENT between each joint and the adjacent joint that is one joint closer to the shoulder. For example, l1 is
        the length (distance) between the elbow and wrist joints.
        '''
        shoulder = self.joints[0]
        elbow = self.joints[1]
        wrist = self.joints[2]

        l0 = elbow.get_dist2shoulder() - shoulder.get_dist2shoulder()
        l1 = wrist.get_dist2shoulder() - elbow.get_dist2shoulder()
        l2 = self.effector.get_dist2shoulder() - wrist.get_dist2shoulder()
        
        shoulder_angle = shoulder.get_angle()
        elbow_angle = elbow.get_angle()
        wrist_angle = wrist.get_angle()

        shoulder_pos = [0, 0]
        # print(shoulder_angle)
        elbow_pos = [l0*np.cos(shoulder_angle), l0*np.sin(shoulder_angle)]
        wrist_pos = [elbow_pos[0]+l1*np.cos(shoulder_angle+elbow_angle),
                     elbow_pos[1]+l1*np.sin(shoulder_angle+elbow_angle)]
        hand_pos = [wrist_pos[0]+l2*np.cos(shoulder_angle+elbow_angle+wrist_angle),
                    wrist_pos[1]+l2*np.sin(shoulder_angle+elbow_angle+wrist_angle)]

        pos = np.array([shoulder_pos, elbow_pos, wrist_pos, hand_pos])

        return pos

    def reset_joint_angles(self):
        '''Resets the angle of each joint in the arm to its initial angle.'''
        for joint in self.joints:
            joint.reset_angle()

    def randomize_joint_angles(self):
        '''Sets the angle of each joint to a random value within the applicable ergonomic range.'''
        for joint in self.joints:
            joint.randomize_angle()

    def get_movement_dir(self, eff_pos_2, eff_pos_1):
        '''Compute the movement direction of the end effector (hand) between two (x, y ) positions (`eff_pos_2` and
        `eff_pos_1`).

        Parameters:
        -----------
        eff_pos_2: ndarray. shape=(2,). The (x, y) coordinate of the end effector AFTER the movement.
        eff_pos_1: ndarray. shape=(2,). The (x, y) coordinate of the end effector BEFORE the movement.

        Returns:
        -----------
        float. The angle (in radians) between the two end effector positions. Angle assumed to be mapped in the range
        [-π, +π] — the convention used by the atan2 function.

        NOTE: The end effector position parameters (`eff_pos_2`, `eff_pos_1`) have different meanings depending on
        whether we are in training or testing (prediction) mode.

        Training:
            - eff_pos_2 = post-babble effector (x, y) position
            i.e. the hand position after the small movement of the hand caused by the random muscle activation.
            - eff_pos_1 = pre-babble effector (x, y) position
            i.e. the hand position before the small movement of the hand caused by the random muscle activation.

        Testing:
            - eff_pos_2 = desired effector (x, y) position (i.e. target position)
            - eff_pos_1 = current effector (x, y) position

        HINT: Check out np.atan2
        '''
        return np.arctan2(eff_pos_2[1] - eff_pos_1[1], eff_pos_2[0] - eff_pos_1[0])

    def train(self, epochs=100, randomize_joints_every=10, n_babbles=9, verbose=True, print_every=20, visualize=True):
        '''Trains the arm using the Outstar neural network to learn the mapping between motorneuron activations (sink)
        and the necessary muscle activations (source) to get the hand closer to the target.

        Parameters:
        -----------
        epochs: int. Number of training epochs, which is equal to the number of randomly selected arm postures used
            (not counting babbling).
        randomize_joints_every: int. Randomize the joint angles (set new random arm posture) after this many epochs.
        n_babbles: int. Number of small random movements to make per epoch. Could be thought of as number of iterations
            per epoch.
        verbose: bool. Whether or not to print out progress during training. If False, there should be no print outs
            during training.
        print_every: int. Print out current epoch and any other progress information every this many epochs.
        visualize: bool. Whether to create a plot showing the arm in the workspace that animates as the arm babbles and
            undergoes new postures.

        TODO:
        1. On each epoch, have the arm perform a set of `n_babbles`.
        2. On each babble, generate a new set of random muscle activations. Apply them to move the arm, meaning that the
        muscle activations cause updates to the joint angles.
        3. Once the arm moves slightly, compute the direction (i.e. angle) in which the end effector (hand) moved by
        comparing the positions of the hand before/after the movement caused by the current babble.
        4. Want you have the muscle activations, a record of the initial joint angles before the arm moved on the current
        babble, and the hand movement direction before/after the babble, run a Outstar training iteration to update the
        weights.
        5. Every `randomize_joints_every` epochs, randomize the joint angles to make a large change in the arm's posture.

        HINT:
        - One of the methods you already implemented computes the current hand position.
        - In Task 2c you wrote code that should be very helpful for TODO item 2 above.
        '''
        # Keep me at the start of the method
        if visualize:
            arm_plot = ArmPlot()

        # print the verbose message:
        print(f'Starting to train network ....')

        for i in range(epochs):
            if i % randomize_joints_every == 0:
                self.randomize_joint_angles()

            if i % print_every == 0 and verbose is True:
                print(f'Epoch {i+1}/{epochs} start ...')

            for j in range(n_babbles):
                pre_pos = self.get_joint_positions()
                initial_joint_angle = self.get_joint_angles()
                
                # generate a new set of random muscle activations
                muscles = self.outstar_net.get_sink()
                rand_muscle_act = muscles.randomize_acts()

                # use the muscle activations to update joint angles
                for k in range(len(self.joints)):
                    self.joints[k].update_angle(rand_muscle_act[k*2: k*2+2])

                # compute direction in which the end effector moved
                # compare postions of the hand before / after the movement caused by the current angle
                after_pos = self.get_joint_positions()
                move_dir = self.get_movement_dir(after_pos[-1], pre_pos[-1])
                
                
                # update weights
                self.outstar_net.train_step(rand_muscle_act, initial_joint_angle, move_dir)
                
                if i % print_every == 0 and (j == 0) and verbose is True:
                    print(f'Shoulder: {pre_pos[0]}, Elbow: {pre_pos[1]}, Wrist: {pre_pos[2]}, Hand: {pre_pos[3]}')
                    
                if i % print_every == 0 and (j == n_babbles - 1) and verbose is True:
                    print(f'Shoulder: {pre_pos[0]}, Elbow: {pre_pos[1]}, Wrist: {pre_pos[2]}, Hand: {pre_pos[3]}')

                if visualize:
                    arm_plot.update(pre_pos)

        print("Training finished!")

    def get_joint_angles(self):
        angles = np.zeros(len(self.joints))

        for i in range(len(self.joints)):
            angles[i] = self.joints[i].get_angle()

        return angles
    def test(self, all_target_pos, target_dist_tol=2.0, visualize=True, verbose=True, train=False):
        '''Have the arm perform reaching movements to intercept each of the targets one-by-one. Runs the Outstar network
        in prediction mode: given the sensed state of the arm by the motorneurons (source), compute the muscle activations
        (sink) needed to get the hand closer to intercepting the next target.

        Parameters:
        -----------
        all_target_pos: ndarray. shape=(num_targets, 2). The (x, y) coordinates of the targets in the workspace that the
        for which the arm should reach.
        target_dist_tol: float. How close does the hand position have to be to the target for it to count as a successful
            interception? This distance could be in whatever metric you like (L1, Euclidean, etc.).
        visualize: bool. Whether to create an animated plot showing the arm in the workspace that arm reaches for targets.
        verbose: bool. Whether to print out information during the target reaching process.

        TODO:
        1. Make sure that the arm starts in the default posture (i.e. it is not in the last position after training).
        2. Make one target the reaching goal one at a time.
        3. Keep trying to reach for the current target until the distance between the end effector (hand) and current
        target is bigger than the allowed distance tolerance.
        4. Run one prediction step in the Outstar network given the current joint angles and the angle between
        the hand's current position and the current target position.
        5. Update the position (joint angles) of the arm with the predicted muscle activations. To get smoother arm
        movements, set the `rot_rate` parameter of `update_angle` to the output of the self.get_dist_scaling method. See
        docstring for what to pass in.
        '''
        # Keep me at the start of the method
        if visualize:
            arm_plot = ArmPlot()

        num_targets = all_target_pos.shape[0]
        
        # Arm starts in the default postion
        self.reset_joint_angles()
        
        print("Start reaching...")
        
        # make one target the reaching goal at a time
        for i in range(num_targets):
            curr_target = all_target_pos[i, :]
            if verbose:
                print(f'{i+1}/{num_targets}, currently reaching for {curr_target}')
                
            pre_pos = self.get_joint_positions()[-1]
                
            # distance between hand and target
            initial_dist = self.get_dist_between_targets(pre_pos, curr_target)
            curr_dist = initial_dist
            num_trial = 1
            while curr_dist > target_dist_tol:
                num_trial =  num_trial + 1
                
                # print(f"{num_trial}: current distance from hand to target: {curr_dist:.2f}")
                
                pre_pos = self.get_joint_positions()[-1]
                initial_joint_angle = self.get_joint_angles()
                
                if visualize:
                    arm_plot.update(self.get_joint_positions(), all_target_pos, i)
                    
                move_dir = self.get_movement_dir(curr_target, pre_pos)

                # predict muscle act given current joint angles and angle between hand and target
                pred_muscle_act = self.outstar_net.predict_step(self.get_joint_angles(), move_dir)

                # update joint angles of the arm with the predicted muscle activations
                for k in range(len(self.joints)):
                    self.joints[k].update_angle(pred_muscle_act[k*2: k*2+2])
                                                # angle_step=self.get_dist_scaling(initial_dist, curr_dist))
                    # print(self.get_joint_angles())
                
                post_pos = self.get_joint_positions()[-1]
                
                # update current distance
                curr_dist = self.get_dist_between_targets(post_pos, curr_target)
                move_dir_2 = self.get_movement_dir(post_pos, pre_pos)                
                
                # update weights
                if train:
                    self.outstar_net.train_step(pred_muscle_act, initial_joint_angle, move_dir_2)
            
            if verbose:
                print(f'Number of times tried: {num_trial} \nDistance from hand to target: {curr_dist:.2f}\n')
            
            if visualize:
                arm_plot.update(self.get_joint_positions(), all_target_pos, i, 1.0)

        print("Finished all reaching")

    def get_dist_between_targets(self, pos1, pos2):
        return np.sqrt(np.sum((pos2-pos1)**2))

    def get_dist_scaling(self, initial_target_dist, curr_target_dist, alpha=0.9, beta=0.002, gamma=0.1):
        '''Scaling function ensuring approximately bellshaped velocity curves characteristic of
        real reaching movements

        Parameters:
        -----------
        initial_target_dist: float. The initial distance between the hand and the current target before starting to reach
            for the current target (i.e. estimate of total distance expected to travel).
        curr_target_dist: float. Current distance between the hand and the current target.
        alpha: float. Hyperparameter that adjusts the scaling function.
        beta: float. Hyperparameter that adjusts the scaling function.
        gamma: float. Hyperparameter that adjusts the scaling function.

        Returns:
        -----------
        float. Rate to scale the differnce in agonist/antagonist muscle activations when updating each joint angle.

        (This method is provided to you and should not require modification)
        '''
        # Determine ratio between current distance from target and initial distance to target
        dist_ratio = curr_target_dist / initial_target_dist

        if dist_ratio <= 0.5:
            return ((alpha * (2*dist_ratio - 2)**4) / (beta + (2*dist_ratio - 2)**4)) + gamma
        else:
            return ((alpha * (2*dist_ratio)**4) / (beta + (2*dist_ratio)**4)) + gamma

    def plot(self, all_target_pos_xy=None, fig_sz=(8, 8), fig_num=0):
        '''Plots the arm in its current position within the workspace as well as any targets.

        Parameters:
        -----------
        all_target_pos_xy: ndarray. shape=(N_targets, 2). The (x, y) coordinates of targets that will be reached for.
            Not used during training.
        fig_sz: tuple of floats. len=2. Height and width of the figure in matplotlib `Figure` format.
        fig_num: int. Unique identifer for the arm plot. This allows matplotlib to locate the figure across method calls.
            This should not need to be changed.

        Returns:
        -----------
        Figure. The matplotlib figure object used to draw the arm in the workspace.

        (This is provided to you and should not require modification)
        '''
        if self.curr_plot is None:
            self.curr_plot = ArmPlot(fig_sz=fig_sz, fig_num=fig_num)

        # Get current arm position
        arm_pos = self.get_joint_positions()

        # Update the plot
        self.curr_plot.update(arm_pos, all_target_pos_xy=all_target_pos_xy)

        # Return the figure
        return self.curr_plot.get_figure()
