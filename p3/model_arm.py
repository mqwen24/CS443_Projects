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
        pass

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

            # # Execute me everytime a babble happens.
            # if visualize:
            #     arm_plot.update(positions_prev)
        
        N, M = x.shape
        # print the verbose message:
        print(f'Starting to train network ....')

        # handling edge cases for the mini_batch_sz
        if mini_batch_sz > N:
            mini_batch_sz = N
        if mini_batch_sz <= 0:
            mini_batch_sz = 1

        num_iter = 0
        num_epochs = 0

        if plot_wts_live:
            fig = plt.figure(figsize=fig_sz)

        while num_epochs < n_epochs:
            # generate mini batch
            idx = np.random.choice(N, size=(mini_batch_sz,), replace=True, p=None)
            x_mini_batch = x[idx, :]

            # forward pass
            net_in = self.net_in(x_mini_batch)
            net_act = self.net_act(net_in)

            # weight update
            self.update_wts(x_mini_batch, net_in, net_act, lr)

            if (num_iter == 0) or (int(num_iter / (N / mini_batch_sz)) == num_epochs + 1):
                print("Epoch: ", num_epochs)
                # should go in training loop
                if plot_wts_live and num_epochs % print_every == 0:
                    # print(self.wts.T.shape, n_wts_plotted[0], n_wts_plotted[1])
                    
                    # file_name = "epoch_" + str(num_epochs) + ".jpg"
                    # file_path = os.path.join("plot_wts_live/", file_name)
                    # img = self.np2image(self.wts)
                    # img.save(file_path)
                    
                    draw_grid_image(x=self.wts.T, n_cols=n_wts_plotted[0], n_rows=n_wts_plotted[1], sample_dims=(28, 28, 1), title=f'Net receptive fields (Epoch {num_epochs})')
                    fig.canvas.draw()

                num_epochs = num_epochs + 1

            num_iter = num_iter + 1

        print("Training finished!")

    def test(self, all_target_pos, target_dist_tol=2.0, visualize=True, verbose=True):
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

        # # Plot
        # # Execute me every time the arm moves
        # if visualize:
        #     arm_plot.update(arm_pos, all_target_pos)

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
