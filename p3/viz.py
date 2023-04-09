'''vis.py
Plotting code
CS443: Bio-Inspired Machine Learning
Oliver W. Layton
Project 3: Outstar learning
'''
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

from IPython import display

class ArmPlot:
    '''Makes a dynamic, animated plot of the arm and targets in the workspace.'''
    def __init__(self, fig_sz=(8, 8), fig_num=0):
        '''Creates the Figure object used to plot the arm and sets up overlapping Cartesian and polar axes.
        '''
        self.fig_sz = fig_sz
        self.fig_num = fig_num

        # Make the figure
        self.fig = plt.figure(num=self.fig_num, figsize=self.fig_sz)

        # Plot bounding box
        rect = [0.1, 0.1, 0.8, 0.8]  # [left, bottom, width, height]
        # Cartesian axis (used for everything)
        self.fig.add_axes(rect)
        # Polar axis (used to get a sense of radial dist and angles)
        self.fig.add_axes(rect, polar=True, frameon=False)

    def get_figure(self):
        '''Returns a reference to the matplotlib Figure object used to plot the arm and workspace.'''
        return self.fig

    def update(self, curr_arm_pos, all_target_pos_xy=None):
        '''Plot the current state of the arm in the workspace.

        Parameters:
        -----------
        curr_arm_pos: ndarray. shape=(4, 2). The (x, y) coordinates of the shoulder joint, elbow joint, wrist joint, and
            end effector (hand), in that order.
        all_target_pos: ndarray. shape=(num_targets, 2). The (x, y) coordinates of all the targets in the workspace.
        '''
        # Get and name the Cartesian and polar axes from fig
        axes = self.fig.axes
        ax_c, ax_p = axes

        # Clear arm plotted on previous time step
        ax_c.clear()
        ax_p.clear()

        # Setup up polar plot component
        ax_p.set_ylim(0, 100)
        ax_p.set_yticks(np.arange(0, 120, 20))
        ax_p.set_xticks([])
        ax_p.set_theta_direction(-1)

        # Setup up cartesian plot component
        ax_c.set_ylim(-100, 100)
        ax_c.set_yticks([-100, -50, 50, 100])
        ax_c.set_xlim(-100, 100)
        ax_c.set_xticks([-100, -50, 50, 100])
        ax_c.grid(False)

        # Setup arm visualization
        ax_c.add_artist(Ellipse((-30, 0), 70, 20, color='r', fill=False))  # shoulder
        ax_c.add_artist(Circle((-30, 0), 15, color='r', fill=False))  # head
        ax_c.add_artist(Circle((-37, 13), 2, color='r', fill=False))  # left eye
        ax_c.add_artist(Circle((-23, 13), 2, color='r', fill=False))  # right eye

        # Draw arm current state
        ax_c.plot(curr_arm_pos[:, 0], curr_arm_pos[:, 1], linewidth=6)  # hand
        ax_c.plot(curr_arm_pos[-1, 0], curr_arm_pos[-1, 1], '<', markersize=20, label='Hand (end effector)')

        # Only show targets if (x, y) positions passed in
        if all_target_pos_xy is not None:
            ax_c.plot(all_target_pos_xy[:, 0], all_target_pos_xy[:, 1], 'o', markersize=20, label='Targets')

        ax_c.legend(loc='lower right')

        self.fig.canvas.draw()
        # time.sleep(0.05)
        
        display.clear_output(wait=True)
        display.display(self.fig)
        time.sleep(0.05)


def plot_wts(wts, wts_norm, src_wt_ind=0, feat_vec=None):
    '''Plots the sink and wt values over time in a 2x1 plot configuration.

    Parameters:
    -----------
    snk_acts: ndarray. shape=(n_time_steps, n_sink_cells). Activation of sink cells over time.
    wts: ndarray. shape=(n_time_steps, n_source_cells, n_sink_cells). Wt values in Outstar network over time.
    t_max_snk: int. Limit the period of time over which the SINK activation is plotted to the 1st `t_max` secs.
    src_wt_ind: int. When plotting the weights, only show weights connecting the source neuron at this index and all
        sink cells.
    dt: float. Number of secs per time step.
    ylim_snk: tuple. (min, max) y axis bounds to use when plotting the sink activations
    ylim_wts: tuple. (min, max) y axis bounds to use when plotting the wts
    '''
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 5))

    ylabel = ['Wts', 'Wts (Normalized)']
    y = [wts, wts_norm]

    for i in range(len(axes)):
        curr_wts = y[i]

        if i == 1 and feat_vec is not None:
            for th in feat_vec:
                axes[i].axhline(th, linestyle='--', color='k', alpha=0.3)

        for j in range(curr_wts.shape[2]):
            axes[i].plot(curr_wts[:, src_wt_ind, j], label='w_'+str(j+1))
        axes[i].set_ylabel(ylabel[i])
        axes[i].legend()

    axes[i].set_xlabel('Time step (training iteration)')
    plt.show()
