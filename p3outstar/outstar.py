'''outstar.py
Outstar neural network and Source, Sink layer classes needed to build the network
CS 443: Bio-Inspired Machine Learning
YOUR NAMES HERE
Project 3: Outstar learning
'''
from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    '''Class with methods that both Source and Sink cells must both implement
    '''

    @abstractmethod
    def get_num_units(self):
        '''Abstract method, leave this blank'''
        pass

    @abstractmethod
    def net_in(self):
        '''Abstract method, leave this blank'''
        pass

    @abstractmethod
    def net_act(self):
        '''Abstract method, leave this blank'''
        pass

    @abstractmethod
    def forward(self):
        '''Abstract method, leave this blank'''
        pass


class Source(Layer):
    '''Any class that inherits from `Source` should serve the role of source cells in the Outstar network and implement
    all the methods in the parent class (`Layer`)

    Leave this class empty.'''
    pass


class Sink(Layer):
    '''Any class that inherits from `Sink` should serve the role of sink cells in the Outstar network and implement
    all the methods in the parent class (`Layer`).'''

    def set_outstar_net(self, new_outstar):
        '''Set the `Outstar` network object associated with the `Sink` cells

        Parameters:
        -----------
        new_outstar: Outstar. New Outstar network associated with this sink layer.
        '''
        self.outstar = new_outstar


class Outstar:
    '''Represents an Outstar Neural Network
    '''

    def __init__(self, source, sink, lr):
        '''Outstar network constructor that:
        - stores the source and sink layers (and other parameters passed in) as instance variables
        - initializes the wts to random values between 0 and 1 (shape: (n_source_cells, n_sink_cells))
        - associates this Outstar network instance with the sink layer object.

        Parameters:
        -----------
        source: Source. Layer of source cells in the Outstar network.
        sink: Source. Layer of sink cells in the Outstar network.
        lr: float. Learning rate used in Outstar wt update rule.
        '''
        self.source = source
        self.sink = sink
        self.lr = lr

        self.wts = np.random.uniform(low=0, high=1, size=(source.get_num_units(), sink.get_num_units()))

        # Associate this outstar net with the sink layer. Adjust code as needed to suit your variable naming conventions
        # Keep this here
        self.sink.set_outstar_net(self)

    def get_wts(self):
        '''Returns the wts.

        Returns:
        -----------
        ndarray. shape=(n_source_cells, n_sink_cells).
        '''
        return self.wts

    def set_wts(self, new_wts):
        '''Replaces the network weights with the passed in parameter. Used by test code.

        Parameters:
        -----------
        new_wts: ndarray. shape=(n_source_cells, n_sink_cells). New weights between source and sink layers.
        '''
        self.wts = new_wts

    def get_source(self):
        '''Returns the Source layer object'''
        return self.source

    def get_sink(self):
        '''Returns the Sink layer object'''
        return self.sink

    def update_wts(self, source_act, sink_act):
        '''Updates the weights (shape=(n_source_cells, n_sink_cells)) based on the Outstar rule
        (see notebook for refresher on the equation).

        Parameters:
        -----------
        source_act: ndarray. shape=(n_source_cells,). net_act of source cell layer at the current time step.
        sink_act: ndarray. shape=(n_sink_cells,). net_act of sink cell layer at the current time step.

        HINT: It may be helpful to add one or more singleton dimensions when doing the computation
        '''
        source_act = source_act[:, np.newaxis]
        self.wts = self.wts + self.lr * (source_act * (sink_act - self.wts))

    def train_step(self, muscle_net_acts, joint_angles_prev, move_dir):
        '''Do all operations in Outstar network on one training step/iteration.

        We pass in muscle acts because we need the same activations that moved the arm from its pre/post babble position
        to do the weight updates
        
        '''
        source_acts = self.source.forward(joint_angles_prev, move_dir)
        # Last step
        self.update_wts(source_acts, muscle_net_acts)

    def predict_step(self, joint_angles, move_dir):
        '''Do all operations in Outstar network on one prediction step/iteration.

        Goal: compute muscle activation from current joint angles (arm state) and the direction of target relative to
        effector.

        Returns:
        -----------
        ndarray. shape=(num_total_muscles,). Activation of each of the (6) muscles computed based on the
            source activation and the Outstar wts.
        '''
        x = self.source.forward(joint_angles, move_dir)
        y = x @ self.wts
        return y
