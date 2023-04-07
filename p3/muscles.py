'''muscles.py
Classes that defines the collection of muscles at all segments of the model arm
CS443: Bio-Inspired Machine Learning
YOUR NAMES HERE
Project 3: Outstar learning
'''
import numpy as np
from outstar import Sink


class Muscles(Sink):
    '''Set of arm muscles organized in agonist/antagonist pairs that move joints. This class represents ALL the muscles
    in the arm (NOT just 1 or 2).

    Sinks in the Outstar neural network.

    Reason why this class represents all the muscles (6): efficiency. Allows us to store muscle activations in one array,
    which allows for vectorization and matrix multiplication.'''
    def __init__(self, n_ag_ant_pairs=3):
        '''Muscles constructor

        Parameters:
        -----------
        n_ag_ant_pairs. int. Number of agonist/antagonist muscle pairs in the arm. Because this is in terms of PAIRS,
            there are twice as many total number of muscles in the arm.
        '''
        self.num_units = 2 * n_ag_ant_pairs
        self.acts = self.randomize_acts()

    def get_num_units(self):
        '''Returns the total number of muscles in the arm.'''
        return self.num_units

    def net_in(self, net_act_source=None):
        '''Computes the net input for all of the muscles.

        Parameters:
        -----------
        net_act_source: None or ndarray. shape=(num_source_neurons,). Muscles get their input from motorneurons
            (source cells).

        Returns:
        -----------
        ndarray. shape=(num_total_muscles,). Net input of each of the (6) muscles. Because muscles cells (sink) only get
            input from motorneurons (source), when there is no source input, the muscle net input is all 0s.

        NOTE:
        -Until instructed otherwise in the notebook, ignore for now what happens when motorneuron activation DOES arrive.
        '''
        pass

    def randomize_acts(self):
        '''Generates one random activation in each agonist/antagonist muscle pair.

        Muscle activation array is organized as:
            [ag1, ant1, ag2, ant2, ag3, ant3]

        Example of randomly generated muscle activations that IS valid: [0.1, 0, 0.31, 0, 0, 0.53]
        Example of randomly generated muscle activations that IS valid: [0, 0.2, 0.13, 0, 0.45, 0]
        Example of randomly generated muscle activations IS NOT valid: [0.1, 0.2, 0.31, 0, 0, 0.53]

        Returns:
        -----------
        ndarray. shape=(num_total_muscles,). Array of random activations for each of the (6) muscles. Only one random
        activation in each agonist/antagonist muscle pair. Each nonzero activation is a random number between 0 and 1.
        '''
        num_muscles = self.num_units
        num_pairs = int(self.num_units / 2)
        
        rand_nums = np.random.uniform(low=0.0, high=1.0, size=(num_pairs,))
        # print(rand_nums)
        
        rand_pos = np.array([])
        for i in range(num_pairs):
            pos = np.random.randint(low=2*i, high=2*i+2, size=(1,))
            rand_pos = np.hstack([rand_pos, pos])
        
        #pos1 = np.random.randint(low=0, high=2, size=(1,))
        #pos2 = np.random.randint(low=2, high=4, size=(1,))
        #pos3 = np.random.randint(low=4, high=6, size=(1,))
        #rand_pos = np.hstack([pos1, pos2, pos3])
        
        rand_pos = rand_pos.astype(int)
        # print(rand_pos)
        
        rand_acts = np.zeros(shape=(num_muscles,))
        rand_acts[rand_pos] = rand_nums
        #print(rand_acts)
        
        return rand_acts
        

    def net_act(self, net_in, net_act_source=None):
        '''Computes the activation for all of the muscles.

        Parameters:
        -----------
        net_in: ndarray. shape=(num_muscles,). Net input of the muscles computed by self.net_in.
        net_act_source: None or ndarray. shape=(num_source_neurons,). Activations from the source cells (motorneurons).

        Returns:
        -----------
        ndarray. shape=(num_total_muscles,). Activation of each of the (6) muscles. When there is no input from
            motorneurons the muscles should generate one random activation per agonist/antagonist muscle pair.

        NOTE:
        -Until instructed otherwise in the notebook, ignore for now what happens when motorneuron activation DOES arrive.
        '''
        pass

    def forward(self, net_act_source=None):
        '''Do forward pass on sink neurons

        Parameters:
        -----------
        net_act_source: ndarray or None. shape=(n_source_cells,). net_act of source cell layer at the current time step.

        Returns:
        -----------
        ndarray. shape=(num_total_muscles,). Activation of each of the (6) muscles.
        '''
        if net_act_source is None:
            self.acts = self.randomize_acts()
        
        return self.acts
