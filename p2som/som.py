'''som.py
Self-organizing map implemented in Numpy
MUQING WEN, ZHOUYI QIAN
CS443: Bio-Inspired Machine Learning
Project 2: Word Embeddings and Self-Organizing Maps (SOMs)
'''
import numpy as np


def lin2sub(ind, the_shape):
    '''Utility function that takes a linear index and converts it to subscripts.
    No changes necessary here.

    Parameters:
    ----------
    ind: int. Linear index to convert to subscripts
    the_shape: tuple. Shape of the ndarray from which `ind` was taken.

    Returns:
    ----------
    tuple of subscripts

    Example: ind=2, the_shape=(2,2) -> return (1, 0).
        i.e. [[_, _], [->SUBSCRIPT OF THIS ELEMENT<-, _]]
    '''
    return np.unravel_index(ind, the_shape)


class SOM:
    '''A 2D self-organzing map (SOM) neural network.
    '''
    def __init__(self, map_sz, num_feats):
        '''Creates a new SOM with random weights in range [-1, 1]

        Parameters:
        ----------
        map_sz: tuple of 2 ints. (n_rows, n_cols). Number of units in each dimension of the SOM.
            e.g. map_sz=(9, 10) -> the SOM will have 9x10 units arranged in a grid
            of 9 rows and 10 columns.
        num_feats: int. Number of features in a SINGLE data sample feature vector.

        TODO:
        - Initialize weights (self.wts) to standard normal random values (mu=0, sigma=1)
            Shape=(n_rows, n_cols, num_feats).
        - Normalized the weights so that EACH som unit's weight vector length is 1.
        '''
        self.num_feats = num_feats
        self.n_rows, self.n_cols = map_sz
        
        self.wts = np.random.normal(loc=0.0, scale=1.0, size=(self.n_rows, self.n_cols, self.num_feats))
        
        norm_factor = np.sqrt(np.sum(np.square(self.wts), axis=2, keepdims=True))
        # print(norm_factor.shape)
        
        self.wts = self.wts / norm_factor

    def get_wts(self):
        '''Returns the weight vector.

        No changes necessary here.
        '''
        return self.wts

    def get_bmu(self, input_vector):
        '''Compute the best matching unit (BMU) given an input data vector. THE BMU is the unit with
        the closest weights to the data vector. Uses Euclidean distance (L2 norm) as the distance
        metric.

        Parameters:
        ----------
        input_vector: ndarray. shape=(num_feats,). One data sample vector.

        Returns:
        ----------
        tuple of the BMU (row, col) in the SOM grid.

        NOTE: For efficiency, you may not use any loops.
        '''
        # print(np.sum(np.square(input_vector - self.wts), axis=2))
        dists = np.sqrt(np.sum(np.square(input_vector - self.wts), axis=2))
        # print(dists)
        return np.unravel_index(np.argmin(dists), (self.n_rows, self.n_cols))

    def get_nearest_wts(self, data):
        '''Find the nearest SOM wt vector to each of data sample vectors.

        Parameters:
        ----------
        data: ndarray. shape=(N, num_feats) for N data samples.

        Returns:
        ----------
        ndarray. shape=(N, num_feats). The most similar weight vector for each data sample vector.
        '''
        N, num_feats = data.shape
        
        nearest_wts = np.zeros(shape=(N, num_feats))
        # print(nearest_wts.shape)
        # print(self.wts.shape)
        for i in range(N):
            bmu = self.get_bmu(data[i])
            nearest_wts[i] = self.wts[bmu]
        
        return nearest_wts

    def gaussian(self, bmu_rc, sigma):
        '''Generates a "normalized" 2D Gaussian, where the max value is 1, and is centered on `bmu_rc`.

        Parameters:
        ----------
        bmu_rc: tuple. (row, col) in the SOM grid of current best-matching unit (BMU).
        sigma: float. Standard deviation of the Gaussian at the current training iteration.
            The parameter passed in has already been decayed.

        Returns:
        ----------
        ndarray. shape=(n_rows, n_cols). 2D Gaussian, weighted by the the current learning rate.

        TODO:
        - Evaluate a Gaussian on a 2D grid with shape=(n_rows, n_cols) centered on `bmu_rc`.


        HINT:
        This will likely involve generating 2D grids of (row, col) index values (i.e. positions in
        the 2D grid) in the range [0, ..., n_rows-1, 0, ..., n_cols-1].
            shape of som_grid_cols: (n_rows, n_cols)
            shape of som_grid_rows: (n_rows, n_cols)
        You already solved this problem in Project 0 of CS343 :)
        If you adopt this approach, move the initialization of these row and column matrices to the
        constructor, since generating them everytime you call `gaussian` will slow down your code.

        NOTE: For efficiency, you should not use any for loops.
        '''
        row, col = bmu_rc
        row_grid, col_grid = np.meshgrid(np.arange(self.n_rows), np.arange(self.n_cols), indexing="ij")
        dist_rc =  np.sqrt(np.square(row - row_grid) + np.square(col - col_grid))
        dist2_rc = np.square(dist_rc)
        
        denom = 2* np.square(sigma)
        gaussian = np.exp(-dist2_rc / denom)
        return gaussian

    def update_wts(self, input_vector, bmu_rc, lr, sigma):
        '''Applies the SOM update rule to change the BMU (and neighboring units') weights,
        bringing them all closer to the data vector (cooperative learning).

        Parameters:
        ----------
        input_vector: ndarray. shape=(num_feats,). One data sample.
        bmu_rc: tuple. BMU (x,y) position in the SOM grid.
        lr: float. Current learning rate during learning.
        sigma: float. Current standard deviation of Gaussian neighborhood in which units cooperate
            during learning.

        NOTE: For efficiency, you should not use any for loops.
        '''
        gaussian = self.gaussian(bmu_rc, sigma)
        gaussian = np.expand_dims(gaussian, axis=2)
        gap = input_vector - self.wts
        self.wts = self.wts + lr* gaussian * gap

    def decay_param(self, hyperparam, rate):
        '''Takes a hyperparameter (e.g. lr, sigma) and applies a time-dependent decay function.

        Parameters:
        ----------
        hyperparam: float. Current value of a hyperparameter (e.g. lr, sigma) whose value we will decay.
        rate: float. Multiplicative decay factor.

        Returns:
        ----------
        float. The decayed parameter.
        '''
        p = rate* hyperparam
        return p

    def fit(self, x, n_epochs, lr=0.2, lr_decay=0.9, sigma=10, sigma_decay=0.9, print_every=1, verbose=True):
        '''Train the SOM on data

        Parameters:
        ----------
        x: ndarray. shape=(N, num_feats) N training data samples.
        n_epochs: int. Number of training epochs to do
        lr: float. INITIAL learning rate during learning. This will decay with time
            (iteration number). The effective learning rate will only equal this if t=0.
        lr_decay: float. Multiplicative decay rate for learning rate.
        sigma: float. INITIAL standard deviation of Gaussian neighborhood in which units
            cooperate during learning. This will decay with time (iteration number).
            The effective learning rate will only equal this if t=0.
        sigma_decay: float. Multiplicative decay rate for Gaussian neighborhood sigma.
        print_every: int. Print the epoch, lr, sigma, and BMU error every `print_every` epochs.
            NOTE: When first implementing this, ignore "BMU error". You will be computing this soon,
            at which point you can go back and add it.
        verbose: boolean. Whether to print out debug information at various stages of the algorithm.
            NOTE: if verbose=False, nothing should print out during training. Messages indicating start and
        end of training are fine.

        TODO:
        Although this is an unsupervised learning algorithm, the training process is similar to usual:
        - Train SGD-style — one sample at a time. For each epoch you should either sample with replacement or without
        replacement (shuffle) between epochs (your choice).
            - If you shuffle the entire dataset each epoch, be careful not to accidentally modify the original data
            passed in.
        - Within each epoch: compute the BMU of each data sample, update its weights and those of its neighbors, and
        decay the learning rate and Gaussian neighborhood sigma.
        '''
        N = x.shape[0]
        
        # print the verbose message:
        if verbose:
            print(f'Starting to train network ....')
            
        mini_batch_sz = N

        # handle edge cases for the mini_batch_sz
        if mini_batch_sz > N:
            mini_batch_sz = N
        if mini_batch_sz <= 0:
            mini_batch_sz = 1
            
        n_iter = int(n_epochs * (N / mini_batch_sz))
        
        # the number of iteration equals to the number of epoch multiply the number of batch there are in one sample
        i_epoch = -1
        
        for i in range(n_iter):
            # generate mini batch
            idx = np.random.choice(N, size=mini_batch_sz, replace=False)
            x_mini_batch = x[idx, :]
            
            # nearest_wts = self.get_nearest_wts(x_mini_batch)
            
            for j in range(mini_batch_sz):
                bmu = self.get_bmu(x_mini_batch[j])
                self.update_wts(input_vector=x_mini_batch[j], bmu_rc=bmu, lr=lr, sigma=sigma)
            
            
            decay_rate = np.exp(-i / (n_iter/2))
            lr = self.decay_param(lr, lr_decay)
            sigma = self.decay_param(sigma, sigma_decay)
            
            if int(i / (N / mini_batch_sz)) > i_epoch:
                
                i_epoch = i_epoch + 1

                if i_epoch % print_every == 0 and verbose:
                    print(
                        f'Epoch {i_epoch}/{n_epochs}, Learning rate: {lr:.4f}%, Sigma: {sigma:.4f}%.')

                
    def error(self, data):
        '''Computes the quantization error: average error incurred by approximating all data vectors
        with the weight vector of the BMU.

        Parameters:
        ----------
        data: ndarray. shape=(N, num_feats) for N data samples.

        Returns:
        ----------
        float. Average error over N data vectors

        TODO:
        - Progressively average the Euclidean distance between each data vector
        and the BMU weight vector.
        '''
        pass

    def u_matrix(self):
        '''Compute U-matrix, the distance each SOM unit wt and that of its 8 local neighbors.

        Returns:
        ----------
        ndarray. shape=(map_sz, map_sz). Total Euclidan distance between each SOM unit
            and its 8 neighbors.

        NOTE: Remember to normalize the U-matrix so that its range of values always span [0, 1].

        '''
        pass
