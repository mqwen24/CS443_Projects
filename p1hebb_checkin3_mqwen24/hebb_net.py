'''hebb_net.py
Bio-inspired neural network that implements the Hebbian learning rule and competition among neurons in the network
MUQING WEN, ZHOUYI QIAN
CS443: Bio-Inspired Machine Learning
Project 1: Hebbian Learning

NOTE: This network should be implemented in Numpy rather than TensorFlow
'''
import numpy as np
import matplotlib.pyplot as plt
from viz import draw_grid_image


class HebbNet:
    '''Single layer bio-inspired neural network in which neurons compete with each other and learning occurs via a
    variant of Hebbian learning rule (Oja's Rule).
    '''

    def __init__(self, num_features, num_neurons, wt_minmax=(0., 1.), kth_place_inhibited=6, inhib_value=0.4,
                 load_wts=False, saved_wts_path='export/wts.npy'):
        '''Hebbian network constructor

        Parameters:
        -----------
        num_features: int. Num input features (M)
        num_neurons: int. Num of neurons in the network (H)
        wt_minmax: tuple of 2 floats. wt_minmax[0] is the min possible wt value when wts initialized. wt_minmax[1] is
            the max possible wt value when wts initialized.
        kth_place_inhibited: int. In the neural competition that occurs when processing each data sample, the neuron
            that achieves the kth highest net_in value ("neuron came in kth place") is inhibited, which means
            the kth place neuron gets netAct value of `-inhib_value`.
        inhib_value: float. Non-negative number (≥0) that represents the netAct value assigned to the inhibited neuron
            (with the kth highest netAct value).
        load_wts: bool. Whether to load weights previously saved off by the network after successful training.
        saved_wts_path: str. Path from the working project directory where the weights previously saved by the net are
            stored. Used if `load_wts` is True.

        TODO:
        - Create instance variables for the parameters
        - Initialize the wts.
            - If loading wts, set the wts by loading the previously saved .npy wt file.
            - Otherwise, create uniform random weights between the range `wt_minmax`. shape=(M, H).
        '''
        self.M = num_features
        self.H = num_neurons
        self.wt_min = wt_minmax[0]
        self.wt_max = wt_minmax[1]
        self.k = kth_place_inhibited - 1
        self.inhib_value = inhib_value
        self.load_wts = load_wts
        self.saved_wts_path = saved_wts_path

        if load_wts:
            self.wts = np.load(saved_wts_path)

        else:
            self.wts = np.random.uniform(low=self.wt_min, high=self.wt_max, size=(self.M, self.H))

    def get_wts(self):
        '''Returns the Hebbian network wts'''
        return self.wts

    def set_wts(self, wts):
        '''Replaces the Hebbian network weights with `wts` passed in as a parameter.

        Parameters:
        -----------
        wts: ndarray. shape=(M, H). New Hebbian network weights.
        '''
        self.wts = wts

    def net_in(self, x):
        '''Computes the Hebbian network Dense net_in

        Parameters:
        -----------
        x: ndarray. shape=(B, M)

        Returns:
        -----------
        netIn: ndarray. shape=(B, H)
        '''
        return x @ self.wts

    def net_act(self, net_in):
        '''Compute the Hebbian network activation, which is a function that reflects competition among the neurons
        based on their net_in values.

        NetAct (also see notebook):
        - 1 for neuron that achieves highest net_in value to sample i
        - -delta for neuron that achieves kth highest net_in value to sample i
        - 0 for all other neurons

        Parameters:
        -----------
        net_in: ndarray. shape=(B, H)

        Returns:
        -----------
        netAct: ndarray. shape=(B, H)

        Tips:
        - It might be helpful to think of competition as an assignment operation.
        - Remember arange indexing? It might be useful depending on your implementation strategy.
        - No loops should be needed.
        '''
        B, H = net_in.shape
        indx = np.arange(B)

        sorted_indx = np.argsort(net_in, axis=1)
        sorted_indx = np.flip(sorted_indx, axis=1)
        # print("sorted index: ", sorted_indx)
        max_indx = sorted_indx[:, 0]
        # print("max index: ", max_indx)
        kth_indx = sorted_indx[:, self.k]
        # print("kth index: ", kth_indx)
        net_act = np.zeros((B, H))
        net_act[indx, kth_indx] = -self.inhib_value
        net_act[indx, max_indx] = 1

        return net_act

    def update_wts(self, x, net_in, net_act, lr, eps=1e-10):
        '''Update the Hebbian network wts according to a modified Hebbian learning rule (competitive Oja's rule).
        After computing the weight change based on the current mini-batch, the weight changes (gradients) are normalized
        by the largest gradient (in absolute value). This has the effect of making the largest weight change equal in
        absolute magnitude to the learning rate `lr`. See notebook for equations.

        Parameters:
        -----------
        net_in: ndarray. shape=(B, H)
        net_act: ndarray. shape=(B, H)
        lr: float. Learning rate hyperparameter
        eps: float. Small non-negative number used in the wt normalization step to prevent possible division by 0.

        Tips:
        - This is definitely a scenario where you should the shapes of everything to guide you through and decide on the
        appropriate operation (elementwise multiplication vs matrix multiplication).
        '''
        d_wts = (np.transpose(x) @ net_act) - (self.wts * np.sum(net_in * net_act, axis=0))

        d_wts = d_wts / (np.max(np.abs(d_wts)) + eps)

        self.wts = self.wts + lr * d_wts

    def fit(self, x, n_epochs=1, mini_batch_sz=128, lr=2e-2, plot_wts_live=False, fig_sz=(9, 9), n_wts_plotted=(10, 10),
            print_every=1, save_wts=True):
        '''Trains the Hebbian network on the training samples `x` using unsupervised Hebbian learning (no y classes required!).

        Parameters:
        -----------
        x: ndarray. shape=(N, M). Data samples.
        n_epochs: int. Number of epochs to train the network.
        mini_batch_sz: int. Mini-batch size used when training the Hebbian network.
        lr: float. Learning rate used with Hebbian weight update rule
        plot_wts_live: bool. Whether to plot the weights and update throughout training every `print_every` epochs.
        save_wts: bool. Whether to save the Hebbian network wts (to self.saved_wts_path) after training finishes.

        TODO:
        Very similar workflow to usual:
        - In each epoch setup mini-batch. You can sample with replacement or without replacement (shuffle) between epochs
        (your choice).
        - Compute forward pass for each mini-batch then update the weights.
        - If plotting the wts on the current epoch, update the plot (via `draw_grid_image`) to show the current wts.
        - Print out which epoch we are on `print_every` epochs
        - When training is done, save the wts if `save_wts` is True.
        '''
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
                    draw_grid_image(x=self.wts.T, n_cols=n_wts_plotted[0], n_rows=n_wts_plotted[1], sample_dims=(28, 28, 1), title=f'Net receptive fields (Epoch {num_epochs})')
                    fig.canvas.draw()
                else:
                    print(f'Starting epoch {num_epochs+1}/{n_epochs}')

                num_epochs = num_epochs + 1

            num_iter = num_iter + 1

        print("Training finished!")

        if save_wts:
            print("Saving weights ...")
            with open(self.saved_wts_path, 'wb') as file:
                np.save(file, self.wts)

