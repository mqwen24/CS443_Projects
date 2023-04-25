'''mlp.py
Constructs, trains, tests 2 layer multilayer layer perceptron networks
YOUR NAMES HERE
CS343: Neural Networks
Project 2: Multilayer Perceptrons
'''
import numpy as np


class MLP:
    '''
    MLP is a class for multilayer perceptron network.

    The structure of our MLP will be:

    Input layer (X units) ->
    Hidden layer (Y units) with Rectified Linear activation (ReLu) ->
    Output layer (Z units) with softmax activation

    Due to the softmax, activation of output neuron i represents the probability that
    the current input sample belongs to class i.

    NOTE: We will keep our bias weights separate from our feature weights to simplify computations.
    '''
    def __init__(self, num_input_units, num_hidden_units, num_output_units):
        '''Constructor to build the model structure and intialize the weights. There are 3 layers:
        input layer, hidden layer, and output layer. Since the input layer represents each input
        sample, we don't learn weights for it.

        Parameters:
        -----------
        num_input_units: int. Num input features
        num_hidden_units: int. Num hidden units
        num_output_units: int. Num output units. Equal to # data classes.
        '''
        self.num_input_units = num_input_units
        self.num_hidden_units = num_hidden_units
        self.num_output_units = num_output_units

        self.y_wts = None
        self.y_b = None
        self.z_wts = None
        self.z_b = None
        self.initialize_wts(num_input_units, num_hidden_units, num_output_units)

    def get_y_wts(self):
        '''Returns a copy of the hidden layer wts'''
        return self.y_wts.copy()

    def initialize_wts(self, M, H, C, std=0.1):
        ''' Randomly initialize the hidden and output layer weights and bias term

        Parameters:
        -----------
        M: int. Num input features
        H: int. Num hidden units
        C: int. Num output units. Equal to # data classes.
        std: float. Standard deviation of the normal distribution of weights

        Returns:
        -----------
        No return

        TODO:
        - Initialize self.y_wts, self.y_b and self.z_wts, self.z_b
        with the appropriate size according to the normal distribution with standard deviation
        `std` and mean of 0.
          - For wt shapes, they should be be equal to (#prev layer units, #associated layer units)
            for example: self.y_wts has shape (M, H)
          - For bias shapes, they should equal the number of units in the associated layer.
            for example: self.y_b has shape (H,)
        '''
        # keep the random seed for debugging/test code purposes
        np.random.seed(0)

        self.y_wts = np.random.normal(loc=0, scale=std, size=(M, H))
        self.y_b = np.random.normal(loc=0, scale=std, size=(H,))

        self.z_wts = np.random.normal(loc=0, scale=std, size=(H, C))
        self.z_b = np.random.normal(loc=0, scale=std, size=(C,))
        

    def accuracy(self, y, y_pred):
        ''' Computes the accuracy of classified samples. Proportion correct

        Parameters:
        -----------
        y: ndarray. int-coded true classes. shape=(Num samps,)
        y_pred: ndarray. int-coded predicted classes by the network. shape=(Num samps,)

        Returns:
        -----------
        float. accuracy in range [0, 1]
        '''
        return np.sum(y == y_pred) / y.shape[0]

    def one_hot(self, y, num_classes):
        '''One-hot codes the output classes for a mini-batch

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
        num_classes: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: One-hot coded class assignments.
            e.g. if y = [0, 2, 1] and num_classes = 4 we have:
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
        '''
        one_hot = np.zeros((y.shape[0], num_classes))
        x_cor = np.arange(y.shape[0])
        one_hot[x_cor, y] = 1
        return one_hot

    def predict(self, features):
        ''' Predicts the int-coded class value for network inputs ('features').

        NOTE: Loops of any kind are NOT ALLOWED in this method!

        Parameters:
        -----------
        features: ndarray. shape=(mini-batch size, num features)

        Returns:
        -----------
        y_pred: ndarray. shape=(mini-batch size,).
            This is the int-coded predicted class values for the inputs passed in.
            Note: You can figure out the predicted class assignments without applying the
            softmax net activation function — it will not affect the most active neuron.
        '''
        # we would use relu function to process the hidden activation
        hidden = features @ self.y_wts + self.y_b
        hidden[hidden <= 0] = 0

        # we would use the sigmoid function to process the output activation
        output = hidden @ self.z_wts + self.z_b

        y_pred = np.argmax(output, axis=1)
        return y_pred

    def forward(self, features, y, reg=0):
        '''
        Performs a forward pass of the net (input -> hidden -> output).
        This should start with the features and progate the activity
        to the output layer, ending with the cross-entropy loss computation.
        Don't forget to add the regularization to the loss!

        NOTE: Implement all forward computations within this function
        (don't divide up into separate functions for net_in, net_act). Doing this all-in-one method
        is not good design, but as you will discover, having the
        forward computations (y_net_in, y_net_act, etc.) easily accessible in one place makes the
        backward pass a lot easier to track during implementation. In future projects, we will
        rely on better OO design.

        NOTE: Loops of any kind are NOT ALLOWED in this method!

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size N, Num features M)
        y: ndarray. int coded class labels. shape=(mini-batch-size N,)
        reg: float. regularization strength.

        Returns:
        -----------
        y_net_in: ndarray. shape=(N, H). hidden layer "net in"
        y_net_act: ndarray. shape=(N, H). hidden layer activation
        z_net_in: ndarray. shape=(N, C). output layer "net in"
        z_net_act: ndarray. shape=(N, C). output layer activation
        loss: float. REGULARIZED loss derived from output layer, averaged over all input samples

        NOTE:
        - To regularize loss for multiple layers, you add the usual regularization to the loss
          from each set of weights (i.e. 2 in this case).
        '''
        # we would use relu function to process the hidden activation
        y_net_in = features @ self.y_wts + self.y_b

        y_net_act = np.copy(y_net_in)
        y_net_act[y_net_act <= 0] = 0

        z_net_in = y_net_act @ self.z_wts + self.z_b
        z_net_act = np.copy(z_net_in)
        
        log_a = -np.max(z_net_act, axis=1, keepdims=True)
        exp_net_in = np.exp(z_net_act + log_a)
        sum_row = np.sum(exp_net_in, axis=1, keepdims=True)
        z_net_act = exp_net_in / sum_row

        num_sample = y.shape[0]
        x_cor = np.arange(num_sample)
        net_act_correct = z_net_act[x_cor, y]

        loss = -np.sum(np.log(net_act_correct), axis=0) / num_sample + reg * np.sum(np.square(self.z_wts)) / 2 + reg * np.sum(np.square(self.y_wts)) / 2

        return y_net_in, y_net_act, z_net_in, z_net_act, loss

    def backward(self, features, y, y_net_in, y_net_act, z_net_in, z_net_act, reg=0):
        '''
        Performs a backward pass (output -> hidden -> input) during training to update the
        weights. This function implements the backpropogation algorithm.

        This should start with the loss and progate the activity
        backwards through the net to the input-hidden weights.

        I added dz_net_act for you to start with, which is your cross-entropy loss gradient.
        Next, tackle dz_net_in, dz_wts and so on.

        I suggest numbering your forward flow equations and process each for
        relevant gradients in reverse order until you hit the first set of weights.

        Don't forget to backpropogate the regularization to the weights!
        (I suggest worrying about this last)

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size, Num features)
        y: ndarray. int coded class labels. shape=(mini-batch-size,)
        y_net_in: ndarray. shape=(N, H). hidden layer "net in"
        y_net_act: ndarray. shape=(N, H). hidden layer activation
        z_net_in: ndarray. shape=(N, C). output layer "net in"
        z_net_act: ndarray. shape=(N, C). output layer activation
        reg: float. regularization strength.

        Returns:
        -----------
        dy_wts, dy_b, dz_wts, dz_b: The following backwards gradients
        (1) hidden wts, (2) hidden bias, (3) output weights, (4) output bias
        Shapes should match the respective wt/bias instance vars.

        NOTE:
        - Regularize each layer's weights like usual.
        '''
        num_samp = len(z_net_act)
        
        z_one_hot = self.one_hot(y, self.num_output_units)
        y_one_hot = self.one_hot(y, self.num_hidden_units)
        
        # 5 loss WRT netAct
        dz_net_act = -1/(num_samp * z_net_act)

        # calculate gradient from output layer

        dz_net_act_dz_net_in = z_net_act *(z_one_hot - z_net_act)

        dz_net_in = dz_net_act * dz_net_act_dz_net_in

        dz_wts = (dz_net_in.T @ y_net_act).T + reg * self.z_wts

        dz_b = np.sum(dz_net_in, 0)

        # calculate gradient from input layer
        dy_net_act = dz_net_in @ self.z_wts.T

        dy_net_act_dy_net_in = np.ones((dy_net_act.shape))
        dy_net_act_dy_net_in[y_net_act==0] = 0
        dy_net_in = dy_net_act * dy_net_act_dy_net_in

        dy_wts = (dy_net_in.T@features).T+ reg * self.y_wts

        dy_b = np.sum(dy_net_in, 0)

        return dy_wts, dy_b, dz_wts, dz_b

        

    def fit(self, features, y, x_validation, y_validation,
            resume_training=False, n_epochs=500, lr=0.0001, mini_batch_sz=256, reg=0, verbose=2,
            print_every=100):
        ''' Trains the network to data in `features` belonging to the int-coded classes `y`.
        Implements stochastic mini-batch gradient descent

        Parameters:
        -----------
        features: ndarray. shape=(Num samples N, num features).
            Features over N inputs.
        y: ndarray.
            int-coded class assignments of training samples. 0,...,numClasses-1
        x_validation: ndarray. shape=(Num samples in validation set, num features).
            This is used for computing/printing the accuracy on the validation set at the end of each
            epoch.
        y_validation: ndarray.
            int-coded class assignments of validation samples. 0,...,numClasses-1
        resume_training: bool.
            False: we clear the network weights and do fresh training
            True: we continue training based on the previous state of the network.
                This is handy if runs of training get interupted and you'd like to continue later.
        n_epochs: int.
            Number of training epochs
        lr: float.
            Learning rate
        mini_batch_sz: int.
            Batch size per epoch. i.e. How many samples we draw from features to pass through the
            model per training epoch before we do gradient descent and update the wts.
        reg: float.
            Regularization strength used when computing the loss and gradient.
        verbose: int.
            0 means no print outs. Any value > 0 prints Current epoch number and training loss every
            `print_every` (e.g. 100) epochs.
        print_every: int.
            If verbose > 0, print out the training loss and validation accuracy over the last epoch
            every `print_every` epochs.
            Example: If there are 20 epochs and `print_every` = 5 then you print-outs happen on
            on epochs 0, 5, 10, and 15 (or 1, 6, 11, and 16 if counting from 1).

        Returns:
        -----------
        loss_history: Python list of floats.
            Recorded training loss on every epoch for the current mini-batch.
        train_acc_history: Python list of floats.
            Recorded accuracy on every training epoch on the current training mini-batch.
        validation_acc_history: Python list of floats.
            Recorded accuracy on every epoch on the validation set.

        TODO:
        -----------
        The flow of this method should follow the one that you wrote in softmax_layer.py.
        The main differences are:
        1) Remember to update weights and biases for all layers!
        2) At the end of an epoch (calculated from iterations), compute the training and
            validation set accuracy. This is only done once every epoch because "peeking" slows
            down the training.
        3) Add helpful printouts showing important stats like num_epochs, num_iter/epoch, num_iter,
        loss, training and validation accuracy, etc, but only if verbose > 0 and consider `print_every`
        to control the frequency of printouts.
        '''
        num_samps, num_features = features.shape
        num_classes = self.num_output_units
        
        loss_history = []
        train_acc_history = []
        validation_acc_history = []

        if resume_training == False:
            # initialize wts/bias
            self.initialize_wts(num_features, self.num_hidden_units, num_classes, std=0.1)
        
        num_iter = int(n_epochs * (num_samps / mini_batch_sz))

        # print the verbose message:
        if verbose > 0:
            print(
                f'Starting to train network...There will be {n_epochs} epochs and {num_iter} iterations total, {int(num_iter/n_epochs)} iter/epoch.')

        # handling edge cases for the mini_batch_sz
        if mini_batch_sz > num_samps:
            mini_batch_sz = num_samps
        if mini_batch_sz <= 0:
            mini_batch_sz = 1

        # the number of iteration equals to the number of epoch multiply the number of batch there are in one sample
        prev_epoch = -1
        for i in range(num_iter):

            # generate mini batch
            idx = np.random.choice(num_samps, size=(mini_batch_sz,), replace=True, p=None)
            
            mini_batch = features[idx, :]
            y_mini_batch = y[idx]

            y_net_in, y_net_act, z_net_in, z_net_act, loss = self.forward(mini_batch, y_mini_batch, reg)

            # record training loss
            loss_history.append(loss)
            
            dy_wts, dy_b, dz_wts, dz_b = self.backward(mini_batch, y_mini_batch, y_net_in, y_net_act, z_net_in, z_net_act, reg)

            self.y_wts = self.y_wts - lr * dy_wts
            self.y_b = self.y_b - lr * dy_b
            
            self.z_wts = self.z_wts - lr * dz_wts
            self.z_b = self.z_b - lr * dz_b
            
            # compute the training/validation acc after every epoch            
            i_epoch = int(i / (num_samps / mini_batch_sz))
            if i_epoch != prev_epoch:
                prev_epoch = i_epoch
                # compute training accuray
                y_pred_train = self.predict(mini_batch)
                train_acc = self.accuracy(y_mini_batch, y_pred_train)
                train_acc_history.append(train_acc)

                # compute test accuracy
                y_pred_val = self.predict(x_validation)
                validation_acc = self.accuracy(y_validation, y_pred_val)
                validation_acc_history.append(validation_acc)
           
            # if verbose is true, print out loss and cycle count every "print_every" cycles
            
            if (i_epoch % print_every == 0 or i == 0) and verbose > 0:
                print(f' Completed Epoch {i_epoch}/{n_epochs}. Training loss: {loss:.2f}. Validation Accuracy: {validation_acc*100:.2f}%.')

        if verbose > 0:
            print("Finished training!")
            
        return loss_history, train_acc_history, validation_acc_history
