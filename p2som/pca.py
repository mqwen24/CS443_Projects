'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
YOUR NAME HERE
CS 251 Data Analysis Visualization
Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data=None):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset.

        (No changes should be needed)
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        self.arr = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs
    
    def get_A(self):
        return self.A

    def get_arr(self):
        return self.arr
    
    def get_normalized(self):
        return self.normalized

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`

        NOTE: You should do this wihout any loops
        NOTE: np.cov is off-limits here — compute it from "scratch"!
        '''
        centered = self.center_data(data)
        scalar = 1/(len(centered)-1)
        cov = scalar*(centered.T@centered)
        return cov
    
    def test_print(self):
        print("test")

    def normalize_together(self, data):
        min_s = data.min()  # s = scalar
        max_s = data.max()
        range_s = max_s - min_s
    
        normalized = (data-min_s)/range_s
        self.normalized = True
        return normalized

    def normalize_separately(self, data):  
        min_arr = data.min(0)
        max_arr = data.max(0)
        range_arr = max_arr - min_arr
    
        normalized = (data-min_arr)/range_arr
        self.normalized = True
        return normalized

    def center_data(self, data):
        col_mean = data.mean(0)
        centered = data - col_mean

        return centered

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''
        sum_e_vals = e_vals.sum()
        prop_var = e_vals/sum_e_vals
        prop_var = list(prop_var)
        return prop_var

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''
        cum_list = []
        cum_list.append(prop_var[0])
        i = 1
        while i < len(prop_var):
            cum_i = cum_list[i-1]+prop_var[i]
            cum_list.append(cum_i)
            i = i+1
        return cum_list

    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        TODO:
        - Select the relevant data (corresponding to `vars`) from the data pandas DataFrame
        then convert to numpy ndarray for forthcoming calculations.
        - If `normalize` is True, normalize the selected data so that each variable (column)
        ranges from 0 to 1 (i.e. normalize based on the dynamic range of each variable).
            - Before normalizing, create instance variables containing information that would be
            needed to "undo" or reverse the normalization on the selected data.
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        '''
        self.vars = vars
        self.A = np.array(self.data[self.vars])
        self.arr = self.A.copy()

        if normalize==True:
            self.A = self.normalize_separately(self.A)
        
        cov = self.covariance_matrix(self.A)
        #print(cov)
        e_vals, e_vecs = np.linalg.eig(cov)
        idx = e_vals.argsort()[::-1]
        self.e_vals = e_vals[idx]
        self.e_vecs = e_vecs[:,idx]

        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)
        
    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''
        if num_pcs_to_keep == None:
            cum_var = self.cum_var
        else:
            cum_var = self.cum_var[:num_pcs_to_keep]
        n = len(cum_var) + 1
        x = range(1, n)
        y = cum_var

        plt.figure(figsize=(7,7))
        plt.title('Elbow Plot')
        plt.xlabel('cummlative variance accounted for')
        plt.ylabel('number of PCs kept')
        plt.scatter(x, y, marker='o')
        plt.plot(x, y)

        return x, y

    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        '''
        r_matrix = self.e_vecs[:, pcs_to_keep]
        centered = self.center_data(self.A)
        self.A_proj = centered@r_matrix
        return self.A_proj

    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars)

        TODO:
        - Project the data on the `top_k` PCs (assume PCA has already been performed).
        - Project this PCA-transformed data back to the original data space
        - If you normalized, remember to rescale the data projected back to the original data space.
        '''
        r_matrix = self.e_vecs[:, :top_k]
        centered = self.center_data(self.A)
        self.A_proj = centered@r_matrix
        
        col_mean = self.arr.mean(0)

        if self.normalized == False:
            A_recon = self.A_proj @ r_matrix.T + col_mean
        else:
            range_arr = self.arr.max(0) - self.arr.min(0)
            A_recon = range_arr*(self.A_proj @ r_matrix.T) + col_mean
        
        return A_recon