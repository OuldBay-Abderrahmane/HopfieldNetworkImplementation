# -*- coding: utf-8 -*-
import numpy as np
import itertools
from matplotlib import pyplot as plt
from matplotlib import animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
# -------------------- Storkey key functions ----------------------#


def generate_h_matrix(state, weights):
    '''
    This function takes a state matrix and a weights matrix to return their
    associated h matrix, useful for the Storkey learning rule.

    Parameters
    --  --  --  --  --
    state : numpy.array
        One dimensional.    
    weights : numpy.array
        Must be of shape (j, j), j being the size of state.

    Returns
    --  --  -- -
    h_matrix_ij : numpy.array
        Matrix of shape (j, j), j being the size of state.

    Examples
    --  --  --  --
    >>> generate_h_matrix(np.array([1, 1, -1, -1]), np.array([[ 0.  , -0.75,  0.25, -0.25], [-0.75,  0.  , -0.25,  0.25], [ 0.25, -0.25,  0.  ,  0.25], [-0.25,  0.25,  0.25,  0.  ]]))
    array([[-0.75,  0.  , -0.5 , -1.  ],
           [ 0.  , -0.75, -1.  , -0.5 ],
           [-0.5 ,  0.  , -0.25,  0.  ],
           [ 0.  , -0.5 ,  0.  , -0.25]])

    '''
    summation = weights.dot(state)  # sum product of weights and state
    summation = np.reshape(weights.dot(state), (-1, 1))
    h_matrix_i = np.diagonal(weights) * state
    h_matrix_i = h_matrix_i[:, np.newaxis]  # line to column vector

    h_matrix_j = weights * state

    # calculating h matrix by subtracting the sums that we don't want (according to the formula) to the sum product of the parameters
    h_matrix_ij = np.array(weights.shape)
    h_matrix_ij = summation - h_matrix_i - h_matrix_j
    np.fill_diagonal(h_matrix_ij, np.diagonal(
        h_matrix_ij) + np.diagonal(weights) * state)  # correction of the diagonal
    return h_matrix_ij


def storkey_rule(pattern, weight_old):
    '''
    This function applies the Storkey rule to return the a new weights matrix.

    Parameters
    --  --  --  --  --
    pattern : 1D numpy.array
    weight_old : 2D numpy.array

    Returns
    --  --  -- -
    new_matrix : numpy.array
    Storkey weights matrix. Symmetrical matrix with 0 on it's diagonal,
        of shape (j, j), j being the size of each individual pattern.



    Examples
    --  --  --  --
    >>> storkey_rule(np.array([1, -1, -1, 1]), np.array([[ 0.  , -0.75,  0.25, -0.25], [-0.75,  0.  , -0.25,  0.25], [ 0.25, -0.25,  0.  ,  0.25], [-0.25,  0.25,  0.25,  0.  ]]))
    array([[ 0.125, -1.25 ,  0.   ,  0.   ],
           [-1.25 ,  0.125,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.625, -0.25 ],
           [ 0.   ,  0.   , -0.25 ,  0.625]])

    >>> storkey_rule(np.array([1, -1]), np.array([[0., -0.75,  0.25, -0.25], [-0.75, 0., -0.25,  0.25], [0.25, -0.25,  0.,  0.25], [-0.25,  0.25,  0.25,  0.]]))
    Error : storkey_rule(pattern, weight_old) - algorithms.py - The dimensions of the ndarray are incorrect.

    '''
    try:
        h_matrix_ij = generate_h_matrix(pattern, weight_old)
        # calculating the parameters of the Storkey rule in matrices
        p_h_matrix_ij = h_matrix_ij*pattern
        p_h_matrix_ji = (h_matrix_ij*pattern).transpose()

        # application of Storkey rule
        new_matrix = np.array(weight_old.shape)
        new_matrix = weight_old + \
            (np.outer(pattern, pattern)-p_h_matrix_ji -
             p_h_matrix_ij)*(1./pattern.shape[0])
        return new_matrix
    except AttributeError:
        print("Error : storkey_rule(pattern, weight_old) - algorithms.py - The parameters aren't of the correct type be sure to put a numpy.array of the correct size for both.")
    except ValueError:
        print("Error : storkey_rule(pattern, weight_old) - algorithms.py - The dimensions of the ndarray are incorrect.")
    except TypeError:
        print("Error : storkey_rule(pattern, weight_old) - algorithms.py - The parameters aren't of the correct type be sure to put a numpy.array of the correct size for both.")


class DataSaver:
    def __init__(self):
        '''
        The DataSaver class has to store the sequence of states 
        and their associated energy in an attribute.
        Attributes
        ----------
        data : list
            Stores a dynamical systems history.
        energies : list
            Stores the energy associated with each state stored in data.
        '''
        self.data = []
        self.energies = []

    def reset(self):
        self.data.clear()

    def store_iter(self, state, weights):
        '''
        This method stores a state in data 
        and its associated energy in energies.
        '''
        self.data.append(state)
        self.energies.append(self.compute_energy(state, weights))

    def compute_energy(self, state, weights):
        '''
        This method receives a state pattern and a weights matrix and is able to calculate the associated energy.
        The energy obtained corresponds to the energy of a Hopfield network.

        Parameters
        --  --  --  --  --
        state : numpy.array
            One dimensional.
        weights : numpy.array
            Must be of shape (j, j), j being the size of state.

        Returns
        --  --  -- --
        float
        Energy of the given pattern.

        Example
        --  --  -- --
        >>> Data = DataSaver()
        >>> Data.compute_energy(np.array([1, -1, -1, 1], dtype = object), np.array([[0, .33333333, -.33333333, -.33333333], [.33333333, 0, -1., .33333333], [-.33333333, -1., 0, -.33333333], [-.33333333, .33333333, -.33333333, 0]], dtype = object)) == 1.3333333299999997
        True
        '''
        try:
            # Calculation
            return (-0.5 * state.dot(weights).dot(state))
        except AttributeError:
            print("Error : energy(state, weights) - algorithms.py - The parameters aren't of the correct type be sure to put a numpy.array of the correct size for both.")

    def get_data(self):
        return self.data

    def save_video(self, out_path, image_size):
        """
        This method generates a video of the evolution of the system which is saves in outer-path.
        Parameters
        ----------
        out_path : string
            path where the video is saved
        Returns
        -------
        None.
        """
        try:
            # creating the axis
            fig, ax = plt.subplots()

            self.data = list(map(lambda x: x.reshape(
                image_size, image_size), self.data))
            ims = list(
                map(lambda x: [ax.imshow(x, animated=True,
                    cmap="binary")], self.data))

            anim = animation.ArtistAnimation(
                fig, ims, interval=500)
            FFwriter = animation.FFMpegWriter()
            anim.save(out_path, writer=FFwriter)

            return None

        except AttributeError:
            print("Error : save_video(state_list, out_path) - visualization.py - The parameters aren't of the correct type be sure to put a list of np.array of the correct size for state_list and a string for out_path.")
        except IndexError:
            print("Error : save_video(state_list, out_path)  - visualization.py -  The numpy array in the list are not of the correct dimension. Be sure to use an array of size 2D.")

    def plot_energy(self):
        """
        This method generates a plot of the energy evolution of the system which in energy file.

        Returns
        -------
        None.
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(np.arange(len(self.data)), self.energies)
        ax.set(xlabel='Convergence step', ylabel='Energy',
               title='Energy of a perturbed pattern in our Hopefield network')

        return fig

# -------------------- HopfieldNetwork class----------------------#


class HopfieldNetwork:
    def __init__(self, patterns, rule="hebbian"):
        '''
        The HopfieldNetwork class is initialized by passing the patterns and the learning rule as a
        string ("hebbian" or "storkey") and it generates the weights accordingly.
        '''
        self.patterns = patterns  # Init patterns

        if (rule == "hebbian"):
            self.weights = self.hebbian_weights(patterns)
        elif (rule == "storkey"):
            self.weights = self.storkey_weights(patterns)
        else:
            raise AttributeError()

    def hebbian_weights(self, patterns):
        '''
        This method applies the hebbian learning rule on some given patterns
        to return the associated hebbian weights matrix.
        "Neuron that fire together, wire together"
        Each weight is the average contribution of each pattern:
        Each pattern contributes positively to a certain weight if the state
        of the two connected neurons is the same, and negatively otherwise.

        Parameters
        --  --  --  --  --
        patterns : numpy.array
            Input array with shape (i, j). j should be the size of each individual
            pattern, and i the number of patterns.

        Returns
        --  --  -- -
        W : numpy.array
            Hebbian weights matrix. Symmetrical matrix with 0 on it's diagonal,
            of shape (j, j), j being the size of each individual pattern.

        Example
        --  --  -- --
        >>> X = HopfieldNetwork (np.array([[1, -1, -1, 1], [1., 1., -1., 1.], [-1., 1., -1., 1.]]),"hebbian")
        >>> np.diag(X.weights)
        array([0., 0., 0., 0.])
        >>> X.weights
        array([[ 0.        , -0.33333333, -0.33333333,  0.33333333],
               [-0.33333333,  0.        , -0.33333333,  0.33333333],
               [-0.33333333, -0.33333333,  0.        , -1.        ],
               [ 0.33333333,  0.33333333, -1.        ,  0.        ]])
        >>> X.hebbian_weights(2)
        Error : hebbian_weights(patterns)  - algorithms.py -  The parameter isn't of the correct type be sure to put a numpy.array for patterns.
        >>> X.hebbian_weights(np.array([1, -1, -1, 1]))
        Error : hebbian_weights(patterns)  - algorithms.py -  The numpy array isn't of the correct dimension. Be sure to use an array of size 2D.
        '''
        try:
            # Initialization of Hebbian matrix
            hebbian_matrix = np.zeros((patterns.shape[1], patterns.shape[1]))
            shape = patterns.shape[0]
            for pattern in patterns:
                hebbian_matrix += np.outer(pattern, pattern) / shape
            np.fill_diagonal(hebbian_matrix, 0)
            return hebbian_matrix

        except AttributeError:
            print("Error : hebbian_weights(patterns)  - algorithms.py -  The parameter isn't of the correct type be sure to put a numpy.array for patterns.")
        except IndexError:
            print("Error : hebbian_weights(patterns)  - algorithms.py -  The numpy array isn't of the correct dimension. Be sure to use an array of size 2D.")

    def storkey_weights(self, patterns):
        '''
        This method applies the Storkey rule to return the a new weights matrix.
        This learning rule allows the network to memorize a larger number of patterns,
        given the same number of neurons.

        Parameters
        --  --  --  --  --
        patterns : numpy.array
            Input array with shape (i, j). j should be the size of each individual
            pattern, and i the number of patterns.

        Returns
        --  --  -- -
        W_array : numpy.array
        Storkey weights matrix. Symmetrical matrix with 0 on it's diagonal,
            of shape (j, j), j being the size of each individual pattern.

        Examples
        --  --  --  --
        >>> X = HopfieldNetwork (np.array([[1., 1 , -1. , -1.], [1., 1., -1, 1.], [-1., 1, -1., 1.]]),"storkey")
        >>> X.weights
        array([[ 1.125,  0.25 , -0.25 , -0.5  ],
               [ 0.25 ,  0.625, -1.   ,  0.25 ],
               [-0.25 , -1.   ,  0.625, -0.25 ],
               [-0.5  ,  0.25 , -0.25 ,  1.125]])
        >>> X.storkey_weights(2)
        Error : storkey_weights(patterns) - algorithms.py - The parameter isn't of the correct type be sure to put a numpy.array for patterns.
        >>> X.storkey_weights(np.array([1, -1, -1, 1]))
        Error : storkey_weights(patterns) - algorithms.py -  The numpy array isn't of the correct dimension. Be sure to use an array of size 2D.
        '''
        try:
            # Initialization of Storkey matrix
            shape = patterns.shape[1]
            W_array = np.zeros((shape, shape))

            # calculation the Storkey weights with the previous defined function storkey_rule
            for pattern in patterns:
                W_array = storkey_rule(pattern, W_array)
            return W_array

        # return W_array[-1]
        except AttributeError:
            print("Error : storkey_weights(patterns) - algorithms.py - The parameter isn't of the correct type be sure to put a numpy.array for patterns.")
        except IndexError:
            print("Error : storkey_weights(patterns) - algorithms.py -  The numpy array isn't of the correct dimension. Be sure to use an array of size 2D.")

    def update(self, state):
        '''
        This method receives a state pattern and a weights matrix,
        and applies the update rule to return a new updated state.
        The update rule consist in the matrix product of the weight and the state.

        Parameters
        --  --  --  --  --
        state : numpy.array
            One dimensional.

        Returns
        --  --  -- -
        state : numpy.array
            New updated state.

        Example
        --  --  -- --
        >>> X = HopfieldNetwork (np.array([[1, -1, -1, 1], [1., 1., -1., 1.], [-1., 1., -1., 1.]]),"hebbian")
        >>> X.update(np.array([1. , -1. , -1., 1.], dtype = object)).shape[0] == 4
        True
        '''
        try:
            # Scalar product between our current pattern and our weights matrices
            state = np.dot(self.weights, state)
            state[state < 0] = -1.  # Change negative values to -1
            state[state >= 0] = 1.  # Change negative values to 1
            return state
        except AttributeError:
            print("Error : update(state, weights) - algorithms.py - The parameters aren't of the correct type be sure to put a numpy.array of the correct size for both.")
        except ValueError:
            print(
                "Error : update(state, weights) - algorithms.py - The dimensions of the ndarray are incorrect.")

    def update_async(self, state):
        '''
        This method receives a state pattern and a weights matrix,
        and applies the asynchronous update rule to return a new updated state.
        The asynchronous update rule update only one random element (the i-th) of a given
        state vector by giving it the scalar product of the i-th line of the weights matrix
        and the original state vector.

        Parameters
        --  --  --  --  --
        state : numpy.array
            One dimensional.

        Returns
        --  --  -- -
        state : numpy.array
            New updated state.

        Example
        --  --  -- --
        >>> X = HopfieldNetwork (np.array([[1, -1, -1, 1], [1., 1., -1., 1.], [-1., 1., -1., 1.]]),"hebbian")
        >>> X.update_async(np.array([1. , -1. , -1., 1.], dtype = object)).shape[0] == 4
        True
        '''
        rand_int = np.random.randint(0, high=self.weights.shape[0])
        newState = np.copy(state)
        newState[rand_int] = np.dot(self.weights[rand_int, :], newState)
        if(newState[rand_int] < 0):
            newState[rand_int] = -1
        else:
            newState[rand_int] = 1
        return newState

    def dynamics(self, state, saver, max_iter=20):
        '''
        This method receives a state pattern, a weights matrix and maximum number of iteration possible
        and applies the update rule to try reaching a convergence.
        The dynamics method take the dynamical system from an initial state until convergence or until a maximum number of steps is reached.
        This method saves the full history of updates made in the saver.

        Parameters
        --  --  --  --  --
        state : numpy.array
            One dimensional.
        saver : DataSaver
            Object where the full history of updates will be save.
        max_iter : int
            Maximum number of iterations allowed.

        Returns
        --  --  -- -
        None

        Example
        --  --  -- --
        >>> X = HopfieldNetwork (np.array([[1, -1, -1, 1], [1., 1., -1., 1.], [-1., 1., -1., 1.]]),"hebbian")
        >>> Data = DataSaver()
        >>> X.dynamics(np.array([-1,-1,-1,-1]), Data)
        >>> np.size(Data.data[0])==np.size(Data.data[-1])
        True
        '''
        # history = saver.get_data()  # Initialization of our history array
        # history[saver.iteration] = state
        saver.store_iter(state, self.weights)
        iter = 0
        saver.store_iter(self.update(state), self.weights)  # First update
        iter += 1
        # Stop if current state the same as previous one in history array
        while not np.allclose(saver.get_data()[-2], saver.get_data()[-1]):
            iter += 1
            saver.store_iter(self.update(state), self.weights)
            if iter == max_iter:  # Stop if reached max_iter times
                break
        # saver.store_iter(self.update(state), self.weights)
        return None

    def dynamics_async(self, state, saver, max_iter=1000, convergence_num_iter=100, skip=10):
        '''
        This method receives a state pattern, a weights matrix, a maximum number of iteration possible
        and convergence number of iterations and applies the asynchronous update rule to try reaching a convergence.
        The asynchronous dynamics method take the dynamical system from an initial state until convergence or until a maximum number of steps is reached.
        This method saves the full history of asynchronous updates made in the saver.

        Parameters
        --  --  --  --  --
        state : numpy.array
            One dimensional.
        saver : DataSaver
            Object where the full history of updates will be save.
        max_iter : int
            Maximum number of iterations allowed.
        convergence_num_iter : int
            Criterion to consider the pattern having reached convergence.

        Returns
        --  --  -- -
        None

        Example
        --  --  -- --
        >>> X = HopfieldNetwork (np.array([[1, -1, -1, 1], [1., 1., -1., 1.], [-1., 1., -1., 1.]]))
        >>> Data = DataSaver()
        >>> X.dynamics_async(np.array([-1,-1,-1,-1]), Data)
        >>> np.size(Data.data[0])==np.size(Data.data[-1])
        True
        '''
        # Initialization of history of state
        convergence_step = 0
        saver.store_iter(state, self.weights)
        # saver.type = "async"
        iter = 0
        history = [state]
        iter += 1
        for _ in itertools.repeat(max_iter):
            updated_state = self.update_async(history[-1])
            iter += 1
            if ((iter % skip) == 0):
                saver.store_iter(updated_state, self.weights)
            history.append(updated_state)
            if (np.array_equal(history[-2], history[-1])):
                convergence_step += 1
                if convergence_step == convergence_num_iter:
                    saver.store_iter(updated_state, self.weights)
                    break
            else:
                convergence_step = 0
        return None

# -------------------- Generation and key patterns functions ----------------------#


def generate_patterns(num_patterns, pattern_size):
    '''
    This function generates the patterns.
    Patterns is a 2-dimensional numpy array, in which each row (num_patterns in total) is a
    random binary pattern (possible values: {1, -1}) of size pattern_size.

    Parameters
    --  --  --  --  --
    num_patterns : interger
        Input the number of pattern
    pattern_size : interger
        Input the size of the patterns

    Returns
    --  --  -- -
    patterns : 2-dimensional numpy.array
    Patterns matrix of shape (i, j), i being the total number of patterns, j being the size of each individual pattern.

    Example
    --  --  -- --

    >>> generate_patterns(2, 3).shape == (2, 3)
    True
    >>> generate_patterns(np.array([[1, 1, -1]], dtype = object), 3)
    Error : generate_patterns(num_patterns, pattern_size)  - algorithms.py -  The parameters aren't of the correct type be sure to put integers.
    '''
    try:
        # generate a matrix of chosen size which values are randomly chosen between 1 and -1
        return np.random.choice([-1., 1.], size=(num_patterns, pattern_size))
    except ValueError:
        print("Error : generate_patterns(num_patterns, pattern_size)  - algorithms.py -  The parameters aren't of the correct type be sure to put integers.")


def perturb_pattern(pattern, num_perturb):
    '''
    This function perturbs a given pattern.
    It samples num_perturb elements of the input pattern uniformly at random and changes their sign.

    Parameters
    --  --  --  --  --
    pattern : 1-dimensional numpy.array
        Input the array that will be perturb
    num_pertub : interger
        Input the number of perturbations that will occur

    Returns
    --  --  -- -
    pattern : 1-dimensional numpy.array
    The perturbed pattern

    Example
    --  --  -- --

    >>> np.equal(perturb_pattern(np.array([1, -1, 1, 1, -1, 1], dtype = object), 6), np.array([1, -1, 1, 1, -1, 1], dtype = object)).all() == False
    True
    >>> perturb_pattern(3, np.array([[1, 1, -1]], dtype = object))
    Error : perturb_pattern(pattern, num_perturb)  - algorithms.py -  The parameters aren't of the correct type be sure to put a numpy.array for pattern and an integer for the number of perturbations.
    '''
    try:
        # chose random locations in the pattern
        indexes = np.random.choice(pattern.shape[0], size=num_perturb)
        newPattern = np.zeros(pattern.shape[0], dtype=int) + pattern
        # replace the original state by its opposite at the randomly chosen locations
        newPattern[indexes] = newPattern[indexes]*-1
        return newPattern

    except AttributeError:
        print("Error : perturb_pattern(pattern, num_perturb)  - algorithms.py -  The parameters aren't of the correct type be sure to put a numpy.array for pattern and an integer for the number of perturbations.")


def pattern_match(memorized_patterns, pattern):
    '''
    This function sees if a match exists between a pattern and a memorized one.

    Parameters
    --  --  --  --  --
    memorized_pattern : numpy.array
        Input array with shape (i, j). j should be the size of each individual
        pattern, and i the number of patterns.
    pattern : numpy.array
        Input a 1-dimensional array

    Returns
    --  --  -- -
    i : the index of the row corresponding to the matching pattern if the a memorized pattern matches
    none if no memorized pattern matches

    Example
    --  --  -- --

    >>> pattern_match(np.array([[1. , -1. , -1.], [1., 1., 1.], [1., -1., 1.]], dtype = object), np.array([1. , -1. , -1.], dtype = object))
    0

    >>> print(pattern_match(np.array([1., 1., -1.], dtype = object), np.array([[1., -1., -1.], [1., 1., 1.], [1., -1., 1.]], dtype = object)))
    It didn't converge be sure to verify that the parameters are of the correct type. pattern_match(memorized_patterns, pattern) - algorithms.py
    None
    '''
    try:
        return (np.where(
            np.equal(pattern, memorized_patterns).all(axis=1) == True)[0][0])  # return the index of the true element (found with np.equal) where the match occurs

    except IndexError:
        return None

    except AttributeError:
        return None


def is_symmetric(X):
    '''
    This function verifies if the given matrix X is symmetric.
    A matrix is symmetric if the transpose matrix is equal to the original matrix.

    Parameters
    --  --  --  --  --

    X : numpy.array
        Must be of shape (j, j).

    Returns
    --  --  -- -
    Bool
    True if the matrix is symmetric or false if it is not.

    Example
    --  --  -- --
    >>> is_symmetric(np.array([[1,2,3],[2,1,2],[3,2,1]]))
    True
    '''
    try:
        # Test on transpose
        if(X.shape[0] == X.shape[1]):
            return np.allclose(X, X.transpose())
        else:
            return False
    except AttributeError:
        print("Error : is_symmetric(X) - algorithms.py - The parameter isn't of the correct type be sure to put a 2D numpy.array representing a square matrix.")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
