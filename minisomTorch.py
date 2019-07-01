from math import sqrt

from numpy import (array, unravel_index, nditer, linalg, random, power, pi,
                   zeros, arange, outer, meshgrid, dot, logical_and, cov,
                   argsort, linspace, transpose, exp,
                   log, sum,
                   )
from collections import defaultdict, Counter
from warnings import warn
from sys import stdout
from time import time

# for unit tests
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from numpy.testing import assert_array_equal
import unittest

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm

"""
    Minimalistic implementation of the Self Organizing Maps (SOM). There are some minor changes from the 
    original minisom, like the decay functions.
"""


def _incremental_index_verbose(m):
    """Yields numbers from 0 to m-1 printing the status on the stdout."""
    progress = f'\r [ {0:{len(str(m))}} / {m} ] {0:3.0f}% - ? it/s'
    stdout.write(progress)
    beginning = time()
    for i in range(m):
        yield i
        it_per_sec = (time() - beginning) / (i + 1)
        progress = f'\r [ {i + 1:{len(str(m))}} / {m} ]'
        progress += f' {100 * (i + 1) / m:3.0f}%'
        progress += f' - {it_per_sec:4.5f} it/s'
        stdout.write(progress)


def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.

    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return sqrt(dot(x, x.T))


def asymptotic_decay(learning_rate, t, max_iter):
    """Decay function of the learning process.
    Parameters
    ----------
    learning_rate : float
        current learning rate.

    t : int
        current iteration.

    max_iter : int
        maximum number of iterations for the training.
    """
    return learning_rate / (1 + t / (max_iter / 2))


def exponential_decay(learning_rate, e, num_epochs):
    """
    Decay function of the learning_rate and sigma.
    :param learning_rate: float initial learning rate
    :param e: int current epoch
    :param num_epochs: int total number of epochs
    :return: float current learning rate
    """
    return learning_rate * exp(-e / num_epochs)


class MiniSom(object):
    def __init__(self, x, y, input_len, sigma=None, learning_rate=0.5,
                 decay_function=exponential_decay,
                 neighborhood_function='manhattan',
                 random_seed=None
                 ):
        """Initializes a Self Organizing Maps.

        A rule of thumb to set the size of the grid for a dimensionality
        reduction task is that it should contain 5*sqrt(N) neurons
        where N is the number of samples in the dataset to analyze.

        E.g. if your dataset has 150 samples, 5*sqrt(150) = 61.23
        hence a map 8-by-8 should perform well.

        Parameters
        ----------
        x : int
            x dimension of the SOM.

        y : int
            y dimension of the SOM.

        input_len : int
            Number of the elements of the vectors in input.

        sigma : float, optional (default=1.0)
            Spread of the neighborhood function, needs to be adequate
            to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T)
            where T is #num_iteration/2)
            learning_rate, initial learning rate
            (at the iteration t we have
            learning_rate(t) = learning_rate / (1 + t/T)
            where T is #num_iteration/2)

        decay_function : function (default=None)
            Function that reduces learning_rate and sigma at each iteration
            the default function is:
                        learning_rate / (1+t/(max_iterarations/2))

            A custom decay function will need to to take in input
            three parameters in the following order:

            1. learning rate
            2. current iteration
            3. maximum number of iterations allowed


            Note that if a lambda function is used to define the decay
            MiniSom will not be pickable anymore.

        neighborhood_function : function, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map
            possible values: 'gaussian', 'mexican_hat', 'bubble', 'manhattan'

        random_seed : int, optional (default=None)
            Random seed to use.
        """
        if random_seed != None:
            torch.random.manual_seed(random_seed)
        self._learning_rate = learning_rate
        if sigma != None:
            self._sigma = sigma
        else:
            self._sigma = sqrt(x**2 + y**2)/2
        self._input_len = input_len
        # random initialization
        self._weights = torch.rand(x * y, input_len).to(device) * 2 - 1
        self._weights /= torch.norm(self._weights, dim=1)[:, None]

        self._neigx = torch.arange(x).float().to(device)
        self._neigy = torch.arange(y).float().to(device)  # used to evaluate the neighborhood function
        self._xdim = x
        self._ydim = y
        self._decay_function = decay_function
        self.step = 0
        neig_functions = {'gaussian': self._gaussian,
                          'mexican_hat': self._mexican_hat,
                          'bubble': self._bubble,
                          'triangle': self._triangle,
                          'manhattan': self._manhattan,
                          }

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        if neighborhood_function in ['triangle',
                                     'bubble'] and divmod(sigma, 1)[1] != 0:
            warn('sigma should be an integer when triangle or bubble' +
                 'are used as neighborhood function')

        self.neighborhood = neig_functions[neighborhood_function]

    def get_weights(self):
        """Returns the weights of the neural network."""
        return self._weights

    def _gaussian(self, sigma, win=None):
        """Returns a Gaussian centered in the center of matrix."""
        d = 2 * pi * sigma * sigma
        totaldim = self._xdim * self._ydim
        if 'NoneType' in str(type(win)):
            totaldim = self._xdim * self._ydim
            neighborhoodperneuron = torch.zeros(totaldim, totaldim).float().to(device)
            for i in range(self._xdim * self._ydim):
                x, y = unravel_index(i, (self._xdim, self._ydim))
                ax = torch.exp(-torch.pow(self._neigx - x, 2) / d)
                ay = torch.exp(-torch.pow(self._neigy - y, 2) / d)
                neighborhoodperneuron[i] = torch.einsum("i,j->ij", ax,ay).reshape(totaldim)
        else:
            y = win // self._ydim
            x = win % self._ydim
            ax = torch.exp(-torch.pow(self._neigx - x, 2) / d)
            ay = torch.exp(-torch.pow(self._neigy - y, 2) / d)
            neighborhoodperneuron = torch.einsum("i,j->ij", ax,ay).reshape(totaldim)
        return neighborhoodperneuron

    def _mexican_hat(self, sigma, win=None):
        """Mexican hat centered in c."""
        d = 2 * pi * sigma * sigma
        totaldim = self._xdim * self._ydim
        if 'NoneType' in str(type(win)):
            totaldim = self._xdim * self._ydim
            neighborhoodperneuron = torch.zeros(totaldim, totaldim).float().to(device)
            for i in range(self._xdim * self._ydim):
                x, y = unravel_index(i, (self._xdim, self._ydim))
                xx, yy = torch.meshgrid(self._neigx, self._neigy)
                p = torch.pow(xx - x, 2) + torch.pow(yy - y, 2)
                neighborhoodperneuron[i] = (torch.exp(-p / d) * (1 - 2 / d * p)).reshape(self._xdim * self._ydim)
        else:
            y = win // self._ydim
            x = win % self._ydim
            xx, yy = torch.meshgrid(self._neigx, self._neigy)
            p = torch.pow(xx - x, 2) + torch.pow(yy - y, 2)
            neighborhoodperneuron = (torch.exp(-p / d) * (1 - 2 / d * p)).reshape(self._xdim * self._ydim)
        return neighborhoodperneuron

    def _bubble(self, sigma, win=None):
        """
        Constant function centered in c with spread sigma.
        sigma should be an odd value.
        """
        totaldim = self._xdim * self._ydim
        if 'NoneType' in str(type(win)):
            totaldim = self._xdim * self._ydim
            neighborhoodperneuron = torch.zeros(totaldim, totaldim).float().to(device)
            for i in range(self._xdim * self._ydim):
                x, y = unravel_index(i, (self._xdim, self._ydim))
                a = self._neigx > x - sigma
                b = self._neigx < x + sigma
                ax = a * b * 1
                a = self._neigy > y - sigma
                b = self._neigy < y + sigma
                ay = a * b * 1
                neighborhoodperneuron[i] = torch.einsum("i,j->ij", ax,ay).float().reshape(totaldim)
        else:
            x = win % self._ydim
            y = win // self._ydim
            a = self._neigx > (x - sigma)
            b = self._neigx < (x + sigma)
            ax = a * b
            a = self._neigy > (y - sigma)
            b = self._neigy < (y + sigma)
            ay = a * b
            neighborhoodperneuron = torch.einsum("i,j->ij", ax, ay).float().reshape(totaldim)
        return neighborhoodperneuron

    def _triangle(self, sigma, win=None):
        """Triangular function centered in c with spread sigma."""
        totaldim = self._xdim * self._ydim
        if 'NoneType' in str(type(win)):
            totaldim = self._xdim * self._ydim
            neighborhoodperneuron = torch.zeros(totaldim, totaldim).float().to(device)
            for i in range(self._xdim * self._ydim):
                x, y = unravel_index(i, (self._xdim, self._ydim))
                triangle_x = (-torch.abs(x - self._neigx)) + sigma
                triangle_y = (-torch.abs(y - self._neigy)) + sigma
                triangle_x[triangle_x < 0] = 0.
                triangle_y[triangle_y < 0] = 0.
                neighborhoodperneuron[i] = torch.einsum("i,j->ij", triangle_x, triangle_y).float().reshape(totaldim)
        else:
            x = win % self._ydim
            y = win // self._ydim
            triangle_x = -torch.abs(x - self._neigx) + sigma
            triangle_y = -torch.abs(y - self._neigy) + sigma
            triangle_x[triangle_x < 0] = 0.
            triangle_y[triangle_y < 0] = 0.
            neighborhoodperneuron = torch.einsum("i,j->ij", triangle_x,triangle_y).float().reshape(totaldim)
        return neighborhoodperneuron


    def _manhattan(self, sigma=-1, win=None):
        """
        Creates a distance table with manhattan distances taking into consideration
        a rectangular map of (xdim, ydim)
        :return: matrix of (x*y)x(x*y) dimensions with the distances
        """

        totaldim = self._xdim * self._ydim
        if 'NoneType' in str(type(win)):
            mandisttable = zeros((totaldim, totaldim))
            for i in range(totaldim):
                for j in range(totaldim):
                    xipos, yipos = unravel_index(i, (self._xdim, self._ydim))
                    xjpos, yjpos = unravel_index(j, (self._xdim, self._ydim))
                    mandisttable[i, j] = abs(xipos - xjpos) + abs(yipos - yjpos)
        else:
            mandisttable = zeros((totaldim))
            y = win // self._ydim
            x = win % self._ydim
            for i in range(self._xdim):
                for j in range(self._ydim):
                    mandisttable[i, j] = abs(x - i) + abs(y - j)
        return torch.from_numpy(mandisttable).float().to(device)

    def _check_iteration_number(self, num_iteration):
        if num_iteration < 1:
            raise ValueError('num_iteration must be > 1')

    def _check_input_len(self, data):
        """Checks that the data in input is of the correct shape."""
        data_len = len(data[0])
        if self._input_len != data_len:
            msg = 'Received %d features, expected %d.' % (data_len,
                                                          self._input_len)
            raise ValueError(msg)

    def winner(self, input):
        differences = input - self._weights
        return (differences ** 2).sum(dim=1).argmin()

    def winner_batch(self, input):
        differences = (self._weights - input[:, None])  # here we assume weights are 2 dimensional
        return (differences ** 2).sum(dim=2).argmin(dim=1)

    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        if 'numpy' in str(type(data)):
            data = torch.from_numpy(data).float().to(device)
        self._check_input_len(data)
        error = 0
        for x in data:
            error += torch.sqrt((((x - self._weights[self.winner(x)])) ** 2).sum())
        return error / len(data)

    def quantization_error_batch(self, data, num_batches, batchsize):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        if 'numpy' in str(type(data)):
            data = torch.from_numpy(data).float().to(device)
        self._check_input_len(data)
        error = 0
        for b in range(num_batches):
            x = data[b * batchsize:(b + 1) * batchsize]
            error += torch.sqrt(((x - self._weights[self.winner_batch(x)]) ** 2).sum(dim=1)).sum()
        return error / len(data)

    def random_weights_init(self, data):
        """Initializes the weights of the SOM
        picking random samples from data."""
        self._check_input_len(data)
        if 'numpy' in str(type(data)):
            data = torch.from_numpy(data).float().to(device)
        for i in range(self._xdim * self._ydim):
            rand_i = torch.randint(len(data), size=(1,))[0]
            self._weights[i] = data[rand_i]

    def pca_weights_init(self, data):
        """Initializes the weights to span the first two principal components.

        This initialization doesn't depend on random processes and
        makes the training process converge faster.

        It is strongly reccomended to normalize the data before initializing
        the weights and use the same normalization for the training data.
        """
        if self._input_len == 1:
            msg = 'The data needs at least 2 features for pca initialization'
            raise ValueError(msg)
        if 'torch' in str(type(data)):
            data = data.cpu().numpy()
        self._check_input_len(data)
        if len(self._neigx) == 1 or len(self._neigy) == 1:
            msg = 'PCA initialization inappropriate:' + \
                  'One of the dimensions of the map is 1.'
            warn(msg)
        pc_length, pc = linalg.eig(cov(transpose(data)))
        pc_order = argsort(pc_length)
        weights = zeros((self._xdim, self._ydim, data.shape[1]))
        for i, c1 in enumerate(linspace(-1, 1, len(self._neigx))):
            for j, c2 in enumerate(linspace(-1, 1, len(self._neigy))):
                weights[i, j] = c1 * pc[pc_order[0]] + c2 * pc[pc_order[1]]
        weights = weights.reshape(self._xdim * self._ydim, weights.shape[-1])
        self._weights = torch.from_numpy(weights).float().to(device)

    def update(self, x, win, iteration, num_iteration, batch):
        """
        Updates the weights of the neurons.

        :param x: np.array Current pattern to learn.
        :param win: tuple Position of the winning neuron for x (array or tuple).
        :param e: int epoch index
        :param num_epochs: int Maximum number of training epochs.
        """
        if not batch:
            self.eta = self._decay_function(self._learning_rate, iteration, num_iteration)
            # sigma and learning rate decrease with the same rule
            self.sig = self._decay_function(self._sigma, iteration, self._tau)
            self.g = self.neighborhood(self.sig, win)
            self._weights += self.eta * torch.einsum('i, ij->ij', self.g, x - self._weights)
        else:
            self._weights += self.eta * torch.einsum('i, ij->ij', self.g[win], x - self._weights)

    def train_random(self, data, num_iteration, verbose=False, batch=False):
        """Trains the SOM picking samples at random from data.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        num_iterations : int
            Maximum number of iterations (one iteration per sample).

        verbose : bool (default=False)
            If True the status of the training
            will be printed at each iteration.
        batch: bool (default=False)
            If True the neighboorhood map and learning rate updated once the epoch in other case it changes every epoch
        """
        if 'numpy' in str(type(data)):
            data = torch.from_numpy(data).float().to(device)
        num_epochs = num_iteration // len(data)
        if not batch:
            num_epochs = num_iteration
        self._tau = num_epochs / log(self._sigma)

        self._check_iteration_number(num_iteration)
        self._check_input_len(data)
        iterations = range(num_iteration)

        for iteration in tqdm(iterations):
            # calculate the learning rate, sigma and neighborhood for each epoch

            if batch and (iteration % len(data) == 0):
                self.eta = self._decay_function(self._learning_rate, iteration // len(data), num_epochs)
                # sigma and learning rate decrease with the same rule
                self.sig = self._decay_function(self._sigma, iteration // len(data), self._tau)
                self.g = self.neighborhood(self.sig)

            # pick a random sample
            rand_i = torch.randint(len(data), size=(1,))[0]
            self.update(data[rand_i], self.winner(data[rand_i]), iteration, num_iteration, batch)
        if verbose:
            batchsize = 100
            num_batches = len(data) // batchsize
            print(' - quantization error:', self.quantization_error_batch(data, num_batches, batchsize))

    def update_batch(self, data, winnersid, neighborhood):
        """
        update the weights for a specific batch
        :param input: np.array input vectors for this batch (RxQ)
        :param winnersid: np.array winner id (SxQ)
        :param neighborhood: np.array for the distances of neurons of som (SxS)
        weights have dimension (SxR)
        S = SOMydim * SOMxdim
        R = feature number of input vector
        Q = number of samples per batch
        :return:
        """
        neighbor = (neighborhood < self.nd).float().to(device)
        winnersid = winnersid * (torch.rand(winnersid.shape[0], winnersid.shape[1]) < 0.9).float().to(device)
        winnersidwithneighbors = torch.matmul(neighbor, winnersid) + winnersid
        winnerlocalimpact = winnersidwithneighbors.sum(dim=1)
        newweights = winnersidwithneighbors / winnerlocalimpact.reshape(winnerlocalimpact.shape[0], 1)
        newweights = torch.matmul(newweights, data)
        loserindex = newweights != newweights
        dw = newweights - self._weights
        dw[loserindex] = 0
        self._weights += dw * self.eta

    def train_batch(self, data, num_epochs, num_batches, initneigh=None, verbose=False):
        """
        Trains using batches.

        Parameters
        ----------
        data : np.array or list      Data matrix.
        num_iterations : int   Maximum number of iterations (one iteration per sample).
        initneigh: int number of initial neighbors
        verbose : bool (default=False) If True the status of the training will be printed at each iteration.
        """
        if 'numpy' in str(type(data)):
            data = torch.from_numpy(data).float().to(device)
        self._check_iteration_number(num_epochs)
        self._check_input_len(data)
        if initneigh == None:
            self.initneig = num_epochs / log(self._sigma)
        else:
            self.initneig = initneigh
        neighborhood = self.neighborhood()
        batchsize = len(data) // num_batches

        for e in tqdm(range(num_epochs)):

            self.eta = exponential_decay(self._learning_rate, e, num_epochs)
            self.nd = exponential_decay(self.initneig, e, num_epochs)

            for b in range(num_batches):
                databatch = data[b * batchsize: (b + 1) * batchsize]
                winners = self.winner_batch(databatch)
                winnersmat = torch.zeros((self._xdim * self._ydim, winners.shape[0])).to(device)
                for i in range(winners.shape[0]):
                    winnersmat[winners[i], i] = 1

                self.update_batch(databatch, winnersmat, neighborhood)

        torch.cuda.empty_cache()
        if verbose:
            print(' - quantization error:', self.quantization_error_batch(data, num_batches, batchsize).cpu())

    def distance_map(self):
        """Returns the distance map of the weights.
        Each cell is the normalised sum of the distances between
        a neuron and its neighbours."""
        um = zeros((self._xdim, self._ydim))
        it = nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0] - 1, it.multi_index[0] + 2):
                for jj in range(it.multi_index[1] - 1, it.multi_index[1] + 2):
                    if (ii >= 0 and ii < self._xdim and
                            jj >= 0 and jj < self._ydim):
                        w_1 = self._weights[ii * self._ydim + jj, :]
                        w_2 = self._weights[it.multi_index[0] * self._ydim + it.multi_index[1]]
                        um[it.multi_index] += torch.sqrt(((w_1 - w_2) ** 2).sum())
            it.iternext()
        um = um / um.max()
        return um

    def activation_response(self, data):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        if 'numpy' in str(type(data)):
            data = torch.from_numpy(data).float().to(device)
        self._check_input_len(data)
        a = zeros(self._xdim * self._ydim)
        for x in data:
            a[self.winner(x)] += 1
        return a

    def win_map(self, data):
        """Returns a dictionary wm where wm[(i,j)] is a list
        with all the patterns that have been mapped in the position i,j."""
        if 'numpy' in str(type(data)):
            data = torch.from_numpy(data).float().to(device)
        self._check_input_len(data)
        winmap = defaultdict(list)
        for x in data:
            winmap[int(self.winner(x).cpu())].append(x)
        return winmap

    def labels_map(self, data, labels):
        """Returns a dictionary wm where wm[(i,j)] is a dictionary
        that contains the number of samples from a given label
        that have been mapped in position i,j.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        label : np.array or list
            Labels for each sample in data.
        """
        if 'numpy' in str(type(data)):
            data = torch.from_numpy(data).float().to(device)
        self._check_input_len(data)
        winmap = defaultdict(list)
        for x, label in zip(data, labels):
            print(self)
            winmap[int(self.winner(x).cpu())].append(label)
        for position in winmap:
            winmap[position] = Counter(winmap[position])
        return winmap


class TestMinisom(unittest.TestCase):
    def setUp(self):
        self.som = MiniSom(5, 5, 1, decay_function=asymptotic_decay)
        for i in range(5):
            for j in range(5):
                # checking weights normalization
                assert_almost_equal(1.0, linalg.norm(self.som._weights[i*5 + j].cpu().numpy()))
        self.som._weights = torch.zeros((5 * 5, 1)).float().to(device)  # fake weights
        self.som._weights[2*5 + 3] = 5.0
        self.som._weights[1*5 + 1] = 2.0

    def test_decay_function(self):
        assert self.som._decay_function(1., 2., 3.) == 1. / (1. + 2. / (3. / 2))

    def test_check_input_len(self):
        with self.assertRaises(ValueError):
            self.som.train_random([[1, 2]], 1)

        with self.assertRaises(ValueError):
            self.som.random_weights_init(array([[1, 2]]))

        with self.assertRaises(ValueError):
            self.som._check_input_len(array([[1, 2]]))

        self.som._check_input_len(array([[1]]))
        self.som._check_input_len([[1]])

    def test_unavailable_neigh_function(self):
        with self.assertRaises(ValueError):
            MiniSom(5, 5, 1, neighborhood_function='boooom')

    def test_gaussian(self):
        bell = self.som._gaussian(1, 2*5 + 2)
        assert bell.max() == 1.0
        assert bell.argmax() == 12  # unravel(12) = (2,2)

    def test_mexican_hat(self):
        bell = self.som._mexican_hat(1, 2*5 + 2)
        assert bell.max() == 1.0
        assert bell.argmax() == 12  # unravel(12) = (2,2)

    def test_bubble(self):
        bubble = self.som._bubble(1, 2*5 + 2)
        assert bubble[2*5 + 2] == 1.
        assert bubble.sum() == 1.

    def test_triangle(self):
        triangle = self.som._triangle(1, 2*5 + 2)
        assert triangle[2*5 + 2] == 1.
        assert triangle.sum().cpu() == 1.

    def test_win_map(self):
        winners = self.som.win_map(torch.Tensor([[5.0], [2.0]]).to(device))
        assert winners[13][0] == torch.Tensor([5.0]).to(device)
        assert winners[6][0] == torch.Tensor([2.0]).to(device)

    def test_labels_map(self):
        labels_map = self.som.labels_map(torch.Tensor([[5.0], [2.0]]).float().to(device), ['a', 'b'])
        assert labels_map[2 * 5 + 3]['a'] == 1
        assert labels_map[1 * 5 + 1]['b'] == 1

    def test_quantization_error(self):
        assert self.som.quantization_error(torch.Tensor([[5.0], [2.0]]).float().to(device)) == 0.0
        assert self.som.quantization_error(torch.Tensor([[4.0], [1.0]]).float().to(device)) == 1.0

    def test_random_seed(self):
        som1 = MiniSom(5, 5, 2, sigma=2.0, learning_rate=0.5, random_seed=1, neighborhood_function="gaussian")
        som2 = MiniSom(5, 5, 2, sigma=2.0, learning_rate=0.5, random_seed=1, neighborhood_function="gaussian")
        # same initialization
        assert_array_almost_equal(som1._weights.cpu().numpy(), som2._weights.cpu().numpy())
        data = torch.rand(100, 2).float().to(device)
        som1 = MiniSom(5, 5, 2, sigma=2.0, learning_rate=0.5, random_seed=1, neighborhood_function="gaussian")
        som1.train_random(data, 10)
        som2 = MiniSom(5, 5, 2, sigma=2.0, learning_rate=0.5, random_seed=1, neighborhood_function="gaussian")
        som2.train_random(data, 10)
        # same state after training
        assert_array_almost_equal(som1._weights.cpu().numpy(), som2._weights.cpu().numpy())

    def test_train_batch(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = torch.Tensor([[4, 2], [3, 1]]).to(device)
        q1 = som.quantization_error(data)
        som.train_batch(data, 10, 2)
        assert q1 > som.quantization_error(data)

        data = array([[1, 5], [6, 7]])
        q1 = som.quantization_error(data)
        som.train_batch(data, 10, 2, verbose=True)
        assert q1 > som.quantization_error(data)

    def test_train_random(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1, neighborhood_function="gaussian")
        data = torch.Tensor([[4, 2], [3, 1]]).float().to(device)
        q1 = som.quantization_error(data)
        som.train_random(data, 10)
        assert q1 > som.quantization_error(data)

        data = torch.Tensor([[1, 5], [6, 7]]).float().to(device)
        q1 = som.quantization_error(data)
        som.train_random(data, 10, verbose=True)
        assert q1 > som.quantization_error(data)

    def test_random_weights_init(self):
        som = MiniSom(2, 2, 2, random_seed=1)
        som.random_weights_init(torch.Tensor([[1.0, .0]]).float().to(device))
        for w in som._weights:
            assert_array_equal(w[0*2].cpu().numpy(), array([1.0]))
            assert_array_equal(w[0*2+1].cpu().numpy(), array([0]))

    def test_pca_weights_init(self):
        som = MiniSom(2, 2, 2)
        weights = torch.Tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]).reshape(4,2)
        som.pca_weights_init(weights)
        expected = array([[[0., -1.41421356], [1.41421356, 0.]],
                          [[-1.41421356, 0.], [0., 1.41421356]]])
        assert_array_almost_equal(som._weights.cpu().numpy().reshape(2,2,2), expected)

    def test_distance_map(self):
        som = MiniSom(2, 2, 2, random_seed=1)
        som._weights = torch.from_numpy(
            array([[[1., 0.], [0., 1.]], [[1., 0.], [0., 1.]]])).float().reshape(4,2).to(device)
        assert_array_equal(som.distance_map().reshape(2,2), array([[1., 1.], [1., 1.]]))
