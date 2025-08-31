import numpy as np

class Adaline(object):
    def __init__(self, eta=0.01, pathModels='.//models'):
        self.W = None
        self.errorForEachEpoch = list()
        self.eta = eta
        self.pathModels = pathModels

    def initRandomW(self, X):
        W_ = list()
        totalMean_ = np.mean(X)
        totalStd_ = np.std(X)
        means_ = np.mean(X, axis=0)
        stds_ = np.std(X, axis=0)
        ramdon_seed = np.random.RandomState()
        W_.append(ramdon_seed.normal(loc=totalMean_, scale=totalStd_))
        for mean, std in zip(means_, stds_):
            W_.append(ramdon_seed.normal(loc=mean, scale=std))
        return np.array(W_)

    def fit(self, trainingX, trainingY, epochs=50):
        if self.W is None:
            self.W = self.initRandomW(trainingX)
        if self.errorForEachEpoch:
            bestError = min(self.errorForEachEpoch)
        else:
            bestError = np.inf

        for _ in range(epochs):
            randomIndices_ = np.arange(trainingX.shape[0])
            np.random.shuffle(randomIndices_)
            for xi, target in zip(trainingX[randomIndices_, :], trainingY[randomIndices_]):
                net_input = self._net_input(xi)
                activation_function = self._activation(net_input)
                errors = target - activation_function
                self.W[1:] += self.eta * errors * xi
                self.W[0] += self.eta * errors
            error_ = self.error(trainingX, trainingY)
            self.errorForEachEpoch.append(error_)
            if error_ < bestError:
                bestError = error_
        return self

    def _net_input(self, X):
        return np.dot(X, self.W[1:]) + self.W[0]

    def _activation(self, p_net_input):
        return p_net_input

    def _quantization(self, p_activation):
        return np.where(p_activation >= 0.0, 1, 0)

    def predict(self, X):
        net_in = self._net_input(X)
        activation = self._activation(net_in)
        output = self._quantization(activation)
        return output

    def error(self, X, Y):
        net_input = self._net_input(X)
        activation = self._activation(net_input)
        return np.mean((Y - activation) ** 2)

    def score(self, X, y):
        output = self.predict(X)
        accuracy = np.mean(output == y)
        return accuracy