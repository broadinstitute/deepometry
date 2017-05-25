import numpy
import threading

import keras.utils
import imblearn.over_sampling


class Iterator:
    def __init__(self, n, batch_size, shuffle, seed):
        self.batch_index = 0

        self.total_batches_seen = 0

        self.lock = threading.Lock()

        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        self.reset()

        while True:
            if seed is not None:
                numpy.random.seed(seed + self.total_batches_seen)

            if self.batch_index == 0:
                index_array = numpy.arange(n)

                if shuffle:
                    index_array = numpy.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n

            if n > current_index + batch_size:
                current_batch_size = batch_size

                self.batch_index += 1
            else:
                current_batch_size = n - current_index

                self.batch_index = 0

            self.total_batches_seen += 1

            yield index_array[current_index:current_index + current_batch_size], current_index, current_batch_size

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self, *args, **kwargs):
        pass


class NumpyArrayIterator(Iterator):
    def __init__(self, x, y, generator, batch_size=32, shuffle=False, seed=None):
        self.generator = generator

        self.x = x

        self.y = y

        self.num_classes = len(numpy.unique(y))

        Iterator.__init__(self, len(y), batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        return self.x[index_array], keras.utils.to_categorical(self.y[index_array], self.num_classes)


class NumpyArrayGenerator:
    def __init__(self):
        pass

    def flow(self, x, y, shuffle=False, seed=None):
        ros = imblearn.over_sampling.RandomOverSampler(random_state=seed)

        x_res, y_res = ros.fit_sample(x.reshape((len(x), -1)), y)

        x_res = x_res.reshape(x_res.shape[0], *x.shape[1:])

        return NumpyArrayIterator(x_res, y_res, self, batch_size=32, shuffle=shuffle, seed=seed)
