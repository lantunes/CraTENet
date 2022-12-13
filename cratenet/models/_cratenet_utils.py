import numpy as np
from scipy import sparse
from pymatgen import Composition
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback


def featurize_comp_for_cratenet(comp, dictionary, embeddings, max_elements, gap=None):
    """
    TODO write docstring
    :param comp: a composition, either as a string or a pymatgen.Composition
    :param dictionary:
    :param embeddings:
    :param max_elements:
    :param gap: a float representing the band gap (optional)
    :return:
    """
    if type(comp) == str:
        comp = Composition(comp)

    unscaled_vectors = np.zeros((max_elements, len(embeddings[0])))
    amounts = np.zeros(max_elements)

    for i, e in enumerate(comp.elements):
        unscaled_vectors[i] = np.array(embeddings[dictionary[e.name]])
        amounts[i] = comp.to_reduced_dict[e.name]

    amounts = amounts / sum(amounts)

    matrix = sparse.coo_matrix(unscaled_vectors.tolist())

    if gap is not None:
        return matrix, amounts, gap
    return matrix, amounts


def get_unscaled_mae_metric_for_standard_scaler_nout_robust(target_scaler):
    """
    TODO write docstring
    :param target_scaler:
    :return:
    """
    def unscaled_mae(y_true, y_pred):
        y_pred = y_pred[:, ::2]
        y_true *= K.constant(target_scaler.scale_)
        y_true += K.constant(target_scaler.mean_)

        y_pred *= K.constant(target_scaler.scale_)
        y_pred += K.constant(target_scaler.mean_)
        return K.mean(K.abs(y_true - y_pred), axis=-1)
    return unscaled_mae


class MultiHeadEarlyStopping(keras.callbacks.Callback):
    def __init__(self, monitor, weights, patience, operator=np.less, loss_init=np.Inf):
        super(MultiHeadEarlyStopping, self).__init__()
        self._monitor = monitor
        self._weights = weights
        self._patience = patience
        self._operator = operator
        self._loss_init = loss_init

    def on_train_begin(self, logs=None):
        self._wait = 0
        self.stopped_epoch = 0
        self._best_loss = self._loss_init

    def on_epoch_end(self, epoch, logs=None):
        losses = [logs.get(loss) for loss in self._monitor]
        tot_loss = sum([self._weights[i] * losses[i] for i in range(len(losses))])
        if self._operator(tot_loss, self._best_loss):
            print("\n**Model improved from %s to %s**" % (self._best_loss, tot_loss))
            self._best_loss = tot_loss
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self._patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True


class LogMetrics(Callback):
    def __init__(self, current_fold):
        super().__init__()
        self.current_fold = current_fold

    def on_epoch_end(self, epoch, logs=None):
        logs["fold"] = self.current_fold
