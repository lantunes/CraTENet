from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow as tf


class RobustL1Loss(Loss):
    """
    Assumes a Laplacian prior, which results in a loss which places an L1 distance on the residuals.

    Based on equation 5 of:
    Goodall, R. E., & Lee, A. A. (2020). Predicting materials properties without
    crystal structure: Deep representation learning from stoichiometry.
    Nature communications, 11(1), 1-9.
    """
    def __init__(self):
        super(RobustL1Loss, self).__init__(name="robustL1")

    def call(self, y_true, y_pred):

        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        y_pred_mean = K.reshape(y_pred[:, 0], shape=(-1, 1))
        y_pred_log_std = K.reshape(y_pred[:, 1], shape=(-1, 1))

        absolute = K.abs(y_pred_mean - y_true)
        loss = K.sqrt(tf.constant(2.0)) * absolute * K.exp(-y_pred_log_std) + y_pred_log_std

        return K.mean(loss, axis=-1)


class RobustL2Loss(Loss):
    """
    Assumes a Gaussian error distribution, which results in a loss which places an L2 distance on the residuals.

    Based on equation 8 of:
    Kendall, A., & Gal, Y. (2017). What uncertainties do we need in bayesian
    deep learning for computer vision?. Advances in neural information processing systems, 30.
    and on equation 10 of:
    Nix, D. A., & Weigend, A. S. (1994, June). Estimating the mean and variance of the target
    probability distribution. In Proceedings of 1994 ieee international conference on
    neural networks (ICNN'94) (Vol. 1, pp. 55-60). IEEE.
    """
    def __init__(self):
        super(RobustL2Loss, self).__init__(name="robustL2")

    def call(self, y_true, y_pred):

        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        y_pred_mean = K.reshape(y_pred[:, 0], shape=(-1, 1))
        # TODO this should be called y_pred_log_var, since for Robust L2 we predict the variance, not the st. dev.
        y_pred_log_std = K.reshape(y_pred[:, 1], shape=(-1, 1))

        diff = (y_pred_mean - y_true)**2
        loss = tf.constant(1/2) * K.exp(-y_pred_log_std) * diff + tf.constant(1/2) * y_pred_log_std

        return K.mean(loss, axis=-1)


class RobustL1LossMultiOut(Loss):
    """
    Assumes a Laplacian prior, which results in a loss which places an L1 distance on the residuals.

    Based on equation 5 of:
    Goodall, R. E., & Lee, A. A. (2020). Predicting materials properties without
    crystal structure: Deep representation learning from stoichiometry.
    Nature communications, 11(1), 1-9.
    """
    def __init__(self):
        super(RobustL1LossMultiOut, self).__init__(name="robustL1nout")

    def call(self, y_true, y_pred):

        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        y_pred_mean = y_pred[:, ::2]
        y_pred_log_std = y_pred[:, 1::2]

        absolute = K.abs(y_pred_mean - y_true)
        loss = K.sqrt(tf.constant(2.0)) * absolute * K.exp(-y_pred_log_std) + y_pred_log_std

        return K.mean(loss, axis=-1)


class RobustL2LossMultiOut(Loss):
    """
    Assumes a Gaussian error distribution, which results in a loss which places an L2 distance on the residuals.

    Based on equation 8 of:
    Kendall, A., & Gal, Y. (2017). What uncertainties do we need in bayesian
    deep learning for computer vision?. Advances in neural information processing systems, 30.
    and on equation 10 of:
    Nix, D. A., & Weigend, A. S. (1994, June). Estimating the mean and variance of the target
    probability distribution. In Proceedings of 1994 ieee international conference on
    neural networks (ICNN'94) (Vol. 1, pp. 55-60). IEEE.
    """
    def __init__(self):
        super(RobustL2LossMultiOut, self).__init__(name="robustL2nout")

    def call(self, y_true, y_pred):

        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        y_pred_mean = y_pred[:, ::2]
        # TODO this should be called y_pred_log_var, since for Robust L2 we predict the variance, not the st. dev.
        y_pred_log_std = y_pred[:, 1::2]

        diff = (y_pred_mean - y_true)**2
        loss = tf.constant(1/2) * K.exp(-y_pred_log_std) * diff + tf.constant(1/2) * y_pred_log_std

        return K.mean(loss, axis=-1)
