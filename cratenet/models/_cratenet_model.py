from tensorflow.keras.layers import Dense, Dropout, Layer, LayerNormalization, Input, \
    BatchNormalization, Add, Activation, LeakyReLU, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow import keras

from ._cratenet_utils import get_unscaled_mae_metric_for_standard_scaler_nout_robust
from ._cratenet_losses import RobustL1LossMultiOut
from sklearn.preprocessing import StandardScaler


def r2(y_true, y_pred):
    """
    "In the best case, the modeled values exactly match the observed values, which results in SSres=0 and R^2=1.
    A baseline model, which always predicts y_bar, will have R^2=0. Models that have worse predictions than this
    baseline will have a negative R^2." https://en.wikipedia.org/wiki/Coefficient_of_determination
    """
    y_pred = y_pred[:, ::2]
    y_bar = K.mean(y_true)
    ssres = K.sum(K.square((y_true - y_pred)))
    sstot = K.sum(K.square(y_true - y_bar))
    return 1 - (ssres / sstot)


def mae(y_true, y_pred):
    y_pred = y_pred[:, ::2]
    return K.mean(K.abs(y_pred - y_true), axis=-1)


class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "projection_dim": self.projection_dim,
            "query_dense": self.query_dense,
            "key_dense": self.key_dense,
            "value_dense": self.value_dense,
            "combine_heads": self.combine_heads
        })
        return config

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "att": self.att,
            "ffn": self.ffn,
            "layernorm1": self.layernorm1,
            "layernorm2": self.layernorm2,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2
        })
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class FractionalEncoder(Layer):
    def __init__(self, d_model, resolution=5000, log10=False):
        super(FractionalEncoder, self).__init__()
        self.d_model = d_model//2
        self.resolution = resolution
        self.log10 = log10

        x = tf.linspace(tf.cast(0, tf.float64), tf.cast(self.resolution - 1, tf.float64), tf.cast(self.resolution, tf.int32))
        x = tf.reshape(x, shape=(self.resolution, 1))
        fraction = tf.linspace(tf.cast(0, tf.float64), tf.cast(self.d_model - 1, tf.float64), tf.cast(self.d_model, tf.int32))
        fraction = tf.reshape(fraction, shape=(1, self.d_model))
        fraction = tf.repeat(fraction, repeats=[self.resolution], axis=0)
        self.pe = tf.zeros(shape=(self.resolution, self.d_model), dtype=tf.float64)

        pe = tf.Variable(self.pe, trainable=False, name="%s/var_pe" % self.name)
        pe[:, 0::2].assign(tf.sin(x / tf.pow(50, 2 * fraction[:, 0::2] / self.d_model)))
        pe[:, 1::2].assign(tf.cos(x / tf.pow(50, 2 * fraction[:, 1::2] / self.d_model)))
        self.pe = pe.value()

        self.frac = tf.Variable(initial_value=tf.zeros((0, 0)), shape=(None, None), trainable=False, name="%s/var_frac" % self.name)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "d_model": self.d_model,
            "resolution": self.resolution,
            "log10": self.log10,
            "pe": self.pe,
            "frac": self.frac
        })
        return config

    def call(self, frac):
        self.frac.assign(frac)
        if self.log10:
            log2 = tf.math.log(self.frac) / tf.math.log(tf.constant(2.))
            frac = 0.0025 * log2**2
            mask = tf.greater(frac, 1)
            frac = tf.where(mask, 1., frac)
            mask = tf.less(frac, 1 / self.resolution)
            frac = tf.where(mask, 1 / self.resolution, frac)
        else:
            mask = tf.less(self.frac, 1/self.resolution)
            frac = tf.where(mask, 1/self.resolution, self.frac)

        frac_idx = tf.round(frac * self.resolution) - 1
        out = tf.gather(self.pe, tf.cast(frac_idx, tf.int32))

        return out


class InputEncoder(Layer):
    def __init__(self, d_model):
        super(InputEncoder, self).__init__()
        self.d_model = d_model
        self.emb_scaler = tf.Variable(initial_value=1., trainable=True, name="%s/var_emb_scaler" % self.name)
        self.pos_scaler = tf.Variable(initial_value=1., trainable=True, name="%s/var_pos_scaler" % self.name)
        self.pos_scaler_log = tf.Variable(initial_value=1., trainable=True, name="%s/var_pos_scaler_log" % self.name)

        self.pe = tf.Variable(initial_value=tf.zeros((0, 0, 0), dtype=tf.float64), shape=(None, None, None),
                              trainable=False, dtype=tf.float64, name="%s/var_pe_1" % self.name)
        self.ple = tf.Variable(initial_value=tf.zeros((0, 0, 0), dtype=tf.float64), shape=(None, None, None),
                               trainable=False, dtype=tf.float64, name="%s/var_ple" % self.name)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "d_model": self.d_model,
            "emb_scaler": self.emb_scaler,
            "pos_scaler": self.pos_scaler,
            "pos_scaler_log": self.pos_scaler_log,
            "pe": self.pe,
            "ple": self.ple
        })
        return config

    def call(self, x, pe_fren, ple_fren):
        x = x * 2**self.emb_scaler

        pe = tf.zeros_like(x, dtype=tf.float64)
        ple = tf.zeros_like(x, dtype=tf.float64)
        pe_scaler = 2**(1-self.pos_scaler)**2
        ple_scaler = 2**(1-self.pos_scaler_log)**2

        self.pe.assign(pe)
        self.pe[:, :, :self.d_model // 2].assign(pe_fren * tf.cast(pe_scaler, tf.float64))
        pe = self.pe.value()

        self.ple.assign(ple)
        self.ple[:, :, self.d_model // 2:].assign(ple_fren * tf.cast(ple_scaler, tf.float64))
        ple = self.ple.value()

        x = tf.cast(x, tf.float64)
        return x + pe + ple


class PropertyAndUncertainty(Layer):
    def __init__(self, maxlen, x_dim, name=None):
        super(PropertyAndUncertainty, self).__init__(name=name)
        self.maxlen = maxlen
        self.x_dim = x_dim

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "maxlen": self.maxlen,
            "x_dim": self.x_dim
        })
        return config

    def call(self, x, amount_input):
        batch_size = tf.shape(x)[0]
        mask = tf.equal(amount_input, tf.constant(0.))
        mask = tf.reshape(mask, shape=(batch_size, self.maxlen, 1))
        mask = tf.repeat(mask, repeats=[self.x_dim], axis=2)
        x = tf.where(mask, 0., x)
        denom = tf.reduce_sum(tf.where(~mask, 1., x), axis=1)
        x = tf.reduce_sum(x, axis=1)
        x = x / denom

        output = x[:, :-1]
        logits = x[:, -1]

        target = tf.sigmoid(logits) * output[:, 0]
        uncertainty = output[:, 1]

        return tf.concat([
            tf.reshape(target, shape=(batch_size, 1)),
            tf.reshape(uncertainty, shape=(batch_size, 1))], axis=1)


class ExtraInjector(Layer):
    def __init__(self, maxlen):
        super(ExtraInjector, self).__init__()
        self.maxlen = maxlen

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "maxlen": self.maxlen
        })
        return config

    def call(self, g, x):
        g_repeated = tf.tile(tf.expand_dims(g, 1), [1, self.maxlen, 1])

        return g_repeated + x


class CraTENet:
    def __init__(self, maxlen=8, embed_dim=200, num_heads=4, ff_dim=2048, num_transformer_blocks=3,
                 activation=LeakyReLU(alpha=1e-2), d_model=512, n_outputs=1, n_heads=1, n_extra_in=0):
        """
        Supports multi-output regression. There will be n_outputs*2 outputs. The first output for an output pair
        is the target, while the second output is the variance. For example, for n_outputs=3, there will be 6 outputs;
        outputs 1, 3, and 5 are the target values, while outputs 2, 4, and 6 are the variances; output 2 is the variance
        for output 1, output 4 is the variance for output 3, etc.

        :param maxlen: the max length of the sequence
        :param embed_dim: Embedding size for each token
        :param num_heads: Number of attention heads
        :param ff_dim: Hidden layer size in feed forward network inside transformer
        :param num_transformer_blocks: the number of transformer blocks
        :param activation: the activation to use for the residual blocks
        :param d_model: the embedder dimensionality
        :param n_outputs: an int or a list, specifying the number of outputs in each output head
        :param n_heads: the number of output heads
        :param n_extra_in: the number of extra features to inject
        """
        self._n_heads = n_heads

        atoms_input = Input(shape=(maxlen, embed_dim,))
        amount_input = Input(shape=(maxlen,))
        inputs = [atoms_input, amount_input]

        x = Dense(d_model)(atoms_input)  # Embedder

        pe = FractionalEncoder(d_model, resolution=5000, log10=False)(amount_input)
        ple = FractionalEncoder(d_model, resolution=5000, log10=True)(amount_input)

        x = InputEncoder(d_model)(x, pe, ple)

        embed_dim = d_model

        for i in range(num_transformer_blocks):
            x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

        if n_extra_in > 0:
            extra_input = Input(shape=(n_extra_in,))
            inputs.append(extra_input)
            extra = Dense(embed_dim, activation="relu")(extra_input)
            x = ExtraInjector(maxlen)(extra, x)

        outputs = []
        for i in range(n_heads):
            n_out = n_outputs
            if type(n_outputs) == list:
                n_out = n_outputs[i]
            x1 = self._residual_block(1024, x, activation)
            x1 = self._residual_block(512, x1, activation)
            x1 = self._residual_block(256, x1, activation)
            x1 = self._residual_block(128, x1, activation)
            x1 = Dense(n_out * 3)(x1)
            outs = []
            for j in range(n_out):
                name = "regr%s" % (i+1) if n_out == 1 else None
                pu = PropertyAndUncertainty(maxlen, 3, name=name)(x1[:, :, j*3:(j*3)+3], amount_input)
                outs.append(pu)

            if n_out > 1:
                out = Concatenate(name="regr%s" % (i+1))(outs)
            else:
                out = outs[0]
            outputs.append(out)

        self._model = Model(inputs=inputs, outputs=outputs)

    def _residual_block(self, units, input, activation):
        x = Dense(units)(input)
        x = BatchNormalization()(x)
        x = Add()([Dense(units)(input), x])
        x = Activation(activation)(x)
        return x

    def train(self, train_x, train_y, test_x, test_y, batch_size=32, step_size=0.001, num_epochs=10,
              loss=RobustL1LossMultiOut, custom_metrics=None, callbacks=None, loss_weights=None):

        if loss_weights is None:
            loss_weights = [1.0] * self._n_heads

        optimizer = Adam(lr=step_size)

        if custom_metrics is None:
            custom_metrics = [[] for _ in range(self._n_heads)]

        all_metrics = []
        for i in range(self._n_heads):
            metrics = [mae, r2]
            metrics.extend(custom_metrics[i])
            all_metrics.append(metrics)

        losses = [loss() for _ in range(self._n_heads)]

        self._model.compile(loss=losses,
                            loss_weights=loss_weights,
                            optimizer=optimizer,
                            metrics=all_metrics)

        validation_data = (test_x, test_y)
        if test_x is None and test_y is None:
            validation_data = None
        self._model.fit(train_x, train_y, batch_size=batch_size, epochs=num_epochs, validation_data=validation_data,
                        callbacks=callbacks)

    def evaluate(self, X, y, verbose):
        return self._model.evaluate(X, y, verbose)

    def predict(self, X):
        return self._model.predict(X)

    def summary(self):
        return self._model.summary()

    def get_keras_model(self):
        return self._model

    @staticmethod
    def load(model_path, compile=False):
        """
        NOTE: this returns an instance of the Keras model, not this class
        """
        unscaled_mae = get_unscaled_mae_metric_for_standard_scaler_nout_robust(StandardScaler())
        return keras.models.load_model(model_path, custom_objects={"r2": r2, "mae": mae, "unscaled_mae": unscaled_mae},
                                       compile=compile)
