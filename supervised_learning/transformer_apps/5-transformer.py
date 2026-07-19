#!/usr/bin/env python3
"""Defines the Transformer architecture."""

import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """
    Calculate positional encodings.

    Args:
        max_seq_len: Maximum sequence length.
        dm: Dimensionality of the model.

    Returns:
        A numpy array of shape (max_seq_len, dm).
    """
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dimensions = np.arange(dm)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        (2 * (dimensions // 2)) / np.float32(dm)
    )
    angles = positions * angle_rates

    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    return angles.astype(np.float32)


def sdp_attention(Q, K, V, mask=None):
    """
    Calculate scaled dot-product attention.

    Args:
        Q: Query tensor.
        K: Key tensor.
        V: Value tensor.
        mask: Optional attention mask.

    Returns:
        output: Attention output.
        weights: Attention weights.
    """
    scores = tf.matmul(Q, K, transpose_b=True)

    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scores = scores / tf.math.sqrt(dk)

    if mask is not None:
        scores += mask * -1e9

    weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Defines multi-head attention."""

    def __init__(self, dm, h):
        """
        Initialize multi-head attention.

        Args:
            dm: Dimensionality of the model.
            h: Number of attention heads.
        """
        super().__init__()

        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Split the final dimension into multiple heads.

        Args:
            x: Input tensor.
            batch_size: Current batch size.

        Returns:
            Tensor with separated attention heads.
        """
        x = tf.reshape(
            x,
            (batch_size, -1, self.h, self.depth)
        )

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask=None):
        """
        Apply multi-head attention.

        Args:
            Q: Query tensor.
            K: Key tensor.
            V: Value tensor.
            mask: Optional attention mask.

        Returns:
            output: Multi-head attention output.
            weights: Attention weights.
        """
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        attention, weights = sdp_attention(Q, K, V, mask)

        attention = tf.transpose(
            attention,
            perm=[0, 2, 1, 3]
        )

        concatenated = tf.reshape(
            attention,
            (batch_size, -1, self.dm)
        )

        output = self.linear(concatenated)

        return output, weights


class EncoderBlock(tf.keras.layers.Layer):
    """Defines one Transformer encoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize an encoder block."""
        super().__init__()

        self.mha = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation="relu"
        )
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training=False, mask=None):
        """Apply the encoder block."""
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(
            attention,
            training=training
        )

        output1 = self.layernorm1(x + attention)

        hidden = self.dense_hidden(output1)
        feed_forward = self.dense_output(hidden)
        feed_forward = self.dropout2(
            feed_forward,
            training=training
        )

        return self.layernorm2(output1 + feed_forward)


class DecoderBlock(tf.keras.layers.Layer):
    """Defines one Transformer decoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize a decoder block."""
        super().__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation="relu"
        )
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(
        self,
        x,
        encoder_output,
        training=False,
        look_ahead_mask=None,
        padding_mask=None
    ):
        """Apply the decoder block."""
        attention1, _ = self.mha1(
            x,
            x,
            x,
            look_ahead_mask
        )
        attention1 = self.dropout1(
            attention1,
            training=training
        )

        output1 = self.layernorm1(x + attention1)

        attention2, _ = self.mha2(
            output1,
            encoder_output,
            encoder_output,
            padding_mask
        )
        attention2 = self.dropout2(
            attention2,
            training=training
        )

        output2 = self.layernorm2(output1 + attention2)

        hidden = self.dense_hidden(output2)
        feed_forward = self.dense_output(hidden)
        feed_forward = self.dropout3(
            feed_forward,
            training=training
        )

        return self.layernorm3(output2 + feed_forward)


class Encoder(tf.keras.layers.Layer):
    """Defines the Transformer encoder."""

    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        input_vocab,
        max_seq_len,
        drop_rate=0.1
    ):
        """Initialize the encoder."""
        super().__init__()

        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(
            input_vocab,
            dm
        )
        self.positional_encoding = positional_encoding(
            max_seq_len,
            dm
        )

        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate)
            for _ in range(N)
        ]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training=False, mask=None):
        """Apply the complete encoder."""
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(
                x,
                training=training,
                mask=mask
            )

        return x


class Decoder(tf.keras.layers.Layer):
    """Defines the Transformer decoder."""

    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        target_vocab,
        max_seq_len,
        drop_rate=0.1
    ):
        """Initialize the decoder."""
        super().__init__()

        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(
            target_vocab,
            dm
        )
        self.positional_encoding = positional_encoding(
            max_seq_len,
            dm
        )

        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate)
            for _ in range(N)
        ]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(
        self,
        x,
        encoder_output,
        training=False,
        look_ahead_mask=None,
        padding_mask=None
    ):
        """Apply the complete decoder."""
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask
            )

        return x


class Transformer(tf.keras.layers.Layer):
    """Defines the complete Transformer network."""

    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_seq_input,
        max_seq_target,
        drop_rate=0.1
    ):
        """Initialize the Transformer."""
        super().__init__()

        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate
        )

        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate
        )

        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(
        self,
        inputs,
        target,
        training=False,
        encoder_mask=None,
        look_ahead_mask=None,
        decoder_mask=None
    ):
        """Apply the Transformer network."""
        encoder_output = self.encoder(
            inputs,
            training=training,
            mask=encoder_mask
        )

        decoder_output = self.decoder(
            target,
            encoder_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=decoder_mask
        )

        return self.linear(decoder_output)
