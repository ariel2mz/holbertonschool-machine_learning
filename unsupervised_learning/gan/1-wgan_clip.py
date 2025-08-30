#!/usr/bin/env python3
"""
asdasdsa
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """
    asdasdasdas
    """
    def __init__(
        self, generator, discriminator, latent_generator, real_examples,
        batch_size=200, disc_iter=2, learning_rate=.005
    ):
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss
        )

        self.discriminator.loss = (
            lambda x, y: tf.reduce_mean(x) - tf.reduce_mean(y)
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """
        sadasdasdas
        """
        if not size:
            size = self.batch_size
        return self.generator(
            self.latent_generator(size), training=training
        )

    def get_real_sample(self, size=None):
        """
        asdasdsa
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        asdasdsad
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                reals = self.get_real_sample()
                fakes = self.get_fake_sample(training=True)

                d_real = self.discriminator(reals, training=True)
                d_fake = self.discriminator(fakes, training=True)

                discr_loss = self.discriminator.loss(d_fake, d_real)

            grads = tape.gradient(
                discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        with tf.GradientTape() as tape:
            fakes = self.get_fake_sample(training=True)
            d_fake = self.discriminator(fakes, training=True)
            gen_loss = self.generator.loss(d_fake)

        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
