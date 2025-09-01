#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """sadsadadsa"""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=5,
                 learning_rate=0.0001, gp_weight=10.0):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.gp_weight = gp_weight

        self.gen_optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=0.5, beta_2=0.9
        )
        self.disc_optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=0.5, beta_2=0.9
        )


    def replace_weights(self, gen_h5, disc_h5):
        """asdsadsadsa"""
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)

    def get_fake_sample(self, size=None, training=False):
        """sadsadsadas"""
        if size is None:
            size = self.batch_size
        z = self.latent_generator(size)
        return self.generator(z, training=training)

    def get_real_sample(self, size=None):
        """asdsadsadsadsa"""
        if size is None:
            size = self.batch_size
        idx = tf.random.shuffle(
            tf.range(tf.shape(self.real_examples)[0])
        )[:size]
        return tf.gather(self.real_examples, idx)

    def gradient_penalty(self, real, fake):
        """sadsadsadsadsa"""
        batch = tf.shape(real)[0]
        alpha = tf.random.uniform([batch, 1, 1, 1], 0.0, 1.0)
        interpolated = alpha * real + (1 - alpha) * fake

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        grads = tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp


    def train_step(self, _):
        """sadsadsadsadas"""

        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real = self.get_real_sample()
                fake = self.get_fake_sample(training=True)

                d_real = self.discriminator(real, training=True)
                d_fake = self.discriminator(fake, training=True)

                d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
                gp = self.gradient_penalty(real, fake)
                total_d_loss = d_loss + self.gp_weight * gp

            grads = tape.gradient(
                total_d_loss, self.discriminator.trainable_variables
            )
            self.disc_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as tape:
            fake = self.get_fake_sample(training=True)
            d_fake = self.discriminator(fake, training=True)
            g_loss = -tf.reduce_mean(d_fake)

        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {"disc_loss": total_d_loss, "gen_loss": g_loss}
