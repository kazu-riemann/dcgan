import keras
from keras import layers
import numpy as np

def generator(latent_dim, channels):
    generator_input = keras.Input(shape=(latent_dim,))

    x = layers.Dense(128 * 16 * 16)(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((16, 16, 128))(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)

    generator = keras.models.Model(generator_input, x)
    #.. output summary
    # generator.summary()

    return generator

def discriminator(height, width, channels):
    discriminator_input = \
        layers.Input(shape=(height, width, channels))

    x = layers.Conv2D(128, 3)(discriminator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    discriminator = keras.models.Model(discriminator_input, x)
    #.. output summary
    # discriminator.summary()

    discriminator_optimizer = \
        keras.optimizers.RMSprop( \
        lr=0.0008, \
        clipvalue=1.0, \
        decay=1e-8)
    discriminator.compile( \
        optimizer=discriminator_optimizer, \
        loss='binary_crossentropy')

    return discriminator

def adversarial_network(generator, discriminator):
    discriminator.trainable = False
    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)

    gan_optimizer = \
        keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
    gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

    return gan

#__ train model using cifar10 dataset
def train_model(generator, discriminator, gan, \
                    iterations, batch_size, save_dir):
    import os
    from keras.preprocessing import image

    #.. load cifar10 dataset
    (x_train, y_train), (_, _) = \
        keras.datasets.cifar10.load_data()
    #.. select cat images class 3
    x_train = x_train[y_train.flatten() == 3]
    #.. reshape for training input
    x_train = x_train.reshape( \
        (x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

    #.. start train
    start = 0
    for step in range(iterations):
        #.. generate images randomly by the generator
        random_latent_vectors = \
            np.random.normal(size=(batch_size, latent_dim))
        generated_images = generator.predict(random_latent_vectors)

        #.. prepare input data for the discriminator

        stop = start + batch_size
        real_images = x_train[start: stop]
        #.. concatenate fake images and real images
        combined_images = np.concatenate([generated_images, real_images])
        labels = np.concatenate([np.ones((batch_size, 1)),
                                 np.zeros((batch_size, 1))])
        #.. a trick for training GAN
        labels += 0.05 * np.random.random(labels.shape)

        #.. train the discriminator
        d_loss = discriminator.train_on_batch(combined_images, labels)
        random_latent_vectors = \
            np.random.normal(size=(batch_size, latent_dim))
        misleading_targets = np.zeros((batch_size, 1))
        #.. train the generator
        a_loss = gan.train_on_batch(random_latent_vectors,
                                    misleading_targets)

        #.. output result
        start += batch_size
        if start > len(x_train) - batch_size:
            start = 0
        if step % 100 == 0:
            gan.save_weights('gan.h5')
            print('discriminator loss:', d_loss)
            print('adversarial loss:', a_loss)

            #.. output generated image
            img = image.array_to_img(generated_images[0] * 255., scale=False)
            img.save(os.path.join(save_dir,
                                  'generated_cat' + str(step) + '.png'))
            #.. output real image
            img = image.array_to_img(real_images[0] * 255., scale=False)
            img.save(os.path.join(save_dir,
                                  'real_cat' + str(step) + '.png'))

if __name__ == "__main__":
    #.. network size settings
    latent_dim = 32
    height = 32
    width = 32
    channels = 3

    #.. create model
    gen = generator(latent_dim, channels)
    dis = discriminator(height, width, channels)    
    gan = adversarial_network(gen, dis)

    #.. train model
    iterations = 10000
    batch_size = 20
    save_dir = '.'

    train_model(gen, dis, gan, iterations, \
                    batch_size, save_dir)
