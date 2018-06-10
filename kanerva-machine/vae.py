from keras.layers import *
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K

def conv_resnet_block(x, filters):
    x = Conv2D(filters, (4, 4), 2, padding='valid')(x)
    shortcut = x
    x = Conv2D(filters, (3, 3), 2, activation='relu', padding='same')(x)
    x = Conv2D(filters, (3, 3), 2, activation='relu', padding='same')(x)
    x = x + shortcut
    x = Activation('relu')
    return x

def conv_encoder(x, filters):
    x = conv_resnet_block(x, filters)
    x = conv_resnet_block(x, filters)
    x = conv_resnet_block(x, filters)
    return x

def tconv_resnet_block(x, filters)
    x = Conv2DTranspose(filters, (4, 4), 2, padding='valid')(x)
    shortcut = x
    x = Conv2DTranspose(filters, (3, 3), 2, activation='relu', padding='same')(x)
    x = Conv2DTranspose(filters, (3, 3), 2, activation='relu', padding='same')(x)
    x = x + shortcut
    x = Activation('relu')
    return x

def conv_decoder(x, filters):
    x = tconv_resnet_block(x, filters)
    x = tconv_resnet_block(x, filters)
    x = tconv_resnet_block(x, filters)
    return x

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def vae_layer(inputs, filters, code_size):
    x = inputs
    x = conv_encoder(x, filters)

    shape = K.int_shape(x)

    x = Flatten()(x)
    x = Dense(2*code_size, activation='relu')(x)
    z_mean = Dense(code_size, name='z_mean')(x)
    z_log_var = Dense(code_size, name='z_log_var')(x)

    z = Lambda(sampling, output_shape=(code_size,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    #plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(code_size,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    x = conv_decoder(x, filters)

    outputs = Conv2DTranspose(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same',
                                                                              name='decoder_output')

    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    #plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

    outputs = decoder(encoder(inputs)[2])
    return outputs, z, z_mean, z_log_var