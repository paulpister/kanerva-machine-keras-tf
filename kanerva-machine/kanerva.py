import tensorflow as tf
from .vae import vae_layer
from keras import Model
from keras.layers import Input


def generate_memory(R, U, V):
    """R -> K*C matrix as the mean of M
    U -> K*K matrix that provide covariance between rows of M
    V -> C*C matrix that provide cov between columns
    equivalent to p(vec(M)) = N(vec(M)|vec(R), V (x) U)
    V = Ic"""
    R = tf.reshape(R, [-1])
    kron_V_U = tf.contrib.kfac.utils.kronecker_product(V, U)
    memory_vectorized = tf.contrib.distributions.MultivariateNormalFull(R, kron_V_U)
    memory = tf.reshape(memory_vectorized, M.shape)
    return memory


def read(y, memory, addresses):
    b = MLP(y)
    weights = tf.transpose(b) * addresses
    read_data = tf.mean(tf.transpose(weights) * memory)
    return weights, read_data


def writing(images, memory, R, U):
    """online writing"""
    for t in range(len(images)):
        y_t, _ = vae_layer(images[t], filters=16, code_size=30)
        b_t = MLP(y_t)
        w_t = tf.transpose(b_t) * A
        _, z_t, _, z_logvar = vae_layer(images[t], filters=16, code_size=30)

        delta = z_t - w_t * R
        sigma_c = w_t * U
        sigma_z = w_t * U * tf.transpose(w_t) + tf.exp(z_logvar)
        R = R + tf.transpose(sigma_c) * tf.matrix_inverse(sigma_z) * delta
        U = U - tf.transpose(sigma_c) * tf.matrix_inverse(sigma_z) * sigma_c
    memory = generate_memory(R, U)


keys_size = 30,
input_size = 786,
memory_size = 100,
hidden_size = 400,
representation_size = 20,
batch_size = 50

addresses = tf.random_normal(batch_size, keys_size, representation_size)

R = tf.random_normal(memory_size, memory_size)
U = tf.random_normal(keys_size, keys_size)
V = tf.eye(memory_size, memory_size)
memory = generate_memory(R, U, V)
decoded, y, y_mu, y_logvar = vae_layer(input)
w, read_data = read(y, memory, addresses)
z_input = tf.concat(input, read_data, dim=-1)
reconstructed, z, z_mu, z_logvar = vae_layer(z_input, filters=16, code_size=30)
writing(memory, z, w, R, U)


def get_model():
    input = Input(input_size)
    memory = generate_memory(R, U, V)
    decoded, y, y_mu, y_logvar = vae_layer(input)
    w, read_data = read(y, memory, addresses)
    z_input = tf.concat(input, read_data, dim=-1)
    reconstructed, z, z_mu, z_logvar = vae_layer(z_input, filters=16, code_size=30)
    writing(memory, z, w, R, U)

    return Model(input, reconstructed)



model = get_model()

model.compile('adam', 'loss_to_be_implemented')
input = """load omniglot"""
model.fit(input, reconstructed, callbacks=[TensorBoard()], batch_size=batch_size)
