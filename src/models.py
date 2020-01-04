import tensorflow.keras as keras
import tensorflow.keras.layers as klayers

models_dir = 'models/'
input_name = 'input'
policy_output_name = 'policy_output'
value_output_name = 'value_output'

input_shape = (19, 19, 8)


def basic_model():
    """
    Build and return the most basic model possible
    """
    input = keras.Input(input_shape, name=input_name)
    x = klayers.Conv2D(32, 3, activation='relu', padding='same')(input)
    x = klayers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = klayers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = klayers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = klayers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = klayers.Conv2D(32, 3, activation='relu', padding='same')(x)
    policy_head = klayers.Conv2D(1, 3, activation='relu', padding='same')(x)
    policy_head = klayers.Flatten()(policy_head)
    policy_head = klayers.Dense(
        361, activation='softmax', name=policy_output_name)(policy_head)
    value_head = klayers.Flatten()(x)
    value_head = klayers.Dense(
        1, activation='sigmoid', name=value_output_name)(value_head)

    return keras.Model(inputs=input, outputs=[policy_head, value_head])


def resnet():
    nb_filter = 32

    def convolutional_block(x):
        x = keras.layers.Conv2D(nb_filter, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        return x

    def residual_block(x):
        x_skip = x
        x = keras.layers.Conv2D(nb_filter, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(nb_filter, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([x, x_skip])
        x = keras.layers.ReLU()(x)
        return x

    def policy_head(x):
        x = keras.layers.Conv2D(2, 1)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(361, activation='softmax',
                               name=policy_output_name)(x)
        return x

    def value_head(x):
        x = keras.layers.Conv2D(1, 1)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1, activation='sigmoid',
                               name=value_output_name)(x)
        return x

    x = keras.Input(input_shape, name=input_name)
    input = x

    x = convolutional_block(x)
    for _ in range(32):
        x = residual_block(x)

    policy = policy_head(x)
    value = value_head(x)

    return keras.Model(inputs=input, outputs=[policy, value])
