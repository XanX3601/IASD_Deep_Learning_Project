import tensorflow.keras as keras
import tensorflow.keras.layers as klayers

models_dir = 'models/'

input_shape = (19, 19, 8)

def basic_model():
    """
    Build and return the most basic model possible
    """
    input = keras.Input(input_shape, name='input')
    x = klayers.Conv2D(32, 3, activation='relu', padding='same')(input)
    x = klayers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = klayers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = klayers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = klayers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = klayers.Conv2D(32, 3, activation='relu', padding='same')(x)
    policy_head = klayers.Conv2D(1, 3, activation='relu', padding='same')(x)
    policy_head = klayers.Flatten()(policy_head)
    policy_head = klayers.Dense(361, activation='softmax', name='policy')(policy_head)
    value_head = klayers.Flatten()(x)
    value_head = klayers.Dense(1, activation='sigmoid', name='value')(value_head)

    return keras.Model(inputs=input, outputs=[policy_head, value_head])
