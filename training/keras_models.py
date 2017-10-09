from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def example(num_inputs, num_outputs):
    """
    Example Keras model
    """
    model = Sequential()
    model.add(Dense(10, init='glorot_normal', activation='relu', input_dim=num_inputs))
    model.add(Dense(num_outputs, init='glorot_uniform', activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(),
        metrics=[
            'categorical_accuracy',
        ])
    return model
