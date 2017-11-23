from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l2


def example(num_inputs, num_outputs):
    """
    Example Keras model
    """
    model = Sequential()
    model.add(
        Dense(
            10, init="glorot_normal", activation="relu", input_dim=num_inputs))
    model.add(Dense(num_outputs, init="glorot_uniform", activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(),
        metrics=[
            "categorical_accuracy",
        ])
    return model


def smhtt(num_inputs, num_outputs):
    model = Sequential()
    model.add(
        Dense(
            100, init="glorot_normal", activation="tanh",
            input_dim=num_inputs))
    model.add(Dense(num_outputs, init="glorot_uniform", activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Adam(), metrics=[])
    return model


def smhtt_legacy(num_inputs, num_outputs):
    model = Sequential()
    model.add(
        Dense(
            300,
            init="glorot_normal",
            activation="relu",
            W_regularizer=l2(1e-4),
            input_dim=num_inputs))
    model.add(
        Dense(
            300,
            init="glorot_normal",
            activation="relu",
            W_regularizer=l2(1e-4)))
    model.add(
        Dense(
            300,
            init="glorot_normal",
            activation="relu",
            W_regularizer=l2(1e-4)))
    model.add(Dense(num_outputs, init="glorot_uniform", activation="softmax"))
    model.compile(
        loss="mean_squared_error",
        optimizer=Adam(),
        metrics=[
            "categorical_accuracy",
        ])
    return model
