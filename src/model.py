from keras.models import Sequential
from keras import layers
from keras.utils.all_utils import plot_model
from common.model import Base


def get_input_model(name: str, input_dim: int = 1000, output_dim: int = 64):
    input_model = Sequential([
        layers.Embedding(input_dim=input_dim, output_dim=output_dim,
                         embeddings_regularizer="l2", name=f"{name}_embedding"),
        layers.Flatten()
    ], name=f"{name}_model")
    return input_model


class MF(Base):
    def __init__(self, user_dim: int, item_dim: int, output_dim: int = 8):
        super(MF, self).__init__()
        self.user_vector = get_input_model(name="user",
                                           input_dim=user_dim, output_dim=output_dim)
        self.user_bias = layers.Embedding(
            input_dim=user_dim, output_dim=1, embeddings_regularizer="l2", name="user_bias")
        self.item_vector = get_input_model(name="item",
                                           input_dim=item_dim, output_dim=output_dim)
        self.item_bias = layers.Embedding(
            input_dim=item_dim, output_dim=1, embeddings_regularizer="l2", name="item_bias")
        self.dot_product = layers.Dot(axes=1)
        self.add_layer = layers.Add()
        self.predict_layer = layers.Dense(
            1, activation="sigmoid", kernel_regularizer=L2(l2=0.001), bias_regularizer=L2(l2=0.001))

    def call(self, inputs):
        user = self.user_vector(inputs[0])
        item = self.item_vector(inputs[1])
        user_bias = self.user_bias(inputs[0])
        item_bias = self.item_bias(inputs[1])
        dotted = self.dot_product([user, item])
        x = self.add_layer([dotted, user_bias, item_bias])
        x = layers.Flatten()(x)
        return self.predict_layer(x)
