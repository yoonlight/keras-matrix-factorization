from keras.models import Model, Sequential
from keras import layers
from keras.utils.all_utils import plot_model


def get_input_model(name: str, input_dim: int = 1000, output_dim: int = 64):
    input_model = Sequential([
        layers.Embedding(input_dim=input_dim, output_dim=output_dim,
                         name=f"{name}_embedding"),
        layers.Flatten()
    ], name=f"{name}_model")
    return input_model


class MF(Model):
    def __init__(self, user_dim: int, item_dim: int, output_dim: int = 8):
        super(MF, self).__init__()
        self.user_vector = get_input_model(
            input_dim=user_dim, output_dim=output_dim)
        self.item_vector = get_input_model(
            input_dim=item_dim, output_dim=output_dim)
        self.inner_product = layers.Dot(axes=1)
        self.predict_layer = layers.Dense(
            1, kernel_regularizer="l2", bias_regularizer="l2")

    def call(self, inputs):
        user = self.user_vector(inputs[0])
        item = self.item_vector(inputs[1])
        x = self.inner_product([user, item])
        return self.predict_layer(x)

    def build_graph(self, inputs):
        return Model(inputs=inputs, outputs=self.call(inputs))

    def summary(self, inputs, line_length=None, positions=None, print_fn=None, expand_nested=False):
        return self.build_graph(inputs).summary(expand_nested=expand_nested)

    def plot_model(self, inputs, expand_nested=True):
        plot_model(self.build_graph(inputs), expand_nested=expand_nested)
