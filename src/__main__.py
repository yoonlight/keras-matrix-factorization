from dataset import movielens
from src.model import MF
from keras import metrics, layers, callbacks
import numpy as np
from sklearn.model_selection import train_test_split
from common.plot import plot

X, y, num_words_dict, columns, _ = movielens.load_data(implicit=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

user_dim = num_words_dict["user_id"]
item_dim = num_words_dict["movie_id"]

inputs = [layers.Input((1, )), layers.Input((1, ))]

model = MF(user_dim=user_dim, item_dim=item_dim)
model.compile(optimizer="SGD", loss="binary_crossentropy", metrics=[
              metrics.RootMeanSquaredError(), metrics.AUC()])

model.summary(inputs=inputs, expand_nested=True)
model.plot_model(inputs=inputs)

history = model.fit([X_train["user_id"], X_train["movie_id"]], y_train,
                    epochs=50, validation_split=0.33, batch_size=100, verbose=1, callbacks=[callbacks.EarlyStopping(patience=5)])

test_results = model.evaluate([X_test["user_id"], X_test["movie_id"]], y_test)

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']
x_len = np.arange(len(y_loss))
y_test_loss = np.full(x_len.shape, test_results[0])

plot(x_len, y_loss, y_val_loss, y_test_loss)
