import pandas as pd

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN

from sklearn import model_selection

from tensorflow.keras import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import binary_crossentropy

import data_conversion
import matplotlib.pyplot as plt

print("imported libraries!")

agg = data_conversion.get_agg_list()
nagg = data_conversion.get_non_agg_list()

print("generated data!")

aggCount = len(agg)
graphs = agg + nagg
graph_labels = [0 if i < aggCount else 1 for i in range(len(graphs))]


graph_labels_dummies = pd.get_dummies(graph_labels, drop_first=True)

generator = PaddedGraphGenerator(graphs=graphs)

k = 35  # the number of rows for the output tensor
layer_sizes = [32, 32, 32, 1]

dgcnn_model = DeepGraphCNN(
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    k=k,
    bias=False,
    generator=generator,
)
x_inp, x_out = dgcnn_model.in_out_tensors()

x_out = Conv1D(filters=32, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)

x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

x_out = Flatten()(x_out)

x_out = Dense(units=64, activation="relu")(x_out)
x_out = Dropout(rate=0.1)(x_out)

predictions = Dense(units=1, activation="sigmoid")(x_out)

model = Model(inputs=x_inp, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.0001), loss=binary_crossentropy, metrics=["acc"],
)

train_graphs, test_graphs = model_selection.train_test_split(
    graph_labels_dummies, train_size=0.9, test_size=None, shuffle=True
)


gen = PaddedGraphGenerator(graphs=graphs)


train_gen = gen.flow(
    list(train_graphs.index),
    targets=train_graphs.values,
    batch_size=50,
    symmetric_normalization=False,
)

test_gen = gen.flow(
    list(test_graphs.index),
    targets=test_graphs.values,
    batch_size=1,
    symmetric_normalization=False,
)


epochs = 250
history = model.fit(
    train_gen, epochs=epochs, verbose=1, validation_data=test_gen, shuffle=True,
)

sg.utils.plot_history(history)
plt.show()

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
