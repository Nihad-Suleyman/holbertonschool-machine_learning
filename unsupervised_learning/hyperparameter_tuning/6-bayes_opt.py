#!/usr/bin/env python3
"""Tune a convolutional MNIST classifier with Bayesian optimization."""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import GPyOpt


# Reproducibility
np.random.seed(0)
tf.random.set_seed(0)

X_train = None
y_train = None


def load_dataset():
    """Load and preprocess the MNIST training set."""
    global X_train, y_train

    (images, labels), _ = mnist.load_data()
    X_train = images.astype("float32") / 255.0
    X_train = np.expand_dims(X_train, axis=-1)
    y_train = to_categorical(labels, 10)


def build_model(learning_rate,
                units,
                dropout_rate,
                l2_weight):
    """Build CNN model"""

    model = models.Sequential([
        layers.Conv2D(
            32,
            (3, 3),
            activation='relu',
            input_shape=(28, 28, 1)
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),

        layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_weight)
        ),

        layers.Dropout(dropout_rate),

        layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def objective(params):
    """
    Objective function for Bayesian Optimization
    """

    learning_rate = float(params[0][0])
    units = int(params[0][1])
    dropout_rate = float(params[0][2])
    l2_weight = float(params[0][3])
    batch_size = int(params[0][4])

    model = build_model(
        learning_rate,
        units,
        dropout_rate,
        l2_weight
    )

    checkpoint_name = (
        "checkpoint_lr{:.5f}_u{}_d{:.2f}_l2{:.5f}_b{}.keras".format(
            learning_rate,
            units,
            dropout_rate,
            l2_weight,
            batch_size
        )
    )

    checkpoint = ModelCheckpoint(
        checkpoint_name,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=0
    )

    best_validation_accuracy = max(history.history['val_accuracy'])
    return -best_validation_accuracy


# Hyperparameter bounds
BOUNDS = [
    {
        'name': 'learning_rate',
        'type': 'continuous',
        'domain': (1e-4, 1e-2)
    },
    {
        'name': 'units',
        'type': 'discrete',
        'domain': (32, 64, 128, 256)
    },
    {
        'name': 'dropout_rate',
        'type': 'continuous',
        'domain': (0.1, 0.5)
    },
    {
        'name': 'l2_weight',
        'type': 'continuous',
        'domain': (1e-6, 1e-2)
    },
    {
        'name': 'batch_size',
        'type': 'discrete',
        'domain': (32, 64, 128, 256)
    }
]


def main():
    """Run optimization and save its report and convergence plot."""
    load_dataset()
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=objective,
        domain=BOUNDS,
        acquisition_type='EI',
        exact_feval=True
    )
    optimizer.run_optimization(max_iter=30)

    best_params = optimizer.x_opt
    best_score = -optimizer.fx_opt

    with open('bayes_opt.txt', 'w', encoding='utf-8') as report:
        report.write("Bayesian Optimization Report\n")
        report.write("=" * 40 + "\n\n")
        report.write(
            "Best Validation Accuracy: {:.4f}\n\n".format(best_score)
        )
        report.write("Best Hyperparameters:\n")
        report.write(f"Learning Rate: {best_params[0]}\n")
        report.write(f"Units: {int(best_params[1])}\n")
        report.write(f"Dropout Rate: {best_params[2]}\n")
        report.write(f"L2 Weight: {best_params[3]}\n")
        report.write(f"Batch Size: {int(best_params[4])}\n")

    optimizer.plot_convergence()
    plt.savefig("convergence_plot.png")
    plt.show()

    print("Optimization complete.")
    print("Report saved to bayes_opt.txt")
    print("Convergence plot saved to convergence_plot.png")


if __name__ == "__main__":
    main()
