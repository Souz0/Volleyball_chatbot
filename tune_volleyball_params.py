# These imports provide file handling and the TensorFlow tools needed to train and test CNN models.
import os
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# These folders contain the training, validation, and test image datasets.
TRAIN_DIR = "img_Dataset_clean/train"
VAL_DIR = "img_Dataset_clean/validation"
TEST_DIR = "img_Dataset_clean/test"

# These files store the tuning results and the best-performing model.
RESULTS_CSV = "hyperparameter_results.csv"
BEST_MODEL_PATH = "best_sports_ball_classifier.keras"

# These settings control the maximum training length and TensorFlow input performance.
EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE

# These lists define the hyperparameter values that will be tested.
IMG_SIZES = [128, 160]
BATCH_SIZES = [16, 32]
LEARNING_RATES = [0.001, 0.0005, 0.0003]
DROPOUT_RATES = [0.3, 0.4, 0.5]


# This loads the train, validation, and test datasets for a chosen image size and batch size.
def load_datasets(img_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode="rgb"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode="rgb"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode="rgb"
    )

    class_names = train_ds.class_names

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# This builds one CNN model using the chosen hyperparameter values.
def build_model(img_size, num_classes, learning_rate, dropout_rate):
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
    ])

    model = keras.Sequential([
        keras.Input(shape=(img_size, img_size, 3)),
        data_augmentation,
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax")
    ])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# This runs one full training and evaluation experiment for a single parameter combination.
def run_experiment(img_size, batch_size, learning_rate, dropout_rate):
    print("\n" + "=" * 70)
    print(
        f"Running experiment: "
        f"IMG_SIZE={img_size}, "
        f"BATCH_SIZE={batch_size}, "
        f"LR={learning_rate}, "
        f"DROPOUT={dropout_rate}"
    )

    train_ds, val_ds, test_ds, class_names = load_datasets(img_size, batch_size)

    model = build_model(
        img_size=img_size,
        num_classes=len(class_names),
        learning_rate=learning_rate,
        dropout_rate=dropout_rate
    )

    # This stops training early if validation loss no longer improves.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    # This reduces the learning rate when validation loss stops improving.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    best_val_acc = max(history.history["val_accuracy"])
    best_train_acc = max(history.history["accuracy"])
    epochs_ran = len(history.history["loss"])

    result = {
        "img_size": img_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
        "epochs_ran": epochs_ran,
        "best_train_accuracy": round(float(best_train_acc), 4),
        "best_val_accuracy": round(float(best_val_acc), 4),
        "final_val_accuracy": round(float(val_acc), 4),
        "final_test_accuracy": round(float(test_acc), 4),
        "final_val_loss": round(float(val_loss), 4),
        "final_test_loss": round(float(test_loss), 4),
        "class_names": ",".join(class_names),
        "model": model,
    }

    print(
        f"Done. Best val acc={best_val_acc:.4f}, "
        f"final val acc={val_acc:.4f}, "
        f"final test acc={test_acc:.4f}"
    )

    return result


# This saves the summary of all hyperparameter experiments to a CSV file.
def save_results_csv(results, csv_path):
    fieldnames = [
        "img_size",
        "batch_size",
        "learning_rate",
        "dropout_rate",
        "epochs_ran",
        "best_train_accuracy",
        "best_val_accuracy",
        "final_val_accuracy",
        "final_test_accuracy",
        "final_val_loss",
        "final_test_loss",
        "class_names",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {k: v for k, v in r.items() if k != "model"}
            writer.writerow(row)


# This tests all parameter combinations, saves the best model, and writes the results file.
def main():
    all_results = []
    best_result = None

    for img_size in IMG_SIZES:
        for batch_size in BATCH_SIZES:
            for learning_rate in LEARNING_RATES:
                for dropout_rate in DROPOUT_RATES:
                    result = run_experiment(
                        img_size=img_size,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        dropout_rate=dropout_rate
                    )

                    all_results.append(result)

                    if best_result is None or result["best_val_accuracy"] > best_result["best_val_accuracy"]:
                        best_result = result
                        result["model"].save(BEST_MODEL_PATH)
                        print(f"Saved new best model to {BEST_MODEL_PATH}")

    save_results_csv(all_results, RESULTS_CSV)

    print("\n" + "=" * 70)
    print("Hyper-parameter search complete.")
    print(f"Results saved to: {RESULTS_CSV}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print("Best configuration:")
    print(
        f"IMG_SIZE={best_result['img_size']}, "
        f"BATCH_SIZE={best_result['batch_size']}, "
        f"LR={best_result['learning_rate']}, "
        f"DROPOUT={best_result['dropout_rate']}"
    )
    print(
        f"Best val acc={best_result['best_val_accuracy']:.4f}, "
        f"final test acc={best_result['final_test_accuracy']:.4f}"
    )


# This runs the hyperparameter search when the file is executed directly.
if __name__ == "__main__":
    main()