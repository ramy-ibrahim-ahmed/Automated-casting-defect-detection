import tensorflow as tf


def data_loader(train_dir, test_dir, batch_size, img_shape):
    train_data = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        shuffle=True,
        batch_size=batch_size,
        image_size=img_shape,
    )

    val_data = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        shuffle=True,
        batch_size=batch_size,
        image_size=img_shape,
    )

    def swap_labels(image, label):
        return image, 1 - label

    train_data = train_data.map(swap_labels)
    val_data = val_data.map(swap_labels)

    val_batches = tf.data.experimental.cardinality(val_data)
    test_data = val_data.take(val_batches // 2)
    val_data = val_data.skip(val_batches // 2)

    AUTOTUNE = tf.data.AUTOTUNE
    train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)
    test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)

    print(f"Training batches: {tf.data.experimental.cardinality(train_data)}")
    print(f"Validation batches: {tf.data.experimental.cardinality(val_data)}")
    print(f"Test batches: {tf.data.experimental.cardinality(test_data)}")

    return train_data, val_data, test_data
