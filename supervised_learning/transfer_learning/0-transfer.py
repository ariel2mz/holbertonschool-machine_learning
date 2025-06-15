import tensorflow as tf

# importando todo lo necesario
cifar10 = tf.keras.datasets.cifar10
to_categorical = tf.keras.utils.to_categorical
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
MobileNetV2 = tf.keras.applications.MobileNetV2
Input = tf.keras.Input
Lambda = tf.keras.layers.Lambda
Dense = tf.keras.layers.Dense
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Model = tf.keras.Model
Adam = tf.keras.optimizers.Adam

def preprocess_data(X, Y, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))

    dataset = dataset.map(lambda x, y: (
        preprocess_input(tf.image.resize(x, (160, 160))),
        tf.one_hot(tf.squeeze(y), 10)  # Fixed shape issue
    ), num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Load dataset
(X_train, Y_train), (X_val, Y_val) = cifar10.load_data()

# Create data pipelines
train_ds = preprocess_data(X_train, Y_train)
val_ds = preprocess_data(X_val, Y_val, shuffle=False)

# Create base model (same as before)
base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Instead of extracting all features at once (memory intensive),
# we'll create a new model that combines feature extraction and classification
inputs = Input(shape=(160, 160, 3))
x = base_model(inputs)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

combined_model = Model(inputs, outputs)

combined_model.compile(optimizer=Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

# Train the model directly using the data pipeline
history = combined_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

combined_model.save('cifar10.h5')