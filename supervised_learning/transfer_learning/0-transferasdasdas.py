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

"""
def preprocess_data(X, Y):
    X = tf.image.resize(X, (160, 160)).numpy()

    # Converts pixel values from 0–255 range to [-1, 1]
    # This matches what MobileNetV2 was trained on during ImageNet training
    X_p = preprocess_input(X)

    # esto es un onehot d toda la vida :v
    Y_p = to_categorical(Y, 10)

    return X_p, Y_p
"""

def preprocess_data(X, Y, batch_size=8, shuffle=True):

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))

    dataset = dataset.map(lambda x, y: (
        preprocess_input(tf.image.resize(x, (160, 160))),
        tf.one_hot(y, 10)[0]  # fix shape (1,) → scalar
    ), num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

cifar10 = tf.keras.datasets.cifar10

(X_train, Y_train), (X_val, Y_val) = cifar10.load_data()

"""
X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
X_val_p, Y_val_p = preprocess_data(X_val, Y_val)
"""
train_ds = preprocess_data(X_train, Y_train)
val_ds = preprocess_data(X_val, Y_val, shuffle=False)

# hasta ahora solo agarre el dataset cifar y le
# cambie el tamaño/formato a las imagenes

base_model = MobileNetV2(
    input_shape=(160, 160, 3),



    # esto le saca la ultima layer
    # porque el modelo MobileNetV2 esta pensado para clasificar entre 1000
    # y el dataset cifar10 tiene 10 clases entonces le hare mi
    # propia layer de clasificacion para 10
    include_top=False,

    # importo pesos ya entrenados de imagenet en lugar de
    # entrar hasta encontrar mis pesos propios
    weights="imagenet",
    alpha=0.5
)
    # el modelo mobilenet ya esta entrenado y es muy bueno para traducir
    # imagenes a vectores lo que vamos a entrenar nosotros es solo la layer
    # que sacamos para clasificar entre 10 clases
base_model.trainable = False


# Extract frozen features from base model
tfeats = base_model.predict(train_ds, verbose=1)
vfeats = base_model.predict(val_ds, verbose=1)

# Build a simple classifier on top of frozen features
inputs = Input(shape=tfeats.shape[1:])
x = GlobalAveragePooling2D()(inputs)
x = Dense(128, activation='relu')(x)
out = Dense(10, activation='softmax')(x)

classifier = Model(inputs, out)

classifier.compile(optimizer=Adam(),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

Y_train_p = to_categorical(Y_train, 10)
Y_val_p = to_categorical(Y_val, 10)

classifier.fit(
    tfeats, Y_train_p,
    validation_data=(vfeats, Y_val_p),
    epochs=10,
    batch_size=8
)