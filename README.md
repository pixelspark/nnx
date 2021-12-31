# NNX

GPU-accelerated neural network inference from the command line.

Uses [wonnx](https://github.com/haixuanTao/wonnx).

## Usage

```sh
$ nnx ./data/opt-squeeze.onnx --output-name squeezenet0_flatten0_reshape0 --labels ./data/synset.txt --input-image ./data/coffee.png
n03063689 coffeepot: 22.261997
n03297495 espresso maker: 20.724543
n02791124 barber chair: 18.916985
n02841315 binoculars, field glasses, opera glasses: 18.508638
n04254120 soap dispenser: 17.940422
n04560804 water jug: 17.76079
n03764736 milk can: 17.60635
n03976467 Polaroid camera, Polaroid Land camera: 17.103294
n03532672 hook, claw: 16.791483
n03584829 iron, smoothing iron: 16.715918

$ nnx ./data/opt-mnist.onnx --input-image ./data/mnist-7.png
[-1.2942507, 0.5192305, 8.655695, 9.474595, -13.768464, -5.8907413, -23.467274, 28.252314, -6.7598896, 3.9513395]

$ nnx ./data/opt-mnist.onnx --input-image ./data/mnist-7.png --labels ./data/mnist-labels.txt --top=1
Seven
```

- Replace `nnx` with `cargo run --release --` to run development version
- Prepend `RUST_LOG=nnx=info` to see useful information

## End-to-end example with Keras

1. `pip install tensorflow onnx tf2onnx`

2. Create a very simple model for the MNIST digits:

```python
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# train_images will be (60000,28,28) i.e. 60k black-and-white images of 28x28 pixels (which are ints between 0..255)
# train_labels will be (60000,) i.e. 60k integers ranging 0...9
# test_images/test_labels are similar but only have 10k items

# Build model
from tensorflow import keras
from tensorflow.keras import layers

# Convert images to have pixel values as floats between 0...1
train_images_input = train_images.astype("float32") / 255

model = keras.Sequential([
    layers.Reshape((28*28,), input_shape=(28,28)),
    layers.Dense(512, activation = 'relu'),
    layers.Dropout(rate=0.01),
    layers.Dense(10,  activation = 'softmax')
])

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_images_input, train_labels, epochs=20, batch_size=1024)
```

3. Save Keras model to ONNX with inferred dimensions:

```python
import tf2onnx
import tensorflow as tf
import onnx
input_signature = [tf.TensorSpec([1,28,28], tf.float32, name='input')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

from onnx import helper, shape_inference
inferred_model = shape_inference.infer_shapes(onnx_model)

onnx.save(onnx_model, "tymnist.onnx")
onnx.save(inferred_model, "tymnist-inferred.onnx")
```

4. Infer with NNX:

```sh
nnx  ./tymnist-inferred.onnx -i ./data/mnist-7.png
```

5. compare inference result with what Keras would generate (`pip install numpy pillow matplotlib`):

```python
import PIL
import numpy
import matplotlib.pyplot as plt
m5 = PIL.Image.open("data/mnist-7.png").resize((28,28), PIL.Image.ANTIALIAS)
nm5 = numpy.array(m5).reshape((1,28,28))
model.predict(nm5)
```
