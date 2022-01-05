# NNX

GPU-accelerated neural network inference from the command line using [wonnx](https://github.com/haixuanTao/wonnx) and [tract](https://github.com/sonos/tract).

ONNX defines a standardized format to exchange machine learning models. However, up to this point there is no easy way to
perform one-off inference using such a model without resorting to Python. Installation of Python and the required libraries
(e.g. TensorFlow and underlying GPU setup) can be cumbersome. Additionally specific code is always needed to transfer
inputs (images, text, etc.) in and out of the formats required by the model (i.e. image classification models want their
images as fixed-size tensors with the pixel values normalized to specific values, et cetera).

This project provides a very simple all-in-one binary command line tool that can be used to perform inference using ONNX
models on the GPU. Thanks to the [wonnx](https://github.com/haixuanTao/wonnx) library inference is performed on the GPU
through [wgpu][https://wgpu.rs], which is a Rust implementation of the WebGPU standard, supported on Windows, macOS, Linux
and (in the future) even inside the browser, without having to install specific drivers (wgpu will use Direct3D, Metal or
Vulkan depending on the platform). NNX will fall back to inference on the CPU (through [tract](https://github.com/sonos/tract))
when compiled with feature 'cpu' (selected by default). It is possible to force using the CPU backend by specifying `--backend=cpu`.
The CPU backend is typically faster for one-shot inference and for relatively small models (the GPU backend will require
compilation of shaders, which comes with more fixed costs).

NNX tries to make educated guesses about how to transform input and output for a model. These guesses are a default - i.e.
it should always be possible to override them. The goal is to reduce the amount of configuration required to be able to
run a model. Currently the following heuristics are applied:

- The first input and first output specified in the ONNX file are used by default.
- Models taking inputs of shape (1,3,w,h) and (3,w,h) will be fed images resized to w\*h with pixel values normalized to
  0...1 (currently we also apply the SqueezeNet normalization)
- Similarly, models taking inputs of shape (1,1,w,h) and (1,w,h) will be fed black-and-white images with pixel values
  normalized to 0...1.
- When a label file is supplied, an output vector of shape (n,) will be interpreted as providing the probabilities for each
  class. The label for each class is taken from the n'th line in the label file.

## Usage

```sh
$ nnx infer ./data/squeezenet.onnx ./data/coffee.png --labels ./data/squeezenet-labels.txt
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

$ nnx infer ./data/mnist.onnx ./data/mnist-7.png
[-1.2942507, 0.5192305, 8.655695, 9.474595, -13.768464, -5.8907413, -23.467274, 28.252314, -6.7598896, 3.9513395]

$ nnx infer ./data/mnist.onnx ./data/mnist-7.png --labels ./data/mnist-labels.txt --top=1
Seven

$ nnx info ./data/mnist.onnx
+------------------+------------------------------------------------------------------+
| Model version    | 1                                                                |
+------------------+------------------------------------------------------------------+
| IR version       | 3                                                                |
+------------------+------------------------------------------------------------------+
| Producer name    | CNTK                                                             |
+------------------+------------------------------------------------------------------+
| Producer version | 2.5.1                                                            |
+------------------+------------------------------------------------------------------+
| Opsets           | 8                                                                |
+------------------+------------------------------------------------------------------+
| Inputs           | +------------------------------------+-------------+-----------+ |
|                  | | Name                               | Description | Shape     | |
|                  | +------------------------------------+-------------+-----------+ |
|                  | | Input3                             |             | 1x1x28x28 | |
|                  | +------------------------------------+-------------+-----------+ |
|                  | | Parameter5                         |             | 8x1x5x5   | |
|                  | +------------------------------------+-------------+-----------+ |
|                  | | Parameter87                        |             | 16x8x5x5  | |
|                  | +------------------------------------+-------------+-----------+ |
|                  | | Pooling160_Output_0_reshape0_shape |             | 2         | |
|                  | +------------------------------------+-------------+-----------+ |
|                  | | Parameter194                       |             | 1x10      | |
|                  | +------------------------------------+-------------+-----------+ |
|                  | | Parameter193_reshape1              |             | 256x10    | |
|                  | +------------------------------------+-------------+-----------+ |
|                  | | 23                                 |             | 8         | |
|                  | +------------------------------------+-------------+-----------+ |
|                  | | 24                                 |             | 16        | |
|                  | +------------------------------------+-------------+-----------+ |
+------------------+------------------------------------------------------------------+
| Outputs          | +------------------+-------------+-------+                       |
|                  | | Name             | Description | Shape |                       |
|                  | +------------------+-------------+-------+                       |
|                  | | Plus214_Output_0 |             | 1x10  |                       |
|                  | +------------------+-------------+-------+                       |
+------------------+------------------------------------------------------------------+
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
nnx  ./tymnist-inferred.onnx infer -i ./data/mnist-7.png --labels ./data/mnist-labels.txt
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

## Testing

Compare output of the CPU and GPU backend as follows:

```sh
nnx infer ./data/squeezenet.onnx ./data/coffee.jpg --compare
```

The result code will be '0' if the results are considered equal (within tolerance) or will be '-1' if it is not. See also
[test.sh](./test.sh).

Specify `--benchmark` to perform 100 inferences, which makes for a fairer comparison between CPU and GPU backend.
