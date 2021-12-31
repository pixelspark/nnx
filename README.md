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
