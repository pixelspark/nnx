# NNX

GPU-accelerated neural network inference from the command line.

## Usage

```sh
cargo run --release -- ./data/opt-squeeze.onnx --output-name squeezenet0_flatten0_reshape0 --labels ./data/synset.txt --input-image ~/Downloads/Unknown.jpg
```
