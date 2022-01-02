#!/bin/sh
cargo build --release
./target/release/nnx infer ./data/opt-mnist.onnx ./data/mnist-7.png --compare
./target/release/nnx infer ./data/opt-mnist.onnx ./data/mnist-5.jpg --compare
./target/release/nnx infer ./data/opt-squeeze.onnx ./data/coffee.jpg --compare