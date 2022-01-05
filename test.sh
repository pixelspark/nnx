#!/bin/sh
cargo build --release
./target/release/nnx infer ./data/mnist.onnx ./data/mnist-7.png --compare
./target/release/nnx infer ./data/mnist.onnx ./data/mnist-5.jpg --compare
./target/release/nnx infer ./data/squeezenet.onnx ./data/coffee.jpg --compare
./target/release/nnx infer ./data/mobilenetv2-7.onnx ./data/coffee.jpg --fallback --labels ./data/mobilenet-labels.txt --top=1