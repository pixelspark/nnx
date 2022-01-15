#!/bin/sh
cargo build --release
./target/release/nnx infer ./data/mnist.onnx -i Input3=./data/mnist-7.png --compare
./target/release/nnx infer ./data/mnist.onnx -i Input3=./data/mnist-5.jpg --compare
./target/release/nnx infer ./data/tymnist-inferred.onnx -i input=./data/mnist-7.png --compare
./target/release/nnx infer ./data/tymnist-inferred.onnx -i input=./data/mnist-5.jpg --compare
./target/release/nnx infer ./data/squeezenet.onnx -i data=./data/coffee.jpg --labels ./data/squeezenet-labels.txt --top=1
./target/release/nnx infer ./data/mobilenetv2-7.onnx -i input=./data/coffee.jpg --fallback --labels ./data/mobilenet-labels.txt --top=1