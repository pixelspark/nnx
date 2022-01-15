use image::imageops::FilterType;
use image::{ImageBuffer, Pixel, Rgb};
use ndarray::{s, ArrayBase};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use wonnx::utils::Shape;

use wonnx::onnx::{ModelProto, TensorShapeProto, ValueInfoProto};
pub trait ValueInfoProtoUtil {
	fn input_dimensions(&self) -> Vec<usize>;
}

pub trait TensorShapeProtoUtil {
	fn shape_dimensions(&self) -> Vec<usize>;
}

impl ValueInfoProtoUtil for ValueInfoProto {
	fn input_dimensions(&self) -> Vec<usize> {
		match &self.get_field_type().value {
			Some(x) => match x {
				wonnx::onnx::TypeProto_oneof_value::tensor_type(t) => t.get_shape().shape_dimensions(),
				wonnx::onnx::TypeProto_oneof_value::sequence_type(_) => todo!(),
				wonnx::onnx::TypeProto_oneof_value::map_type(_) => todo!(),
				wonnx::onnx::TypeProto_oneof_value::optional_type(_) => todo!(),
				wonnx::onnx::TypeProto_oneof_value::sparse_tensor_type(_) => todo!(),
			},
			None => vec![],
		}
	}
}

impl TensorShapeProtoUtil for TensorShapeProto {
	fn shape_dimensions(&self) -> Vec<usize> {
		self.get_dim()
			.iter()
			.map(|d| match d.value {
				Some(wonnx::onnx::TensorShapeProto_Dimension_oneof_value::dim_value(i)) => i as usize,
				_ => 0,
			})
			.collect()
	}
}

pub trait ModelProtoUtil {
	fn get_input_shape(&self, input_name: &str) -> Option<Shape>;
}

impl ModelProtoUtil for ModelProto {
	fn get_input_shape(&self, input_name: &str) -> Option<Shape> {
		let value_info = self.get_graph().get_input().iter().find(|x| x.get_name() == input_name);
		value_info.map(|vi| vi.get_shape())
	}
}

// Loads an image as (1,1,w,h) with pixels ranging 0...1 for 0..255 pixel values
fn load_bw_image(image_path: &Path, width: usize, height: usize) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>> {
	let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(image_path)
		.unwrap()
		.resize_exact(width as u32, height as u32, FilterType::Nearest)
		.to_rgb8();

	// Python:
	// # image[y, x, RGB]
	// # x==0 --> left
	// # y==0 --> top

	// See https://github.com/onnx/models/blob/master/vision/classification/imagenet_inference.ipynb
	// for pre-processing image.
	// WARNING: Note order of declaration of arguments: (_,c,j,i)
	ndarray::Array::from_shape_fn((1, 1, width, height), |(_, c, j, i)| {
		let pixel = image_buffer.get_pixel(i as u32, j as u32);
		let channels = pixel.channels();

		// range [0, 255] -> range [0, 1]
		(channels[c] as f32) / 255.0
	})
}

// Loads an image as (1, w, h, 3)
fn load_rgb_image(image_path: &Path, width: usize, height: usize) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>> {
	log::info!("load_rgb_image {:?} {}x{}", image_path, width, height);
	let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(image_path)
		.unwrap()
		.resize_to_fill(width as u32, height as u32, FilterType::Nearest)
		.to_rgb8();

	// Python:
	// # image[y, x, RGB]
	// # x==0 --> left
	// # y==0 --> top

	// See https://github.com/onnx/models/blob/master/vision/classification/imagenet_inference.ipynb
	// for pre-processing image.
	// WARNING: Note order of declaration of arguments: (_,c,j,i)
	let mut array = ndarray::Array::from_shape_fn((1, 3, 224, 224), |(_, c, j, i)| {
		let pixel = image_buffer.get_pixel(i as u32, j as u32);
		let channels = pixel.channels();

		// range [0, 255] -> range [0, 1]
		(channels[c] as f32) / 255.0
	});

	// Normalize channels to mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
	let mean = [0.485, 0.456, 0.406];
	let std = [0.229, 0.224, 0.225];
	for c in 0..3 {
		let mut channel_array = array.slice_mut(s![0, c, .., ..]);
		channel_array -= mean[c];
		channel_array /= std[c];
	}

	// Batch of 1
	array
}

pub fn load_image_input(input_image: &Path, input_shape: &Shape) -> Option<ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>> {
	if input_shape.rank() == 3 {
		let mut w = input_shape.dim(1) as usize;
		let mut h = input_shape.dim(2) as usize;
		if w == 0 {
			w = 224;
		}
		if h == 0 {
			h = 224;
		}

		if input_shape.dim(0) == 3 {
			log::info!("input is (3,?,?), loading as RGB image");
			Some(load_rgb_image(input_image, w, h).into_dyn())
		} else if input_shape.dim(0) == 1 {
			log::info!("input is (1,?,?), loading as BW image");
			Some(load_bw_image(input_image, w, h).into_dyn())
		} else {
			None
		}
	} else if input_shape.rank() == 4 {
		let mut w = input_shape.dim(2) as usize;
		let mut h = input_shape.dim(3) as usize;
		if w == 0 {
			w = 224;
		}
		if h == 0 {
			h = 224;
		}

		if input_shape.dim(1) == 3 {
			log::info!("input is (?,3,?,?), loading as RGB image");
			Some(load_rgb_image(input_image, w, h).into_dyn())
		} else if input_shape.dim(1) == 1 {
			log::info!("input is (?,1,?,?), loading as BW image");
			Some(load_bw_image(input_image, w, h).into_dyn())
		} else {
			None
		}
	} else {
		None
	}
}

pub fn get_lines(path: &Path) -> Vec<String> {
	let file = BufReader::new(File::open(path).unwrap());
	file.lines().map(|line| line.unwrap()).collect()
}
