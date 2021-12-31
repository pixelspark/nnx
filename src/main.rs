use image::imageops::FilterType;
use image::{ImageBuffer, Pixel, Rgb};
use ndarray::s;
use protobuf::{self, Message};
use std::path::Path;
use std::{
	collections::HashMap,
	fs,
	io::{BufRead, BufReader},
	path::PathBuf,
};
use structopt::StructOpt;
use wonnx::onnx::{ModelProto, TensorShapeProto, ValueInfoProto};

#[derive(Debug, StructOpt)]
#[structopt(name = "nnx", about = "GPU-accelerated ONNX inference from the command line")]
struct Opt {
	/// Activate debug mode
	// short and long flags (-d, --debug) will be deduced from the field's name
	#[structopt(short, long)]
	debug: bool,

	/// Model file (.onnx)
	#[structopt(parse(from_os_str))]
	model: PathBuf,

	/// Node to take output from (defaults to the first output when not specified)
	#[structopt(long)]
	output_name: Option<String>,

	/// Node to feed input to (defaults to the first input when not specified)
	#[structopt(long)]
	input_name: Option<String>,

	/// Input image
	#[structopt(short = "i", long, parse(from_os_str))]
	input_image: Option<PathBuf>,

	/// Path to a labels file (each line containing a single label)
	#[structopt(short, long, parse(from_os_str))]
	labels: Option<PathBuf>,

	/// Number of labels to print (default: 10)
	#[structopt(long)]
	top: Option<usize>,

	/// Whether to print probabilities
	#[structopt(long)]
	probabilities: bool,
}

fn get_labels(path: &Path) -> Vec<String> {
	let file = BufReader::new(fs::File::open(path).unwrap());
	file.lines().map(|line| line.unwrap()).collect()
}

fn get_shape_dimensions(info: &TensorShapeProto) -> Vec<usize> {
	info.get_dim()
		.iter()
		.map(|d| match d.value {
			Some(wonnx::onnx::TensorShapeProto_Dimension_oneof_value::dim_value(i)) => i as usize,
			_ => 0,
		})
		.collect()
}

fn get_input_dimensions(info: &ValueInfoProto) -> Vec<usize> {
	match &info.get_field_type().value {
		Some(x) => match x {
			wonnx::onnx::TypeProto_oneof_value::tensor_type(t) => get_shape_dimensions(t.get_shape()),
			wonnx::onnx::TypeProto_oneof_value::sequence_type(_) => todo!(),
			wonnx::onnx::TypeProto_oneof_value::map_type(_) => todo!(),
			wonnx::onnx::TypeProto_oneof_value::optional_type(_) => todo!(),
			wonnx::onnx::TypeProto_oneof_value::sparse_tensor_type(_) => todo!(),
		},
		None => vec![],
	}
}

// Loads an image as (1,1,w,h) with pixels ranging 0...1 for 0..255 pixel values
pub fn load_bw_image(image_path: &Path, width: usize, height: usize) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>> {
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
pub fn load_rgb_image(image_path: &Path, width: usize, height: usize) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>> {
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

async fn run() {
	let opt = Opt::from_args();

	let model_path = opt.model.into_os_string().into_string().expect("invalid path");
	let model = ModelProto::parse_from_bytes(&std::fs::read(&model_path).expect("ONNX Model path not found.")).expect("Could not deserialize the Model");

	if opt.debug {
		println!("Model version: {}", model.get_model_version());
		println!("IR version: {}", model.get_ir_version());
		println!("Producer name: {}", model.get_producer_name());
		println!("Producer version: {}", model.get_producer_version());
		let inputs = model.get_graph().get_input();
		for i in inputs {
			println!("Input {} {} {:?}", i.get_name(), i.get_doc_string(), get_input_dimensions(i));
		}

		for opset in model.get_opset_import() {
			println!("Opset: {} {}", opset.get_domain(), opset.get_version());
		}
	}

	let session = wonnx::Session::from_path(&model_path).await.expect("failed to load model");

	if opt.debug {
		println!("Outputs: {:?}", session.outputs);
	}

	let input_name = match opt.input_name {
		Some(input_name) => input_name,
		None => model.get_graph().get_input()[0].get_name().to_string(),
	};

	let input_info = model
		.get_graph()
		.get_input()
		.iter()
		.find(|x| x.get_name() == input_name)
		.expect("input not found");
	let input_dims = get_input_dimensions(input_info);
	if opt.debug {
		println!(
			"Using input: {} ({})",
			input_name,
			input_dims.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("x")
		);
	}

	let mut inputs = HashMap::<String, &[f32]>::new();

	let data = if let Some(input_image) = opt.input_image {
		if input_dims.len() == 3 {
			if input_dims[0] == 3 {
				log::info!("input is (3,?,?), loading as RGB image");
				Some(load_rgb_image(&input_image, input_dims[1], input_dims[2]))
			} else if input_dims[0] == 1 {
				log::info!("input is (1,?,?), loading as BW image");
				Some(load_bw_image(&input_image, input_dims[1], input_dims[2]))
			} else {
				None
			}
		} else if input_dims.len() == 4 {
			if input_dims[1] == 3 {
				log::info!("input is (?,3,?,?), loading as RGB image");
				Some(load_rgb_image(&input_image, input_dims[2], input_dims[3]))
			} else if input_dims[1] == 1 {
				log::info!("input is (?,1,?,?), loading as BW image");
				Some(load_bw_image(&input_image, input_dims[2], input_dims[3]))
			} else {
				None
			}
		} else {
			None
		}
	} else {
		None
	};

	let result = if let Some(d) = data {
		inputs.insert(input_name, d.as_slice().unwrap());
		session.run(inputs).await.expect("run failed")
	} else {
		session.run(inputs).await.expect("run failed")
	};

	let output = match opt.output_name {
		Some(output_name) => &result[&output_name],
		None => result.values().next().unwrap(),
	};

	// Look up label
	match opt.labels {
		Some(labels_path) => {
			let labels = get_labels(&labels_path);

			let mut probabilities = output.iter().enumerate().collect::<Vec<_>>();
			probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());

			let top = opt.top.unwrap_or(10);
			for i in 0..top.min(labels.len()) {
				if opt.probabilities {
					println!("{}: {}", labels[probabilities[i].0], probabilities[i].1);
				} else {
					println!("{}", labels[probabilities[i].0]);
				}
			}
		}
		None => {
			println!("{:?}", output);
		}
	}
}

fn main() {
	pollster::block_on(run());
}
