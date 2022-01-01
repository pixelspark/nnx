use protobuf::{self, Message};
use std::path::Path;
use std::{
	collections::HashMap,
	fs,
	io::{BufRead, BufReader},
	path::PathBuf,
};
use structopt::StructOpt;
use wonnx::onnx::ModelProto;

mod util;
use util::*;

#[derive(Debug, StructOpt)]
#[structopt(name = "nnx", about = "GPU-accelerated ONNX inference from the command line")]
struct Opt {
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

async fn run() {
	env_logger::init();
	let opt = Opt::from_args();
	let debug = log::log_enabled!(log::Level::Info);

	let model_path = opt.model.into_os_string().into_string().expect("invalid path");
	let model = ModelProto::parse_from_bytes(&std::fs::read(&model_path).expect("ONNX Model path not found.")).expect("Could not deserialize the Model");

	if debug {
		log::info!("Model version: {}", model.get_model_version());
		log::info!("IR version: {}", model.get_ir_version());
		log::info!("Producer name: {}", model.get_producer_name());
		log::info!("Producer version: {}", model.get_producer_version());
		let inputs = model.get_graph().get_input();
		for i in inputs {
			log::info!("Input {} {} {:?}", i.get_name(), i.get_doc_string(), i.input_dimensions());
		}

		for opset in model.get_opset_import() {
			log::info!("Opset: {} {}", opset.get_domain(), opset.get_version());
		}
	}

	let session = wonnx::Session::from_path(&model_path).await.expect("failed to load model");

	if debug {
		log::info!("Outputs: {}", session.outputs.iter().map(|x| x.get_name()).collect::<Vec<&str>>().join(","));
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
	let input_dims = input_info.input_dimensions();
	if debug {
		log::info!(
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
