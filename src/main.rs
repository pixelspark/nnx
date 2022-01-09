use async_trait::async_trait;
use ndarray::ArrayBase;
use protobuf::{self, Message};
use std::collections::HashMap;
use std::time::Instant;
use structopt::StructOpt;

use wonnx::onnx::ModelProto;

mod gpu;

mod util;
use util::*;

mod types;
use types::*;

mod info;
use info::info_table;

async fn run() -> Result<(), NNXError> {
	env_logger::init();
	let opt = Opt::from_args();
	let debug = log::log_enabled!(log::Level::Info);

	match opt.cmd {
		Command::Info(info_opt) => {
			// Load the model
			let model_path = info_opt.model.into_os_string().into_string().expect("invalid path");
			let model =
				ModelProto::parse_from_bytes(&std::fs::read(&model_path).expect("ONNX Model path not found.")).expect("Could not deserialize the model");
			let table = info_table(&model);
			table.printstd();
			Ok(())
		}

		Command::Infer(infer_opt) => {
			// Load the model
			let model_path = infer_opt.model.clone().into_os_string().into_string().expect("invalid path");
			let model =
				ModelProto::parse_from_bytes(&std::fs::read(&model_path).expect("ONNX Model path not found.")).expect("Could not deserialize the model");

			let input_name = match &infer_opt.input_name {
				Some(input_name) => input_name.clone(),
				None => model.get_graph().get_input()[0].get_name().to_string(),
			};

			let input_info = model
				.get_graph()
				.get_input()
				.iter()
				.find(|x| x.get_name() == input_name)
				.expect("input not found");
			let input_shape = input_info.input_dimensions();
			if debug {
				log::info!(
					"Using input: {} ({})",
					input_name,
					input_shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("x")
				);
			}

			let mut inputs: HashMap<String, Tensor> = HashMap::new();
			let mut input_shapes = HashMap::new();

			let data: Option<ArrayBase<_, ndarray::IxDyn>> = if let Some(input_image) = &infer_opt.input_image {
				load_image_input(input_image, &input_shape)
			} else {
				None
			};

			if let Some(d) = data {
				let mut shape = input_shape.clone();
				if shape.is_empty() {
					return Err(NNXError::InvalidInputShape);
				}

				// Some models allow us to set the number of items we are throwing at them.
				if shape[0] == 0 {
					shape[0] = 1;
					log::info!("changing first dimension for input {} to {:?}", input_name, shape);
				}

				inputs.insert(input_name.clone(), Tensor { data: d, shape: shape.clone() });
				input_shapes.insert(input_name, shape);
			}

			#[cfg(feature = "cpu")]
			if infer_opt.compare {
				let gpu_backend = Backend::Gpu.for_model(&model_path, &input_shapes).await?;
				let gpu_start = Instant::now();
				if infer_opt.benchmark {
					for _ in 0..100 {
						let _ = gpu_backend.infer(&infer_opt, &inputs, &model).await?;
					}
				}
				let gpu_output = gpu_backend.infer(&infer_opt, &inputs, &model).await?;
				let gpu_time = gpu_start.elapsed();
				log::info!("gpu time: {}ms", gpu_time.as_millis());
				drop(gpu_backend);

				let cpu_backend = Backend::Cpu.for_model(&model_path, &input_shapes).await?;
				let cpu_start = Instant::now();
				if infer_opt.benchmark {
					for _ in 0..100 {
						let _ = cpu_backend.infer(&infer_opt, &inputs, &model).await?;
					}
				}
				let cpu_output = cpu_backend.infer(&infer_opt, &inputs, &model).await?;
				let cpu_time = cpu_start.elapsed();
				log::info!(
					"cpu time: {}ms ({:.2}x gpu time)",
					cpu_time.as_millis(),
					cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
				);
				if gpu_output.len() != cpu_output.len() {
					return Err(NNXError::Comparison(format!(
						"length of GPU result ({}) mismatches CPU result ({})",
						gpu_output.len(),
						cpu_output.len()
					)));
				}

				for i in 0..gpu_output.len() {
					let diff = (gpu_output[i] - cpu_output[i]).abs();
					if diff > 0.00001 {
						return Err(NNXError::Comparison(format!(
							"output element {} differs too much: GPU says {} vs CPU says {} (difference is {})",
							i, gpu_output[i], cpu_output[i], diff
						)));
					}
				}
				if infer_opt.benchmark {
					println!(
						"OK (gpu={}ms, cpu={}ms, {:.2}x)",
						gpu_time.as_millis(),
						cpu_time.as_millis(),
						cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
					);
				} else {
					println!("OK")
				}
				return Ok(());
			}

			let first_result = async {
				let backend = infer_opt.backend.for_model(&model_path, &input_shapes).await?;
				backend.infer(&infer_opt, &inputs, &model).await
			};

			let output = match first_result.await {
				Ok(x) => x,
				Err(e) => {
					#[cfg(feature = "cpu")]
					if infer_opt.fallback {
						match infer_opt.backend.fallback() {
							Some(fallback_backend) => {
								log::warn!("inference with {:?} backend failed: {}", infer_opt.backend, e,);
								log::warn!("trying {:?} backend instead", fallback_backend);
								let fallback_inferer = fallback_backend.for_model(&model_path, &input_shapes).await?;
								fallback_inferer.infer(&infer_opt, &inputs, &model).await?
							}
							None => return Err(e),
						}
					} else {
						return Err(e);
					}

					#[cfg(not(feature = "cpu"))]
					return Err(e);
				}
			};

			// Look up label
			match infer_opt.labels {
				Some(labels_path) => {
					let labels = get_labels(&labels_path);

					let mut probabilities = output.iter().enumerate().collect::<Vec<_>>();
					probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());

					let top = infer_opt.top.unwrap_or(10);
					for i in 0..top.min(labels.len()) {
						if infer_opt.probabilities {
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

			Ok(())
		}
	}
}

#[async_trait]
trait Inferer {
	async fn infer(&self, infer_opt: &InferOptions, inputs: &HashMap<String, Tensor>, model: &ModelProto) -> Result<Vec<f32>, NNXError>;
}

#[cfg(feature = "cpu")]
mod cpu;

impl Backend {
	#[cfg(feature = "cpu")]
	fn fallback(&self) -> Option<Backend> {
		match self {
			#[cfg(feature = "cpu")]
			Backend::Cpu => None,

			Backend::Gpu => {
				#[cfg(feature = "cpu")]
				return Some(Backend::Cpu);

				#[cfg(not(feature = "cpu"))]
				return None;
			}
		}
	}

	async fn for_model(&self, model_path: &str, input_shapes: &HashMap<String, Vec<usize>>) -> Result<Box<dyn Inferer>, NNXError> {
		Ok(match self {
			Backend::Gpu => Box::new(gpu::GPUInferer::new(model_path).await?),
			#[cfg(feature = "cpu")]
			Backend::Cpu => Box::new(cpu::CPUInferer::new(model_path, input_shapes).await?),
		})
	}
}

fn main() -> Result<(), NNXError> {
	pollster::block_on(run())
}
