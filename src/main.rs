use async_trait::async_trait;
use ndarray::Array;
use prettytable::{cell, row, Table};
use protobuf::{self, Message};
use std::time::Instant;
use std::{collections::HashMap, path::Path};
use structopt::StructOpt;
use wonnx::onnx::ModelProto;
use wonnx::utils::Shape;

mod gpu;
mod text;

mod util;
use util::*;

mod types;
use types::*;

mod info;
use info::{info_table, print_graph};

async fn run() -> Result<(), NNXError> {
	env_logger::init();
	let opt = Opt::from_args();

	match opt.cmd {
		Command::Devices => {
			let instance = wgpu::Instance::new(wgpu::Backends::all());
			let adapters = instance.enumerate_adapters(wgpu::Backends::all());
			let mut adapters_table = Table::new();
			adapters_table.add_row(row![b->"Adapter", b->"Vendor", b->"Backend"]);
			for adapter in adapters {
				let info = adapter.get_info();
				adapters_table.add_row(row![info.name, info.vendor, format!("{:?}", info.backend)]);
			}
			adapters_table.printstd();
			Ok(())
		}

		Command::Info(info_opt) => {
			// Load the model
			let model_path = info_opt.model.into_os_string().into_string().expect("invalid path");
			let model =
				ModelProto::parse_from_bytes(&std::fs::read(&model_path).expect("ONNX Model path not found.")).expect("Could not deserialize the model");
			let table = info_table(&model)?;
			table.printstd();
			Ok(())
		}

		Command::Graph(info_opt) => {
			// Load the model
			let model_path = info_opt.model.into_os_string().into_string().expect("invalid path");
			let model =
				ModelProto::parse_from_bytes(&std::fs::read(&model_path).expect("ONNX Model path not found.")).expect("Could not deserialize the model");
			print_graph(&model);
			Ok(())
		}

		Command::Infer(infer_opt) => {
			// Load the model
			let model_path = infer_opt.model.clone().into_os_string().into_string().expect("invalid path");
			let model =
				ModelProto::parse_from_bytes(&std::fs::read(&model_path).expect("ONNX Model path not found.")).expect("Could not deserialize the model");

			let mut inputs: HashMap<String, Tensor> = HashMap::new();

			// Process text inputs
			if !infer_opt.text.is_empty() || !infer_opt.text_mask.is_empty() {
				let tok = text::BertTokenizer::new(Path::new("./data/bertsquad-vocab.txt"));

				// Tokenized text input
				for (text_input_name, text) in &infer_opt.text {
					let text_input_shape = model
						.get_input_shape(text_input_name)?
						.ok_or_else(|| NNXError::InputNotFound(text_input_name.clone()))?;
					let input = tok.get_input_for(text, &text_input_shape)?;
					log::info!("Set {} ({})={}: tokens={:?}", text_input_name, text_input_shape, text, input.data);
					inputs.insert(text_input_name.clone(), input);
				}

				// Tokenized text input
				for (text_input_name, text) in &infer_opt.text_mask {
					let text_input_shape = model
						.get_input_shape(text_input_name)?
						.ok_or_else(|| NNXError::InputNotFound(text_input_name.clone()))?;
					let input = tok.get_mask_input_for(text, &text_input_shape)?;
					log::info!("Set {} ({})={}: mask={:?}", text_input_name, text_input_shape, text, input.data);
					inputs.insert(text_input_name.clone(), input);
				}
			}

			// Process raw inputs
			for (raw_input_name, text) in &infer_opt.raw {
				let raw_input_shape = model
					.get_input_shape(raw_input_name)?
					.ok_or_else(|| NNXError::InputNotFound(raw_input_name.clone()))?;

				let values: Result<Vec<f32>, _> = text.split(',').map(|v| v.parse::<f32>()).collect();
				let mut values = values.map_err(|x| NNXError::TokenizationFailed(x.into()))?;
				values.resize(raw_input_shape.element_count() as usize, 0.0);
				inputs.insert(
					raw_input_name.clone(),
					Tensor {
						data: Array::from_vec(values).into_dyn(),
						shape: raw_input_shape.clone(),
					},
				);
			}

			// Load input image if it was supplied
			for (input_name, image_path) in &infer_opt.input_images {
				let mut input_shape = model.get_input_shape(input_name)?.ok_or_else(|| NNXError::InputNotFound(input_name.clone()))?;

				let data = load_image_input(image_path, &input_shape)?;

				// Some models allow us to set the number of items we are throwing at them.
				if input_shape.dim(0) == 0 {
					input_shape.dims[0] = 1;
					log::info!("changing first dimension for input {} to {:?}", input_name, input_shape);
				}

				inputs.insert(input_name.clone(), Tensor { data, shape: input_shape });
			}

			// Collect input shape data
			let mut input_shapes = HashMap::with_capacity(inputs.len());
			for (k, v) in &inputs {
				input_shapes.insert(k.clone(), v.shape.clone());
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
					let labels = get_lines(&labels_path);

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

	async fn for_model(&self, model_path: &str, input_shapes: &HashMap<String, Shape>) -> Result<Box<dyn Inferer>, NNXError> {
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
