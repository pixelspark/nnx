use std::collections::HashMap;

use crate::{InferOptions, Inferer, NNXError};
use async_trait::async_trait;
use tract_onnx::prelude::*;
use wonnx::onnx::ModelProto;

type RunnableOnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct CPUInferer {
	model: RunnableOnnxModel,
}

impl CPUInferer {
	pub async fn new(model_path: &str, model: &ModelProto, input_shapes: &HashMap<String, Vec<usize>>) -> Result<CPUInferer, NNXError> {
		let mut cpu_model = tract_onnx::onnx().model_for_path(&model_path)?;

		for (input_name, input_shape) in input_shapes {
			let input_index = model
				.get_graph()
				.get_input()
				.iter()
				.enumerate()
				.find(|x| x.1.get_name() == *input_name)
				.unwrap_or_else(|| panic!("input not found with name {}", input_name));

			let fact = InferenceFact::dt_shape(f32::datum_type(), input_shape);
			cpu_model.set_input_fact(input_index.0, fact)?;
		}

		let cpu_model = cpu_model.into_optimized()?.into_runnable()?;
		Ok(CPUInferer { model: cpu_model })
	}
}

#[async_trait]
impl Inferer for CPUInferer {
	async fn infer(&self, infer_opt: &InferOptions, inputs: &HashMap<String, crate::Tensor>, model: &ModelProto) -> Result<Vec<f32>, NNXError> {
		let mut cpu_inputs: HashMap<usize, tract_onnx::prelude::Tensor> = HashMap::new();

		for (input_name, input_tensor) in inputs {
			let input_index = model
				.get_graph()
				.get_input()
				.iter()
				.enumerate()
				.find(|x| x.1.get_name() == input_name)
				.unwrap_or_else(|| panic!("input not found with name {}", input_name));
			log::info!("set input fact {} for cpu model (shape: {:?})", input_index.0, input_tensor.shape);

			cpu_inputs.insert(
				input_index.0,
				tract_onnx::prelude::Tensor::from_shape(&input_tensor.shape, input_tensor.data.as_slice().unwrap())?,
			);
		}

		let mut cpu_inputs_ordered = TVec::new();
		for i in 0..inputs.len() {
			cpu_inputs_ordered.push(cpu_inputs.get(&i).unwrap().clone());
		}

		let result = self.model.run(cpu_inputs_ordered)?;
		log::info!("cpu result: {:?}", result);

		Ok(match &infer_opt.output_name {
			Some(_output_name) => unimplemented!(),
			None => {
				let first = result[0].clone();
				let av = first.to_array_view()?;
				av.as_slice().unwrap().to_vec()
			}
		})
	}
}
