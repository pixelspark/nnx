use std::collections::HashMap;

use wonnx::onnx::ModelProto;

use async_trait::async_trait;

use crate::{InferOptions, Inferer, NNXError};

pub struct GPUInferer {
	session: wonnx::Session,
}

impl GPUInferer {
	pub async fn new(model_path: &str) -> Result<GPUInferer, NNXError> {
		Ok(GPUInferer {
			session: wonnx::Session::from_path(model_path).await?,
		})
	}
}

#[async_trait]
impl Inferer for GPUInferer {
	async fn infer(&self, infer_opt: &InferOptions, inputs: &HashMap<String, crate::Tensor>, _model: &ModelProto) -> Result<Vec<f32>, NNXError> {
		let input_refs = inputs.iter().map(|(k, v)| (k.clone(), v.data.as_slice().unwrap())).collect();
		let mut result = self.session.run(input_refs).await.expect("run failed");

		let result = match &infer_opt.output_name {
			Some(output_name) => result.remove(output_name).unwrap(),
			None => result.values().next().unwrap().clone(),
		};

		log::info!("gpu result: {:?}", result);
		Ok(result)
	}
}
