use ndarray::ArrayBase;
use std::{path::PathBuf, str::FromStr};
use structopt::StructOpt;
use thiserror::Error;
use wonnx::{SessionError, WonnxError};

#[cfg(feature = "cpu")]
use tract_onnx::prelude::*;

#[derive(Debug, StructOpt)]
pub struct InfoOptions {
	/// Model file (.onnx)
	#[structopt(parse(from_os_str))]
	pub model: PathBuf,
}

#[derive(Debug, StructOpt)]
pub enum Backend {
	Gpu,
	#[cfg(feature = "cpu")]
	Cpu,
}

pub struct Tensor {
	pub data: ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>,
	pub shape: Vec<usize>,
}

#[derive(Error, Debug)]
pub enum NNXError {
	#[error("invalid backend selected")]
	InvalidBackend(String),

	#[error("input shape is invalid")]
	InvalidInputShape,

	#[error("output not found")]
	OutputNotFound(String),

	#[error("backend error: {0}")]
	BackendFailed(#[from] WonnxError),

	#[error("backend execution error: {0}")]
	BackendExecutionFailed(#[from] SessionError),

	#[cfg(feature = "cpu")]
	#[error("cpu backend error: {0}")]
	CPUBackendFailed(#[from] TractError),

	#[cfg(feature = "cpu")]
	#[error("comparison failed")]
	Comparison(String),
}

impl FromStr for Backend {
	type Err = NNXError;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		match s {
			"gpu" => Ok(Backend::Gpu),
			#[cfg(feature = "cpu")]
			"cpu" => Ok(Backend::Cpu),
			_ => Err(NNXError::InvalidBackend(s.to_string())),
		}
	}
}

#[derive(Debug, StructOpt)]
pub struct InferOptions {
	/// Model file (.onnx)
	#[structopt(parse(from_os_str))]
	pub model: PathBuf,

	#[structopt(long, default_value = "gpu")]
	pub backend: Backend,

	/// Input image
	#[structopt(parse(from_os_str))]
	pub input_image: Option<PathBuf>,

	// Number of labels to print (default: 10)
	#[structopt(long)]
	pub top: Option<usize>,

	/// Whether to print probabilities
	#[structopt(long)]
	pub probabilities: bool,

	/// Node to take output from (defaults to the first output when not specified)
	#[structopt(long)]
	pub output_name: Option<String>,

	/// Node to feed input to (defaults to the first input when not specified)
	#[structopt(long)]
	pub input_name: Option<String>,

	/// Path to a labels file (each line containing a single label)
	#[structopt(short, long, parse(from_os_str))]
	pub labels: Option<PathBuf>,

	#[cfg(feature = "cpu")]
	#[structopt(long)]
	/// Whether to fall back to the CPU backend type if GPU inference fails
	pub fallback: bool,

	#[cfg(feature = "cpu")]
	#[structopt(long, conflicts_with = "backend")]
	/// Compare results of CPU and GPU inference (100 iterations to measure time)
	pub compare: bool,

	#[cfg(feature = "cpu")]
	#[structopt(long, requires = "compare")]
	/// When comparing, perform 100 inferences to measure time
	pub benchmark: bool,
}

#[derive(Debug, StructOpt)]
pub enum Command {
	/// Perform inference using a model and inputs
	Infer(InferOptions),

	/// Show information about a model, such as its inputs, outputs and the ops it uses
	Info(InfoOptions),

	/// Return a GraphViz direct graph of the nodes in the model
	Graph(InfoOptions),
}

#[derive(Debug, StructOpt)]
#[structopt(name = "nnx", about = "GPU-accelerated ONNX inference from the command line")]
pub struct Opt {
	#[structopt(subcommand)]
	pub cmd: Command,
}
