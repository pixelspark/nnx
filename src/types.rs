use ndarray::ArrayBase;
use std::{path::PathBuf, str::FromStr};
use structopt::StructOpt;
use thiserror::Error;
use wonnx::{utils::Shape, SessionError, WonnxError};

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
	pub shape: Shape,
}

#[derive(Error, Debug)]
pub enum NNXError {
	#[error("invalid backend selected")]
	InvalidBackend(String),

	#[error("input shape is invalid")]
	InvalidInputShape,

	#[error("output not found")]
	OutputNotFound(String),

	#[error("input not found")]
	InputNotFound(String),

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

	#[error("tokenization failed: {0}")]
	TokenizationFailed(Box<dyn std::error::Error + Sync + Send>),
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

/// Parse a single key-value pair
fn parse_key_val<T, U>(s: &str) -> Result<(T, U), Box<dyn std::error::Error>>
where
	T: std::str::FromStr,
	T::Err: std::error::Error + 'static,
	U: std::str::FromStr,
	U::Err: std::error::Error + 'static,
{
	let pos = s.find('=').ok_or_else(|| format!("invalid KEY=value: no `=` found in `{}`", s))?;
	Ok((s[..pos].parse()?, s[pos + 1..].parse()?))
}

#[derive(Debug, StructOpt)]
pub struct InferOptions {
	/// Model file (.onnx)
	#[structopt(parse(from_os_str))]
	pub model: PathBuf,

	#[structopt(long, default_value = "gpu")]
	pub backend: Backend,

	/// Input image
	#[structopt(short = "i", parse(try_from_str = parse_key_val), number_of_values = 1)]
	pub input_images: Vec<(String, PathBuf)>,

	// Number of labels to print (default: 10)
	#[structopt(long)]
	pub top: Option<usize>,

	/// Whether to print probabilities
	#[structopt(long)]
	pub probabilities: bool,

	/// Node to take output from (defaults to the first output when not specified)
	#[structopt(long)]
	pub output_name: Option<String>,

	/// Set an input to tokenized text (-t input_name="some text")
	#[structopt(short = "t", parse(try_from_str = parse_key_val), number_of_values = 1)]
	pub text: Vec<(String, String)>,

	/// Set an input to the mask after tokenizing (-m input_name="some text")
	#[structopt(short = "m", parse(try_from_str = parse_key_val), number_of_values = 1)]
	pub text_mask: Vec<(String, String)>,

	/// Provide raw input (-r input_name=1,2,3,4)
	#[structopt(short = "r", parse(try_from_str = parse_key_val), number_of_values = 1)]
	pub raw: Vec<(String, String)>,

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
	/// List available GPU devices
	Devices,

	/// Perform inference using a model and inputs
	Infer(InferOptions),

	/// Show information about a model, such as its inputs, outputs and the ops it uses
	Info(InfoOptions),

	/// Return a GraphViz direct graph of the nodes in the model
	Graph(InfoOptions),
}

#[derive(Debug, StructOpt)]
#[structopt(name = "nnx", about = "NNX = Neural Network Execute. GPU-accelerated ONNX inference from the command line")]
pub struct Opt {
	#[structopt(subcommand)]
	pub cmd: Command,
}
