use std::collections::{HashMap, HashSet};

use prettytable::{cell, row, table, Table};
use wonnx::onnx::{ModelProto, NodeProto};

use crate::util::ValueInfoProtoUtil;

fn node_identifier(index: usize, node: &NodeProto) -> String {
	format!("\"{} {}\"", index, node.get_op_type())
}

pub fn print_graph(model: &ModelProto) {
	println!("strict digraph {{");
	let graph = model.get_graph();

	let mut outputs: HashMap<String, String> = HashMap::new();
	for (index, node) in graph.get_node().iter().enumerate() {
		for output in node.get_output() {
			outputs.insert(output.clone(), node_identifier(index, node));
		}
	}

	// Special values
	let model_inputs: HashSet<&str> = graph.get_input().iter().map(|v| v.get_name()).collect();
	let initializers: HashSet<&str> = graph.get_initializer().iter().map(|x| x.get_name()).collect();

	for (index, node) in graph.get_node().iter().enumerate() {
		for input in node.get_input() {
			if initializers.contains(input.as_str()) {
				continue;
			}

			if model_inputs.contains(input.as_str()) {
				println!("\t\"Input {}\" -> {}", input, node_identifier(index, node));
			}
			// Find input node
			else if let Some(out_from_node) = outputs.get(input) {
				println!("\t{} -> {}", out_from_node, node_identifier(index, node));
			} else {
				println!("\t\"Unknown: {}\" -> {}", input, node_identifier(index, node));
			}
		}
	}
	println!("}}");
}

pub fn info_table(model: &ModelProto) -> Table {
	let mut inputs_table = Table::new();
	inputs_table.add_row(row![b->"Name", b->"Description", b->"Shape"]);
	let inputs = model.get_graph().get_input();
	for i in inputs {
		inputs_table.add_row(row![
			i.get_name(),
			i.get_doc_string(),
			i.input_dimensions().iter().map(|x| x.to_string()).collect::<Vec<String>>().join("x")
		]);
	}

	let mut outputs_table = Table::new();
	outputs_table.add_row(row![b->"Name", b->"Description", b->"Shape"]);

	let outputs = model.get_graph().get_output();
	for i in outputs {
		outputs_table.add_row(row![
			i.get_name(),
			i.get_doc_string(),
			i.input_dimensions().iter().map(|x| x.to_string()).collect::<Vec<String>>().join("x")
		]);
	}

	let opset_string = model
		.get_opset_import()
		.iter()
		.map(|os| format!("{} {}", os.get_version(), os.get_domain()))
		.collect::<Vec<String>>()
		.join(";");

	// List ops used
	struct Usage {
		attributes: HashSet<String>,
	}

	let mut usage: HashMap<&str, Usage> = HashMap::new();

	let graph = model.get_graph();
	for node in graph.get_node() {
		let usage = if let Some(usage) = usage.get_mut(node.get_op_type()) {
			usage
		} else {
			usage.insert(node.get_op_type(), Usage { attributes: HashSet::new() });
			usage.get_mut(node.get_op_type()).unwrap()
		};

		node.get_attribute().iter().for_each(|a| {
			usage.attributes.insert(a.get_name().to_string());
		});
	}

	let mut usage_table = Table::new();
	usage_table.add_row(row![b->"Op", b->"Attributes"]);
	for (op, usage) in usage.iter() {
		let attrs = usage.attributes.iter().cloned().collect::<Vec<String>>().join(", ");
		usage_table.add_row(row![op, attrs]);
	}

	table![
		[b->"Model version", model.get_model_version()],
		[b->"IR version", model.get_ir_version()],
		[b->"Producer name", model.get_producer_name()],
		[b->"Producer version", model.get_producer_version()],
		[b->"Opsets", opset_string],
		[b->"Inputs", inputs_table],
		[b->"Outputs", outputs_table],
		[b->"Ops used", usage_table]
	]
}
