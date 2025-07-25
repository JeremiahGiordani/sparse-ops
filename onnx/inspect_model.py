# inspect_model.py
import onnx
from onnx import numpy_helper

# Load the ONNX model
model = onnx.load("simple_model.onnx")

# Check model structure and validity
onnx.checker.check_model(model)
print("✅ Model loaded and checked.")

graph = model.graph

print(graph)
print(type(graph))

# Print basic model info
print(f"\n📌 Model Name: {graph.name}")
print(f"Inputs:")
for inp in graph.input:
    print(f"  - {inp.name} : {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")

print(f"\nOutputs:")
for out in graph.output:
    print(f"  - {out.name} : {[d.dim_value for d in out.type.tensor_type.shape.dim]}")

# Print all nodes (operations)
print(f"\n🧠 Nodes (Operators):")
for node in graph.node:
    print(f"  - {node.op_type}: inputs={node.input} → outputs={node.output}")

# Print initializers (weights, biases)
print(f"\n📦 Initializers (Tensors):")
for init in graph.initializer:
    array = numpy_helper.to_array(init)
    print(f"  - {init.name}: shape={array.shape}, dtype={array.dtype}")
