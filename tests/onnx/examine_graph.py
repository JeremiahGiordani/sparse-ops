# inspect_onnx_graph.py
import onnx
from onnx import numpy_helper

def print_graph_info(onnx_model_path: str):
    model = onnx.load(onnx_model_path)
    graph = model.graph

    print(f"📦 Model name: {graph.name}")
    print(f"🔢 #Inputs: {len(graph.input)} | #Outputs: {len(graph.output)}")
    print(f"🔧 #Initializers (weights/biases): {len(graph.initializer)}")
    print(f"🧮 #Nodes (ops): {len(graph.node)}\n")

    print("=== Computation Graph ===")
    op_types = []
    for idx, node in enumerate(graph.node):
        print(f"Op #{idx:03d}")
        print(f"  Type      : {node.op_type}")
        op_types.append(node.op_type)
        print(f"  Inputs    : {list(node.input)}")
        print(f"  Outputs   : {list(node.output)}")

        if node.attribute:
            print("  Attributes:")
            for attr in node.attribute:
                # Print key attributes depending on their type
                if attr.type == onnx.AttributeProto.INT:
                    print(f"    {attr.name} : {attr.i}")
                elif attr.type == onnx.AttributeProto.FLOAT:
                    print(f"    {attr.name} : {attr.f}")
                elif attr.type == onnx.AttributeProto.STRING:
                    print(f"    {attr.name} : {attr.s.decode('utf-8')}")
                elif attr.type == onnx.AttributeProto.INTS:
                    print(f"    {attr.name} : {list(attr.ints)}")
                elif attr.type == onnx.AttributeProto.FLOATS:
                    print(f"    {attr.name} : {list(attr.floats)}")
                else:
                    print(f"    {attr.name} : <Unsupported type {attr.type}>")
        print()
    print(f"Unique op types {set(op_types)}")

if __name__ == "__main__":
    print_graph_info("resnet18.onnx")
