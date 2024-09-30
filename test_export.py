import torch
import onnx
import onnxruntime

from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from pathlib import Path
from torch.utils._pytree import tree_map


def test_onnx_export_seq(tmpdir):
    tdm = Seq(
        Mod(lambda x, y: x * y, in_keys=["x", "y"], out_keys=["z"]),
        Mod(lambda z, x: z + x, in_keys=["z", "x"], out_keys=["out"]),
    )
    x = torch.randn(3)
    y = torch.randn(3)
    torch_input = {"x": x, "y": y}
    torch.onnx.dynamo_export(tdm, x=x, y=y)
    onnx_program = torch.onnx.dynamo_export(tdm, **torch_input)

    onnx_input = onnx_program.adapt_torch_inputs_to_onnx(**torch_input)

    path = Path(tmpdir) / "file.onnx"
    onnx_program.save(str(path))

    # to run inference:
    ort_session = onnxruntime.InferenceSession(
        path, providers=["CPUExecutionProvider"]
    )

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    onnxruntime_input = {
        k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)
    }

    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
    torch.testing.assert_close(
        tree_map(torch.as_tensor, onnxruntime_outputs), tdm(x=x, y=y)
    )

def test_export():
    torch._dynamo.reset_code_caches()
    tdm = Seq(
        Mod(lambda x, y: x * y, in_keys=["x", "y"], out_keys=["z"]),
        Mod(lambda z, x: z + x, in_keys=["z", "x"], out_keys=["out"]),
    )
    x = torch.randn(3)
    y = torch.randn(3)
    out = torch.export.export(tdm, args=(), kwargs={"x": x, "y": y})
    torch.export.save(out, "export.pt2")

    # to run inference
    mod = torch.export.load("export.pt2").module()
    x = torch.randn(3)
    y = torch.randn(3)
    print(mod(x=x, y=y))
    
    
if __name__ == "__main__":
    test_onnx_export_seq(".")
    test_export()
    