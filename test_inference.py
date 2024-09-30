
import numpy as np
import onnx
import onnxruntime

class ONNXModule:
    def __init__(self, path: str):
        self.ort_session = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )
    
    def __call__(self, *args):
        
        outputs = self.ort_session.run(None, args)
        return outputs

mod = ONNXModule("file.onnx")
print([k.shape for k in mod.ort_session.get_inputs()])
out = mod(np.zeros(3), np.zeros(3))
print(out)

