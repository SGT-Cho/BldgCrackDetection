import torch
from transformers import SegformerForSemanticSegmentation
import coremltools as ct

# 모델 경로
model_path = "./checkpoints_segformer5/segformer_finetuned_epoch_32.pth"

# 모델 로드
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=2,
    ignore_mismatched_sizes=True,
)
checkpoint = torch.load(model_path, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# TorchScript 변환
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

wrapped_model = WrappedModel(model)
dummy_input = torch.randn(1, 3, 256, 256)  # 입력 텐서 크기
traced_model = torch.jit.trace(wrapped_model, dummy_input)

# CoreML 변환
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=dummy_input.shape, name="input")],
    outputs=[ct.TensorType(name="output")],
)

# CoreML 모델 저장
coreml_model.save("segformer_finetuned.mlpackage")
print("CoreML 모델이 성공적으로 저장되었습니다: segformer_finetuned.mlmodel")
