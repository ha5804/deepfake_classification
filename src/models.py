import torch
from transformers import ViTForImageClassification, ViTImageProcessor


class DeepFakeModel:
    def __init__(self, model_id: str, device: str):
        self.device = device

        self.model = ViTForImageClassification.from_pretrained(model_id)
        self.processor = ViTImageProcessor.from_pretrained(model_id)

        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, pil_images):
        """
        pil_images: List[PIL.Image]
        return: torch.Tensor (N,)
        """
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1)

        # 0: Fake, 1: Real (baseline 기준)
        return probs[:, 1]
