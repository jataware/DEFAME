import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForObjectDetection

from infact.common.results import ObjectDetectionResult, Evidence
from infact.tools.tool import Tool
from infact.common.action import DetectObjects


class ObjectDetector(Tool):
    name = "object_detector"
    actions = [DetectObjects]
    summarize = False

    def __init__(self, model_name: str = "facebook/detr-resnet-50", **kwargs):
        super().__init__(**kwargs)
        """
        Initialize the ObjectDetector with a pretrained model from Hugging Face.

        :param model_name: The name of the Hugging Face model to use for object detection.
        :param device: The device to run the model on (e.g., -1 for CPU, 0 for GPU).
        :param use_multiple_gpus: Whether to use multiple GPUs if available.
        """
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.device = torch.device(self.device if self.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

    def perform(self, action: DetectObjects) -> Evidence:
        result = self.recognize_objects(action.image.image)
        return [result]  # TODO: Add summary

    def recognize_objects(self, image: Image.Image) -> ObjectDetectionResult:
        """
        Recognize objects in an image.

        :param image: A PIL image.
        :return: An ObjectDetectionResult instance containing recognized objects and their bounding boxes.
        """
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    
            objects = [self.model.module.config.id2label[label.item()] if hasattr(self.model, 'module') else
                       self.model.config.id2label[label.item()] for label in results["labels"]]
            bounding_boxes = [box.tolist() for box in results["boxes"]]

        result = ObjectDetectionResult(
            source=self.model_name,
            objects=objects,
            bounding_boxes=bounding_boxes,
            model_output=outputs)

        self.logger.log(str(result))
        return result
