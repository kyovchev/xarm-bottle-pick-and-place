"""
Bottle Detection Inference Script
Real-time detection on images or video
"""

import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
from PIL import Image
import argparse


class BottleDetector:
    """Inference class for bottle detection"""

    def __init__(self, model_path, num_classes=3, device=None, confidence_threshold=0.01):
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Load model
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded on {self.device}")

    def load_model(self, model_path):
        """Load trained model"""
        # Create model architecture
        model = fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        return model

    def preprocess_image(self, image):
        """Preprocess image for inference"""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # Convert to tensor
        transform = T.Compose([T.ToTensor()])
        return transform(image)

    def detect(self, image):
        """
        Detect bottles in image

        Args:
            image: numpy array (BGR format from OpenCV)

        Returns:
            boxes: list of [x1, y1, x2, y2]
            scores: list of confidence scores
            labels: list of class labels
        """
        # Preprocess
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.preprocess_image(image_rgb)

        # Inference
        with torch.no_grad():
            predictions = self.model([image_tensor.to(self.device)])

        # Extract predictions
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()

        # Filter by confidence
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        return boxes, scores, labels

    def draw_predictions(self, image, boxes, scores, labels, class_names=None):
        """Draw bounding boxes on image"""
        image_copy = image.copy()

        if class_names is None:
            class_names = {1: 'bottle', 2: 'cap'}

        # Different colors for different classes
        class_colors = {
            1: (0, 255, 0),      # Green for bottle
            2: (255, 0, 255),    # Magenta for cap
        }

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)
            
            # Get color for class
            color = class_colors.get(label, (0, 255, 0))

            # Draw box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)

            # Draw label and score
            class_name = class_names.get(label, f'Class {label}')
            text = f'{class_name}: {score:.2f}'

            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image_copy, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)

            # Text
            cv2.putText(image_copy, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image_copy


def detect_image(detector, image_path, output_path=None, show=True):
    """Detect bottles in a single image"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Detect
    boxes, scores, labels = detector.detect(image)

    # Draw results
    result_image = detector.draw_predictions(image, boxes, scores, labels)

    # Count objects by class
    bottles = sum(1 for label in labels if label == 1)
    caps = sum(1 for label in labels if label == 2)

    # Print results
    print(f"\nDetected {len(boxes)} object(s):")
    print(f"  - Bottles: {bottles}")
    print(f"  - Caps: {caps}")
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box.astype(int)
        class_name = 'bottle' if label == 1 else 'cap'
        print(f"  {class_name.capitalize()} {i+1}: bbox=[{x1}, {y1}, {x2}, {y2}], confidence={score:.3f}")

    # Save result
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"\nResult saved to: {output_path}")

    # Show result
    if show:
        cv2.imshow('Bottle Detection', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result_image


def detect_video(detector, video_path, output_path=None, show=True):
    """Detect bottles in video"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        boxes, scores, labels = detector.detect(frame)

        # Count by class
        bottles = sum(1 for label in labels if label == 1)
        caps = sum(1 for label in labels if label == 2)

        # Draw results
        result_frame = detector.draw_predictions(frame, boxes, scores, labels)

        # Add frame info
        text = f'Frame: {frame_count} | Bottles: {bottles} | Caps: {caps}'
        cv2.putText(result_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write to output
        if output_path:
            out.write(result_frame)

        # Show
        if show:
            cv2.imshow('Bottle Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames")
    if output_path:
        print(f"Output saved to: {output_path}")


def detect_webcam(detector):
    """Real-time detection from webcam"""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        boxes, scores, labels = detector.detect(frame)

        # Count by class
        bottles = sum(1 for label in labels if label == 1)
        caps = sum(1 for label in labels if label == 2)

        # Draw results
        result_frame = detector.draw_predictions(frame, boxes, scores, labels)

        # Add info
        text = f'Bottles: {bottles} | Caps: {caps} | FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}'
        cv2.putText(result_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show
        cv2.imshow('Bottle Detection - Webcam', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Bottle Detection Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--output', type=str, help='Path to output file')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--no-show', action='store_true', help='Do not display results')

    args = parser.parse_args()

    # Create detector
    detector = BottleDetector(
        model_path=args.model,
        confidence_threshold=args.confidence
    )

    # Run detection
    show = not args.no_show

    if args.image:
        detect_image(detector, args.image, args.output, show)
    elif args.video:
        detect_video(detector, args.video, args.output, show)
    elif args.webcam:
        detect_webcam(detector)
    else:
        print("Please specify --image, --video, or --webcam")


if __name__ == '__main__':
    main()
