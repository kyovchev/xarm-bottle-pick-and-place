"""
Simple Annotation Tool for Bottle Dataset
Creates COCO format annotations
"""

import cv2
import json
import os
from pathlib import Path


class AnnotationTool:
    def __init__(self, images_dir, output_file='annotations.json'):
        self.images_dir = images_dir
        self.output_file = output_file

        # Load existing annotations if available
        self.load_existing_annotations()

        self.current_image_id = len(self.annotations['images'])
        self.current_annotation_id = len(self.annotations['annotations'])

        # Get all images
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_files.extend(Path(images_dir).glob(ext))

        self.image_files = sorted([str(f) for f in self.image_files])
        self.current_image_index = 0

        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_end_point = None  # Initialize to avoid NoneType error
        self.current_boxes = []  # List of (bbox, class_id, class_name)
        self.current_class = 1  # Default: bottle
        self.class_names = {1: 'bottle', 2: 'cap'}
        self.class_colors = {1: (0, 255, 0), 2: (255, 0, 255)}  # Green for bottle, Magenta for cap

        print(f"Found {len(self.image_files)} images")
        print(f"Loaded {len(self.annotations['images'])} existing annotations")
        print("\nControls:")
        print("  - Click and drag to draw bounding box")
        print("  - Press '1' to switch to BOTTLE class (green)")
        print("  - Press '2' to switch to CAP class (magenta)")
        print("  - Press 'r' to RESET current image (delete all boxes)")
        print("  - Press 'n' for next image")
        print("  - Press 'p' for previous image")
        print("  - Press 'd' to delete last box")
        print("  - Press 's' to save and continue")
        print("  - Press 'q' to quit and save")

    def load_existing_annotations(self):
        """Load existing annotations from file if it exists"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r') as f:
                    self.annotations = json.load(f)
                print(f"✓ Loaded existing annotations from {self.output_file}")
                
                # Validate structure
                if 'categories' not in self.annotations:
                    self.annotations['categories'] = [
                        {'id': 1, 'name': 'bottle'},
                        {'id': 2, 'name': 'cap'}
                    ]
                if 'images' not in self.annotations:
                    self.annotations['images'] = []
                if 'annotations' not in self.annotations:
                    self.annotations['annotations'] = []
                    
            except json.JSONDecodeError:
                print(f"⚠ Could not parse {self.output_file}, starting fresh")
                self.create_fresh_annotations()
        else:
            print(f"No existing annotations found, starting fresh")
            self.create_fresh_annotations()

    def create_fresh_annotations(self):
        """Create fresh annotation structure"""
        self.annotations = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 1, 'name': 'bottle'},
                {'id': 2, 'name': 'cap'}
            ]
        }

    def load_annotations_for_image(self, image_name):
        """Load existing annotations for current image"""
        # Find image info
        image_info = next((img for img in self.annotations['images'] 
                          if img['file_name'] == image_name), None)

        if image_info:
            image_id = image_info['id']
            # Load all annotations for this image
            existing_anns = [ann for ann in self.annotations['annotations'] 
                           if ann['image_id'] == image_id]

            # Convert to current_boxes format
            self.current_boxes = []
            for ann in existing_anns:
                bbox = ann['bbox']
                class_id = ann['category_id']
                class_name = self.class_names.get(class_id, f'class_{class_id}')
                self.current_boxes.append((bbox, class_id, class_name))

            print(f"✓ Loaded {len(self.current_boxes)} existing annotations for {image_name}")
            return True

        return False

    def reset_current_image(self):
        """Reset/delete all annotations for current image"""
        image_path = self.image_files[self.current_image_index]
        image_name = os.path.basename(image_path)

        # Find and remove image info
        self.annotations['images'] = [img for img in self.annotations['images'] 
                                      if img['file_name'] != image_name]

        # Find image_id and remove all related annotations
        image_info = next((img for img in self.annotations['images'] 
                          if img['file_name'] == image_name), None)

        if image_info:
            image_id = image_info['id']
            old_count = len(self.annotations['annotations'])
            self.annotations['annotations'] = [ann for ann in self.annotations['annotations'] 
                                              if ann['image_id'] != image_id]
            removed_count = old_count - len(self.annotations['annotations'])
            print(f"✓ Removed {removed_count} annotations for {image_name}")

        # Clear current boxes
        self.current_boxes = []
        print(f"✓ Reset image: {image_name}")

        # Save immediately
        self.save_annotations()

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_point = (x, y)

            # Add box (x, y, w, h)
            x1, y1 = self.start_point
            x2, y2 = end_point

            # Ensure x1 < x2 and y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            if x2 - x1 > 5 and y2 - y1 > 5:  # Minimum box size
                bbox = [x1, y1, x2 - x1, y2 - y1]
                self.current_boxes.append((bbox, self.current_class, self.class_names[self.current_class]))
                print(f"Added {self.class_names[self.current_class]}: {bbox}")

    def draw_boxes(self, image):
        """Draw all boxes on image"""
        img_copy = image.copy()

        # Draw saved boxes with class-specific colors
        for bbox, class_id, class_name in self.current_boxes:
            x, y, w, h = bbox
            color = self.class_colors[class_id]
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)

            # Label with background
            label = class_name
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_copy, (x, y - label_h - 4), (x + label_w, y), color, -1)
            cv2.putText(img_copy, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw current box being drawn
        if self.drawing and self.start_point and self.current_end_point:
            x1, y1 = self.start_point
            x2, y2 = self.current_end_point
            color = self.class_colors[self.current_class]
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        # Add instructions and status
        cv2.putText(img_copy, f"Image {self.current_image_index + 1}/{len(self.image_files)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_copy, f"Boxes: {len(self.current_boxes)}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show current class
        current_class_name = self.class_names[self.current_class]
        current_class_color = self.class_colors[self.current_class]
        cv2.putText(img_copy, f"Current: {current_class_name}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_class_color, 2)

        # Show if this is a loaded annotation
        image_name = os.path.basename(self.image_files[self.current_image_index])
        existing = any(img['file_name'] == image_name for img in self.annotations['images'])
        if existing:
            cv2.putText(img_copy, "[LOADED]", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return img_copy

    def save_annotations(self):
        """Save annotations to JSON file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        print(f"Annotations saved to {self.output_file}")

    def annotate_current_image(self):
        """Save annotations for current image"""
        image_path = self.image_files[self.current_image_index]
        image_name = os.path.basename(image_path)

        # If no boxes, we can skip this image or remove existing annotations
        if len(self.current_boxes) == 0:
            # Check if image had annotations before
            existing_image = next((img for img in self.annotations['images'] 
                                  if img['file_name'] == image_name), None)
            if existing_image:
                # Remove the image and its annotations
                image_id = existing_image['id']
                self.annotations['images'] = [img for img in self.annotations['images'] 
                                             if img['id'] != image_id]
                self.annotations['annotations'] = [ann for ann in self.annotations['annotations'] 
                                                  if ann['image_id'] != image_id]
                print(f"✓ Removed all annotations for {image_name}")
            else:
                print(f"Skipping {image_name} (no boxes)")
            return

        # Load image to get dimensions
        img = cv2.imread(image_path)
        height, width = img.shape[:2]

        # Check if image already exists in annotations
        existing_image = next((img for img in self.annotations['images'] 
                              if img['file_name'] == image_name), None)

        if existing_image:
            # Update existing image
            image_id = existing_image['id']
            # Remove old annotations for this image
            self.annotations['annotations'] = [ann for ann in self.annotations['annotations'] 
                                              if ann['image_id'] != image_id]
            print(f"✓ Updating annotations for {image_name}")
        else:
            # Add new image info
            image_id = self.current_image_id
            image_info = {
                'file_name': image_name,
                'id': image_id,
                'height': height,
                'width': width
            }
            self.annotations['images'].append(image_info)
            self.current_image_id += 1
            print(f"✓ Adding new annotations for {image_name}")

        # Add annotations with class information
        for bbox, class_id, class_name in self.current_boxes:
            annotation = {
                'id': self.current_annotation_id,
                'image_id': image_id,
                'category_id': class_id,
                'bbox': bbox,  # [x, y, w, h]
                'area': bbox[2] * bbox[3],
                'iscrowd': 0
            }
            self.annotations['annotations'].append(annotation)
            self.current_annotation_id += 1

        print(f"Saved {len(self.current_boxes)} annotations for {image_name}")

        # Print summary
        bottles = sum(1 for _, cid, _ in self.current_boxes if cid == 1)
        caps = sum(1 for _, cid, _ in self.current_boxes if cid == 2)
        print(f"  - Bottles: {bottles}, Caps: {caps}")
        self.current_boxes = []

    def run(self):
        """Main annotation loop"""
        cv2.namedWindow('Annotation Tool')
        cv2.setMouseCallback('Annotation Tool', self.mouse_callback)

        while self.current_image_index < len(self.image_files):
            # Load image
            image_path = self.image_files[self.current_image_index]
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error loading {image_path}")
                self.current_image_index += 1
                continue

            # IMPORTANT: Clear current boxes first before loading
            self.current_boxes = []
            self.current_end_point = None

            # Load existing annotations for this image
            image_name = os.path.basename(image_path)
            self.load_annotations_for_image(image_name)

            while True:
                # Draw and display
                display_image = self.draw_boxes(image)
                cv2.imshow('Annotation Tool', display_image)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('1'):  # Switch to bottle class
                    self.current_class = 1
                    print(f"Switched to: {self.class_names[1]}")

                elif key == ord('2'):  # Switch to cap class
                    self.current_class = 2
                    print(f"Switched to: {self.class_names[2]}")

                elif key == ord('r'):  # Reset current image
                    print(f"\n⚠ Resetting image: {image_name}")
                    self.reset_current_image()
                    # Reload the image to show it's empty
                    continue

                elif key == ord('n'):  # Next image
                    if len(self.current_boxes) > 0:
                        self.annotate_current_image()
                    break

                elif key == ord('p'):  # Previous image
                    if self.current_image_index > 0:
                        if len(self.current_boxes) > 0:
                            self.annotate_current_image()
                        self.current_image_index -= 2  # Will be incremented at end of outer loop
                        break

                elif key == ord('d'):  # Delete last box
                    if self.current_boxes:
                        deleted = self.current_boxes.pop()
                        print(f"Deleted {deleted[2]}: {deleted[0]}")

                elif key == ord('s'):  # Save and continue
                    if len(self.current_boxes) > 0:
                        self.annotate_current_image()
                        self.save_annotations()
                    break

                elif key == ord('q'):  # Quit
                    if len(self.current_boxes) > 0:
                        self.annotate_current_image()
                    self.save_annotations()
                    cv2.destroyAllWindows()
                    return

            self.current_image_index += 1

        print("\nAll images annotated!")
        self.save_annotations()
        cv2.destroyAllWindows()


def split_dataset(annotations_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split annotations into train/val/test sets"""
    import random

    with open(annotations_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]

    # Create image_id sets
    train_ids = {img['id'] for img in train_images}
    val_ids = {img['id'] for img in val_images}
    test_ids = {img['id'] for img in test_images}

    # Split annotations
    train_anns = [ann for ann in data['annotations'] if ann['image_id'] in train_ids]
    val_anns = [ann for ann in data['annotations'] if ann['image_id'] in val_ids]
    test_anns = [ann for ann in data['annotations'] if ann['image_id'] in test_ids]

    # Create split files
    splits = {
        'train_annotations.json': {'images': train_images, 'annotations': train_anns, 'categories': data['categories']},
        'val_annotations.json': {'images': val_images, 'annotations': val_anns, 'categories': data['categories']},
        'test_annotations.json': {'images': test_images, 'annotations': test_anns, 'categories': data['categories']}
    }

    for filename, split_data in splits.items():
        with open(filename, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Created {filename}: {len(split_data['images'])} images, {len(split_data['annotations'])} annotations")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Annotate images for bottle detection')
    parser.add_argument('--images', type=str, required=True, help='Directory with images')
    parser.add_argument('--output', type=str, default='annotations.json', help='Output annotation file')
    parser.add_argument('--split', action='store_true', help='Split into train/val/test after annotation')

    args = parser.parse_args()

    # Run annotation tool
    tool = AnnotationTool(args.images, args.output)
    tool.run()

    # Split dataset if requested
    if args.split:
        print("\nSplitting dataset...")
        split_dataset(args.output)