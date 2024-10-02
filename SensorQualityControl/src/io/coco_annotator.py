import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
import json


class AnnotationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Annotation Tool")

        # Define a fixed window size for images
        self.window_width = 800
        self.window_height = 600

        # Initialize attributes
        self.image_dir = None
        self.image_list = []
        self.current_image = None
        self.current_image_index = 0
        self.annotations = {"images": [], "annotations": [], "categories": []}
        self.bbox_start = None
        self.bboxes = {}

        # Predefined categories
        self.categories = ["Dogs", "Cats", "Birds", "Rabbits"]

        # Create GUI elements
        self.canvas = tk.Canvas(
            root,
            width=self.window_width,
            height=self.window_height,
            cursor="cross",
            bg="white",
        )
        self.canvas.pack(expand=True, fill="both")

        # Add buttons and dropdown for navigation and saving
        self.btn_open = tk.Button(root, text="Open Directory", command=self.load_images)
        self.btn_open.pack(side="left")

        self.btn_save = tk.Button(
            root, text="Save Annotations", command=self.save_annotations
        )
        self.btn_save.pack(side="left")

        self.btn_prev = tk.Button(root, text="Previous Image", command=self.prev_image)
        self.btn_prev.pack(side="left")

        self.btn_next = tk.Button(root, text="Next Image", command=self.next_image)
        self.btn_next.pack(side="left")

        # Add category dropdown (ComboBox)
        self.category_label = tk.Label(root, text="Select Category:")
        self.category_label.pack(side="left")

        self.category_combobox = ttk.Combobox(root, values=self.categories)
        self.category_combobox.current(0)  # Default selection
        self.category_combobox.pack(side="left")

        # Bind mouse events for drawing bounding boxes
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def load_images(self):
        """Load images from a selected directory."""
        self.image_dir = filedialog.askdirectory(title="Select Image Directory")
        self.image_list = [
            f
            for f in os.listdir(self.image_dir)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ]
        self.load_image(0)

    def load_image(self, index):
        """Load an image at the given index and resize it to fit the window."""
        self.current_image_index = index
        image_path = os.path.join(self.image_dir, self.image_list[index])
        self.current_image = Image.open(image_path)

        # Resize the image to fit within the window size while keeping aspect ratio
        self.current_image.thumbnail((self.window_width, self.window_height))

        # Convert to ImageTk format
        self.photo_image = ImageTk.PhotoImage(self.current_image)

        # Clear previous image and display the new image on the canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW)

        # Load existing bounding boxes for the current image
        self.bboxes.setdefault(self.current_image_index, [])
        for bbox in self.bboxes[self.current_image_index]:
            self.canvas.create_rectangle(
                bbox[0], bbox[1], bbox[2], bbox[3], outline="green", tags="bbox"
            )

    def on_click(self, event):
        """Start drawing a bounding box."""
        self.bbox_start = (event.x, event.y)

    def on_drag(self, event):
        """Show the rectangle as the user drags the mouse."""
        self.canvas.delete("current_bbox")
        self.canvas.create_rectangle(
            self.bbox_start[0],
            self.bbox_start[1],
            event.x,
            event.y,
            outline="red",
            tags="current_bbox",
        )

    def on_release(self, event):
        """Complete the bounding box annotation."""
        bbox_end = (event.x, event.y)
        bbox = self.normalize_bbox(self.bbox_start, bbox_end)

        # Store the bounding box for the current image
        self.bboxes[self.current_image_index].append(bbox)

        # Display the final bounding box in green
        self.canvas.create_rectangle(
            bbox[0], bbox[1], bbox[2], bbox[3], outline="green", tags="bbox"
        )

        # Save the bounding box annotation
        self.save_bbox(bbox)

    def normalize_bbox(self, start, end):
        """Normalize the bounding box coordinates."""
        x1, y1 = min(start[0], end[0]), min(start[1], end[1])
        x2, y2 = max(start[0], end[0]), max(start[1], end[1])
        return [x1, y1, x2, y2]

    def save_bbox(self, bbox):
        """Save the bounding box in COCO format."""
        image_id = self.current_image_index + 1
        annotation = {
            "image_id": image_id,
            "bbox": [
                bbox[0],
                bbox[1],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
            ],  # Format as [x, y, width, height]
            "category_id": self.categories.index(self.category_combobox.get())
            + 1,  # Get category id from ComboBox
            "id": len(self.annotations["annotations"]) + 1,
        }
        self.annotations["annotations"].append(annotation)

    def prev_image(self):
        """Navigate to the previous image."""
        if self.current_image_index > 0:
            self.load_image(self.current_image_index - 1)

    def next_image(self):
        """Navigate to the next image."""
        if self.current_image_index < len(self.image_list) - 1:
            self.load_image(self.current_image_index + 1)

    def save_annotations(self):
        """Save all annotations to a COCO JSON file."""
        for image_index, bboxes in self.bboxes.items():
            image_data = {
                "file_name": self.image_list[image_index],
                "height": self.current_image.height,
                "width": self.current_image.width,
                "id": image_index + 1,
            }
            self.annotations["images"].append(image_data)

        # Add category metadata to the "categories" field if not already present
        for category in self.categories:
            category_id = self.categories.index(category) + 1
            if not any(
                cat["id"] == category_id for cat in self.annotations["categories"]
            ):
                self.annotations["categories"].append(
                    {"id": category_id, "name": category}
                )

        save_path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )
        if save_path:
            with open(save_path, "w") as f:
                json.dump(self.annotations, f, indent=4)
            print(f"Annotations saved to {save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    gui = AnnotationGUI(root)
    root.mainloop()
