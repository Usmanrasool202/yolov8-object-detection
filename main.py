from ultralytics import YOLO
import gradio as gr
import os

# Load YOLOv8x model
model = YOLO("yolov8x.pt")

def detect_objects(image):
    if image is None:
        return None
    results = model(image)
    return results[0].plot()

interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Image(type="numpy", label="Detection Result"),
    title="Object Detection Application",
    description="Image-based object detection using a pretrained YOLOv8x model"
)

if __name__ == "__main__":
    # Disable reload & force stable behavior for exe
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )
