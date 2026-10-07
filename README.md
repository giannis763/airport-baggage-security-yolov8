# Airport Baggage Prohibited Item Detection using YOLOv8

An automated computer vision system designed to detect prohibited and suspicious items in airport security X-ray baggage scans.

### Overview
Automating baggage inspection in aviation security reduces human error and inspection bottlenecks. This project uses the YOLOv8 object detection framework, fine-tuned on X-ray baggage imagery (GDXray dataset), to identify and localize restricted items in real time.

### Key Features
* Object Detection: Real-time detection and bounding-box localization using YOLOv8.
* Dataset: Trained and validated on X-ray transmission radiographs.
* Interface: Application interface for running inference on baggage scan inputs.

### Project Structure
* `App/` — Web/desktop application interface and execution scripts.
* `Code/` — Model training, preprocessing, and evaluation notebooks.
* `Model/` — Model configuration and exported artifacts.
* `Report/` — Project report, methodology, and performance metrics.

### Getting Started
1. Clone the repository:
   git clone https://github.com/giannis763/airport-baggage-security-yolov8.git
   cd airport-baggage-security-yolov8

2. Install dependencies:
   pip install -r requirements.txt

3. Run the application:
   cd App
   python app.py
