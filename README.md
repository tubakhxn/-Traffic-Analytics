## dev/creator = tubakhxn

# Traffic Analytics on Roundabout Video (YOLO + Dashboard)

This project performs advanced traffic analytics on a roundabout video using YOLO object detection and tracking. It generates a processed video with:
- Vehicle detection (car, truck, bus, motorcycle)
- Vehicle tracking with unique IDs
- Trajectory lines for each vehicle
- Real-time traffic analytics dashboard (traffic density, vehicle size distribution)

## Features
- **YOLOv8 object detection** for vehicles
- **ByteTrack tracking** for unique vehicle IDs and smooth trajectories
- **Analytics dashboard** with:
  - Traffic density line graph (vehicles per frame)
  - Vehicle size distribution bar chart (small, medium, large)
- **Combined output video** with all overlays and analytics

## How to Fork & Run
1. **Fork this repo** to your own GitHub account.
2. **Clone** your fork locally:
   ```
   git clone https://github.com/yourusername/your-forked-repo.git
   cd your-forked-repo
   ```
3. **Install dependencies** (Python 3.8+ recommended):
   ```
   pip install ultralytics opencv-python numpy matplotlib lap
   ```
4. **Download YOLOv8n model**:
   - Download `yolov8n.pt` from [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and place it in the project folder.
5. **Add your input video** as `video.mp4` in the project folder.
6. **Run the script**:
   ```
   python main.py
   ```
7. **Check the output**:
   - The processed video will be saved as `output.mp4` in the same folder.

## File Structure
- `main.py` — Main script for detection, tracking, analytics, and video generation
- `video.mp4` — Input video file (replace with your own)
- `output.mp4` — Output video with analytics dashboard
- `yolov8n.pt` — YOLOv8n model weights
- `temp_density.png`, `temp_size.png` — Temporary graph images (auto-generated)
- `.venv/` — (optional) Python virtual environment folder

## Credits
- Detection & tracking: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Analytics & dashboard: OpenCV, Matplotlib, NumPy

---

Made with ❤️ by tubakhxn
