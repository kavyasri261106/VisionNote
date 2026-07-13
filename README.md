# VisionNote – AI-Powered Currency Note Recognition for Visually Impaired

## Team Details
| Role | Name | GitHub |
|---|---|---|
| Team Lead | *Karthiga K* | *karthiga* |
| Member | *Kavya Sri D* | *kavyasri261106* |
| Member | *Shahidha Banu S* | *shahidhabanu* |


## Problem Statement
Visually impaired individuals often cannot independently verify the denomination of a currency note, making them dependent on others during cash transactions. This creates a risk of mistakes, fraud, and loss of financial independence.

## Objective
Build a real-time, offline-capable computer vision system that detects Indian currency notes through a camera and announces the denomination out loud, restoring independence and confidence during cash transactions.

## Proposed Solution
A Streamlit-based application that:
1. Captures a live camera feed.
2. Runs a fine-tuned YOLOv8n model to detect and classify the note (₹10 / ₹20 / ₹50 / ₹100 / ₹200 / ₹500).
3. Announces the result via offline text-to-speech (English, Hindi, or Tamil).
4. Withholds announcement if confidence is below threshold, and instead prompts the user to reposition — avoiding the risk of confidently announcing a wrong denomination.
5. Optionally detects multiple notes at once and reads out the total sum.

## Technologies Used
| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend / UI | Streamlit |
| Computer Vision | OpenCV |
| Object Detection | YOLOv8 (Ultralytics) |
| Text-to-Speech | pyttsx3 (fully offline) |
| Image Processing | NumPy |
| Model Training | Ultralytics YOLO CLI/API |

## Dataset
- Self-collected images of Indian currency notes: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500.
- Captured under varied conditions: front/back side, different lighting, folded, rotated, partially visible, varied backgrounds.
- Target: ~300–500 images per class (sufficient for a hackathon-grade prototype).
- Augmentation applied: rotation, flipping (where orientation-appropriate), brightness/contrast adjustment, scaling.
- Labeled in [Roboflow](https://roboflow.com) or LabelImg, exported in YOLOv8 format.


## Methodology / Model Architecture
- **Base model:** YOLOv8n (nano) — chosen for its speed and small footprint, making it suitable for real-time webcam inference on modest hardware.
- **Fine-tuning:** Transfer learning from COCO-pretrained weights onto the custom currency dataset (`train.py`).
- **Confidence gating:** Detections below 85% confidence are not announced; the app instead gives feedback like "move closer" or "improve lighting" (see `CONFIDENCE_THRESHOLD` in `detect.py`).
- **Pipeline:** Camera frame → YOLOv8 inference → best detection selected → bounding box + label drawn on screen → phrase looked up per selected language → spoken via offline TTS on a background thread (so speech never blocks the video feed).

## Installation & Setup Instructions
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd VisionNote

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your trained model weights
#    Place your fine-tuned weights at: model/trained_model.pt
#    (See "Training" below if you haven't trained one yet.)
```

### Training your own model
```bash
# Place your labeled dataset under dataset/ following the YOLO format
# described in dataset/data.yaml, then run:
python train.py --epochs 60 --imgsz 640 --batch 16

# Best weights will be saved to:
#   runs/detect/visionnote_run/weights/best.pt
# Copy that file to model/trained_model.pt
```

## Usage Instructions
```bash
streamlit run app.py
```
1. Open the local URL Streamlit prints (usually `http://localhost:8501`).
2. Select your preferred voice language (English / Hindi / Tamil).
3. Optionally enable "Multiple note mode" to detect and sum several notes at once.
4. Toggle "Start camera" and hold a currency note in front of the webcam.
5. The app will display the detected denomination with a confidence score and speak it aloud.
6. Click "Stop" to end the session.

## Results and Outputs
*(Fill in after training — suggested to include)*:
- Sample screenshot(s) of detection with bounding box + confidence score.
- Final training metrics: mAP, precision, recall (printed by `train.py`, also saved under `runs/detect/<run_name>/`).
- Measured inference FPS on your test hardware.
- Link to demo video (optional but recommended per challenge guidelines).

## Future Scope
- Fake/counterfeit note detection.
- Recognition of torn or damaged notes.
- Support for international currencies.
- Native Android application.
- Integration with smart glasses for hands-free assistive use.
- Voice-guided positioning ("move closer", "rotate slightly left", "lighting too low").
- Vibration feedback on successful recognition.
- Low-light image enhancement pre-processing step.

## References
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Universe](https://universe.roboflow.com/) — public datasets for currency/object detection
- [pyttsx3 Documentation](https://pyttsx3.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---
*Built for HackZen 2026 Open Challenge — Computer Vision track.*
