# 🤟 Real-Time Sign Language Detection using Transformer + MediaPipe

This project implements a real-time sign language gesture classification system using **MediaPipe Holistic** for keypoint extraction and a custom-built **Transformer model (PyTorch)** for sequence classification. It runs directly on your webcam and provides live predictions with a smooth and responsive UI.

---

## 🔍 Overview

- **Keypoint Extraction**: Uses [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html) to extract 3D landmarks of face, pose, and both hands.
- **Sequence Modeling**: A Transformer model learns temporal dynamics over a sequence of 30 frames to classify the gesture.
- **Live Prediction**: Real-time classification with probability bars, sentence construction, and action stability using a debouncing technique.

---

## 📸 How It Works

1. **Capture Webcam Feed**
2. **Extract 3D Landmarks** using MediaPipe Holistic
3. **Buffer 30-Frame Sequences**
4. **Infer with Transformer Model**
5. **Visualize Gesture Probabilities & Sentence**

---

## 🧠 Transformer Model Architecture

The model is based on the **Transformer Encoder** architecture adapted for gesture classification.

### 🔧 Architecture Details:

- **Input**: Shape `(Batch, 30, 1662)` → 30 frames of keypoints (each frame has 1662 features)
- **Positional Encoding**: Injects temporal information
- **Transformer Encoder Layers** (×4):
  - Multi-Head Self Attention
  - Layer Normalization
  - Feed-Forward Network
  - Residual Connections
- **Global Average Pooling**: Compresses time dimension
- **Fully Connected Layer** → Softmax

### 📚 Theory

- **Self-Attention** captures the relationship between keypoints across frames.
- **Multi-head Attention** allows learning from multiple temporal contexts.
- **Positional Encoding** ensures the model knows the order of frames.
- **Transformer Encoders** handle long-range dependencies efficiently, making them ideal for tracking dynamic hand gestures and body motions.

---

## ✋ Recognized Gestures

The current model is trained on the following signs:

| Label      | Description                        |
|------------|------------------------------------|
| `hello`    | Waving gesture                     |
| `thanks`   | Hand from chin outward             |
| `yes`      | Fist nodding motion                |
| `no`       | Hand side-to-side                  |
| `iloveyou` | Classic one-handed "I ❤️ U" sign   |
| `please`   | Hand circling on chest             |
| `stop`     | Open palm gesture                  |
| `help`     | One hand lifting the other         |
| `goodbye`  | Wave with open hand                |
| `okay`     | Thumb and index forming an "O"     |

> ⚠️ You can easily extend this by collecting new gesture data.

---

## 🧪 Training Pipeline

1. **Collect Data**: Capture sequences using MediaPipe and store `.npy` files.
2. **Preprocess**: Shape data into `(N, 30, 1662)` arrays.
3. **Train Model**:
   ```bash
   python train.py
   ```
4. **Save Model**: Save and load `.pt` weights for inference.

---

## ▶️ Run Real-Time Inference

Make sure you have a trained model saved and MediaPipe installed.

```bash
python real_time_inference.py
```

Press `q` to quit.

---

## 📦 Installation

### ✅ Requirements

- Python 3.8+
- PyTorch
- OpenCV
- MediaPipe
- NumPy, SciPy

### 🔧 Setup

```bash
git clone https://github.com/yourusername/sign-language-transformer.git
cd sign-language-transformer
pip install -r requirements.txt
```

---

## 🗂️ Project Structure

```
📂 sign-language-transformer/
├── real_time_inference.py     # Webcam inference code
├── train.py                   # Model training script
├── model.py                   # Transformer model
├── utils.py                   # MediaPipe + keypoint helpers
├── data/                      # Stored keypoint sequences
├── checkpoints/               # Saved model weights
└── requirements.txt
```

---

## 🧠 Sample Key Function

```python
def prob_viz(res, actions, input_frame, colors):
    for num, prob in enumerate(res):
        cv2.rectangle(input_frame, (0, 60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(input_frame, actions[num], (0, 85+num*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return input_frame
```

---

## 📖 References

- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic)
- [Attention Is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
- PyTorch Documentation

---

## 📄 License

This project is open-source under the MIT License.

---

## 🙌 Contributions

Feel free to fork the repo, improve the model, add more gestures, or build a GUI! PRs are welcome.