# 🚀 End-to-End Audio Classification + Visualizer – Fully Built & Deployed 🎧🧠

Inspired by a fantastic tutorial from [Andreas Trolle on YouTube](https://www.youtube.com/@anditrolle), I built a full-stack audio classification system from scratch — and I’m excited to share the final result. It’s fully functional, GPU-accelerated, and visually interactive, powered by **PyTorch**, **Modal**, **FastAPI**, and **Next.js**.

---

## 🔍 What it Does

- Upload any `.wav` audio file (e.g., baby crying, siren, rain)
- Transforms it into a **Mel Spectrogram**
- Passes it through a **CNN with residual connections**
- Outputs **top-3 predictions** with confidence scores
- Visualizes the **waveform**, **spectrogram**, and **CNN feature maps**

---

## 👨‍💻 Frontend Highlights (Next.js + Tailwind)

- Drag-and-drop audio upload with smooth loading states
- Dynamic waveform + spectrogram rendering
- Scrollable feature map viewer for **every CNN layer**
- Clean, minimal design using badges, progress bars, and color scales

---

## ⚙️ Backend & Infrastructure

- CNN model trained on the **ESC-50** environmental sound dataset
- Training includes:
  - Mixup augmentation
  - Label smoothing
  - OneCycleLR scheduler
- Inference served through **FastAPI** inside a **Modal GPU (A10G)** container
- Audio sent as **base64-encoded WAV** — no S3 or streaming needed

---

## 📈 Why This Matters

- Demonstrates how to combine **deep learning + MLOps + frontend** into one cohesive app
- Makes CNN behavior **interpretable** through visual layers
- Fast, portable, and extensible with more data or improved models

---

## 💡 What I Learned

- 🔹 Audio ML: feature engineering, training, evaluation
- 🔹 MLOps: scalable GPU inference deployment with Modal
- 🔹 Full-stack engineering: API design, UI, and data visualization

---

## 🙏 Credits

**Huge thanks to [Andreas Trolle](https://www.youtube.com/@anditrolle)** — his YouTube series laid the groundwork. I built on top of it by adding:

- A custom training pipeline
- An interactive frontend
- Deployment using FastAPI + Modal

---

## 👀 Coming Soon

- Live demo
- GitHub source code

Let me know if you'd like to **test it**, contribute, or dive into the code!

---

### 🔗 Hashtags

`#DeepLearning` `#NextJS` `#PyTorch` `#ModalLabs` `#MLOps` `#MachineLearning`  
`#ESC50` `#AI` `#AudioProcessing` `#FastAPI` `#FeatureMaps` `#CNN` `#FullStackAI` `#React` `#AndreasTrolle`
