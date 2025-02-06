# **Text-to-Image Generator Using StackGAN**

This project integrates **Stacked Generative Adversarial Networks (StackGAN)** with **BERT embeddings** to generate high-resolution images from textual descriptions. The model is trained using the **CUB-200-2011 bird dataset**, allowing for the creation of **photo-realistic bird images** based on user-provided text input.

The system features a **two-stage GAN architecture**, leveraging **BERT embeddings** to better capture the nuanced meaning of input text, ensuring more accurate image generation.

---

## 📌 **Features**
- ✔️ **Text-to-Image Generation** – Generate images from textual descriptions.  
- ✔️ **BERT Embeddings** – Utilize **BERT** for a richer text representation.  
- ✔️ **Two-Stage GAN** – High-resolution image refinement through a **two-stage process**.  

---

## 🚀 **Installation & Setup**

### **1️⃣ Clone the Repository**
  git clone https://github.com/your-username/stackgan-text-to-image.git
  cd stackgan-text-to-image
### **2️⃣ Set Up a Virtual Environment (Optional but Recommended)**
  python -m venv venv
  source venv/bin/activate  # On Linux/Mac
  venv\Scripts\activate  # On Windows
### **3️⃣ Install Dependencies**
  pip install -r requirements.txt
### **4️⃣ Download the Dataset**
  Download the CUB-200-2011 dataset from this link and place it inside the data/ directory.

### **5️⃣ Generate BERT Embeddings**
  Run the script to preprocess textual descriptions and generate embeddings using BERT:
  python generate_bert_embeddings.py

### **6️⃣ Train the Model**
  To train the StackGAN model, run:
  python train.py --epochs 100 --batch_size 64

### **🎨 Usage **
### **✅ Generating Images from Text Input **
  Once the model is trained, you can generate images using:
  python generate.py --text "A yellow bird with black wings and a short beak."
### **✅ Using the Web Interface **
  To use a web-based UI for generating images:
  streamlit run app.py
  Then, open http://localhost:8501/ in your browser.

### **🛠 Software Requirements **
  📌 Programming Language: Python 3.x
  📌 Deep Learning Framework: TensorFlow / PyTorch
  📌 Libraries:
NLP: Hugging Face Transformers, spaCy, NLTK
Image Processing: OpenCV, PIL
GAN Implementation: Torch, TensorFlow-GAN
📌 GPU Acceleration: CUDA, cuDNN for faster training

### **💻 Hardware Requirements **
⚡ GPU: NVIDIA GeForce RTX series / NVIDIA A100
📌 Memory (RAM): 16GB – 32GB recommended
📌 Storage: SSD recommended for faster performance

### **📂 Project Structure**
stackgan-text-to-image/
│── data/                     # Dataset storage
│── models/                   # Saved model checkpoints
│── scripts/                  # Helper scripts for processing
│── train.py                   # Training script
│── generate.py                # Image generation script
│── generate_bert_embeddings.py # BERT embedding generator
│── app.py                     # Web-based UI for text input
│── README.md                  # Project documentation
│── requirements.txt            # Required dependencies

### **🤝 Contributing**
Contributions are welcome! Feel free to open an issue or submit a pull request.
