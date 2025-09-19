# Pattern Analysis – Lab Demonstration 2

**Course:** Pattern Recognition (COMP3710)
**Instructor:** Shekhar Chandra

This lab explored **dimensionality reduction, classification, and deep learning pipelines** using **NumPy, TensorFlow, and PyTorch**. The tasks were divided into **four parts**:

---

## 📌 Part 1 – Discrete Fourier Transform (DFT)

* Implemented and visualized **square waves** and their **Fourier series approximation**.
* Compared **naïve DFT (O(N²))** vs. **FFT (O(N log N))** implementations.
* Re-implemented `square_wave`, `square_wave_fourier`, and `naive_dft` using **PyTorch/TensorFlow** (with GPU acceleration).
* Compared runtime for different data sizes and explained why **FFT is the fastest**.

---

## 📌 Part 2 – Eigenfaces (PCA on Faces)

* Used the **LFW dataset** (Labeled Faces in the Wild).
* Performed **PCA (SVD-based)** to compute **eigenfaces** for dimensionality reduction.
* Visualized eigenfaces and compactness plots (variance explained by components).
* Built a **Random Forest classifier** on PCA-transformed features.
* Re-implemented PCA using **PyTorch/TensorFlow operations**.

---

## 📌 Part 3 – CNN Classifier

* Implemented a simple **CNN** with:

  * Two Conv layers (3×3, 32 filters each)
  * Dense layers for classification
* Used **Adam optimizer** + **categorical cross-entropy loss**.
* Trained and evaluated the CNN on the **LFW dataset**, improving over the PCA+RF baseline.

---

## 📌 **Part 3.3 – DAWNBench Challenge (CIFAR-10)**

Implemented ResNet-18 in TensorFlow/Keras with:

Mixed precision training (FP16 on GPUs)

RandAugment data augmentation

Warmup + Cosine learning rate schedule

AdamW optimizer with EMA (Exponential Moving Average)

Achieved ~94% test accuracy on CIFAR-10 meeting DAWNBench standards.

---

## 📌 Part 4.1 – Recognition (Custom Project)

* Implemented a **Variational Autoencoder (VAE)** on **OASIS brain MRI slices**.
* Encoder–Decoder architecture with latent space sampling (`μ`, `logσ²`, reparameterization trick).
* Trained with **MSE + KL Divergence loss**.
* Visualized the learned **latent manifold** using **UMAP**.
* Generated new MRI-like samples by sampling from latent space.

---

## 📌4.2 MRI Segmentation with U-Net

* Built a U-Net architecture for pixel-level brain MRI segmentation.

* Encoder (downsampling) path captured hierarchical features.

* Decoder (upsampling) path with skip connections restored fine spatial details.

* Trained using a combination of Dice loss + Weighted Cross-Entropy to handle class imbalance.

* Achieved ~0.73 mean Dice score across validation data.

* Demonstrated inference on unseen MRI slices, overlaying predicted masks on input images.

## ✅ Outcomes

* Understood how **Fourier, PCA, CNNs, and VAEs** connect as fundamental building blocks in AI.
* Practiced **NumPy → PyTorch/TF re-implementations**.
* Compared traditional ML (PCA + RF) with deep learning approaches (CNNs, VAEs).
* Prepared ground for larger recognition projects in the next assessment.
