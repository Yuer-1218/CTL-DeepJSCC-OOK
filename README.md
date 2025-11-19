# CTL-DeepJSCC-OOK: Semantic Communication in UOWC

🌊 **Deep Joint Source-Channel Coding for Underwater Optical Wireless Communication (UOWC) with OOK Modulation**

This repository implements a Deep Learning-based Joint Source-Channel Coding (DeepJSCC) scheme tailored for Underwater Optical Wireless Communication (UOWC). It adapts the classic DeepJSCC architecture to support **On-Off Keying (OOK)** modulation by introducing differentiable binarization and a **Binary Symmetric Channel (BSC)** model.

> **Note:** This project is a fork/modification based on the excellent work [Deep-JSCC-PyTorch](https://github.com/chunbaobao/Deep-JSCC-PyTorch.git) by [chunbaobao](https://github.com/chunbaobao). We extend it to support binary discrete channels suitable for UOWC scenarios.

---

## 🚀 Key Features

* **OOK Modulation Adaptation**: Implements a differentiable binarization layer using **Sigmoid + Straight-Through Estimator (STE)** to enable end-to-end training with discrete 0/1 outputs.
* **BSC Channel Model**: Replaces the standard AWGN/Rayleigh channels with a **Binary Symmetric Channel (BSC)** to simulate bit-flip errors caused by underwater turbulence and turbidity.
* **End-to-End Optimization**: The encoder and decoder are jointly optimized to minimize image reconstruction error (MSE/PSNR) under specific Bit Error Rates (BER).

## 🛠️ Architecture

The system consists of three main components:

1.  **Encoder**: Extracts semantic features and maps them to binary symbols (0s and 1s) for OOK transmission.
2.  **Channel (BSC)**: Simulates the underwater channel by flipping bits with a probability $p$ (BER).
3.  **Decoder**: Reconstructs the original image from the noisy binary sequence.

$$\text{Image} \rightarrow \text{Encoder} \rightarrow \{0, 1\}^k \xrightarrow{\text{BSC (BER)}} \text{Noisy Bits} \rightarrow \text{Decoder} \rightarrow \text{Reconstruction}$$

## 📂 Project Structure

```text
.
├── model.py      # DeepJSCC model definition with OOK/STE support
├── channel.py    # Channel implementations (Added BSC support)
├── train.py      # Training pipeline
├── dataset.py    # Dataset loaders
├── utils.py      # Utility functions
└── README.md
