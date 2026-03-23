# Transformer-From-Scratch-PyTorch

A rigorous implementation of the Transformer architecture from the ground up, built exclusively using fundamental tensor operations in PyTorch without relying on high-level APIs like `nn.Transformer`. 

## 🛠 Core Implementation Logic
To ensure a deep understanding of the mathematical foundations of Large Language Models (LLMs), this project eschews "syntax sugar" and high-level abstractions in favor of manual implementation: 

* **Multi-Head Attention (MHA)**: Independently implemented the matrix-based attention scoring mechanism ($QK^T$) and multi-head splitting/concatenation logic. [cite: 42]
* **Sinusoidal Positional Encoding**: Manually derived and implemented the encoding formula to inject sequence order: 
    $$PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_{model}})$$
    $$PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d_{model}})$$
* **Layer Normalization**: Replicated the bottom-up calculation of mean and variance to ensure training stability and deep network convergence. 
* **Masking Infrastructure**: Designed custom Padding Masks and Causal (Look-ahead) Masks to maintain logical isolation during the auto-regressive decoding process. 

## 📊 Experimental Results
The architecture's validity was verified on a small-scale machine translation dataset with the following metrics after 25 epochs: 

| Metric | Initial Value | Final Value | Change |
| :--- | :--- | :--- | :--- |
| **Loss** | 4.06 | 2.00 | -50.7%  |
| **BLEU Score** | - | 59.54 | Verified  |

## 🚀 Key Takeaway
This project demonstrates the ability to decompose complex neural architectures into basic linear algebra operations, ensuring full control over the model's forward and backward passes.