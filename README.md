# Cut2Self-DDM: A Self-Supervised Image Denoiser

**Cut2Self-DDM** is an advanced self-supervised image denoising framework utilizing data-dependent masking (DDM) techniques. Unlike traditional methods employing fixed dropout rates, Cut2Self-DDM adapts dropout probabilities based on the statistical properties of the input images, significantly enhancing the denoising effectiveness and reconstruction quality.

---

## Key Highlights

- **Dynamic Dropout**: Random masking based on input image statistics, enhancing stability and quality.
- **Self-Supervised**: No paired noisy-clean datasets required.
- **Partial Convolutions**: Effectively handles masked or corrupted image regions.
- **Optimized Training**: Improved convergence through adaptive loss computations and training techniques.

---

## Technologies & Libraries

- **Python 3.8+**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **OpenCV**
- **scikit-image**
- **SciPy**
- **Matplotlib**

---

## 📦 Installation & Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/Cut2Self-DDM.git
cd Cut2Self-DDM
```
### Step 2: Create Python Environment (Recommended)

It's recommended to use a virtual environment (such as conda) to manage dependencies:

```bash
conda create -n cut2self python=3.8
conda activate cut2self
```

> **Note:** If you prefer not to use conda, you can skip directly to **Step 3**.


### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```
