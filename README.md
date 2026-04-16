

# CoRe-SCD: A Boundary-Task Co-Refinement Network for Semantic Change Detection


## 1. Experimental Environment

### Hardware
* **GPU**: Single NVIDIA GeForce RTX 4060 
* **Workstation**: Deep Learning Workstation

### Software Stack
* **Framework**: PyTorch
* **Data Processing**: GDAL, OpenCV, NumPy
* **Environment**: Python 3.8+, CUDA 11.0+

---

## 2. Training Hyperparameters

To ensure the reproducibility of the results reported in the paper, all components of the CoRe-SCD framework were trained using the following standardized configurations:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Optimizer** | Adam | Adaptive Moment Estimation |
| **Initial Learning Rate** | 0.1 | With decay scheduling |
| **Batch Size** | 8 |  |
| **Training Epochs** | 80 |  |
| **Loss Weights** | (0.5, 1.0, 1.0, 1.0) |For $\mathcal{L}_{seg}, \mathcal{L}_{change}, \mathcal{L}_{sc}, \mathcal{L}_{bd}$ respectively |
| **Data Augmentation** | Flip & Rotation | Random horizontal flipping and 90° rotation only |

> **Note on Loss Weighting**: The weights (0.5, 1.0, 1.0, 1.0) were determined to be the optimal synergistic equilibrium through our sensitivity analysis (Table IV in the manuscript) to balance the gradient flow between semantic and change detection tasks.

---

## 3. Evaluation & Reproducibility Protocol

### Fairness of Comparison
To ensure absolute fairness in horizontal comparisons, all comparative SOTA models (e.g., SCanNet, ChangeMamba, EGMS-Net, etc.) were re-trained from scratch in the identical hardware environment and with the same hyperparameter settings described above.

### Statistical Rigor
The performance of all models was evaluated using six primary metrics: **OA**, **SeK**, **mIoU**, **Prec**,**Rec**and **F1**. To minimize the impact of random initialization errors and ensure the reliability of the evaluation, the reported results are the **average of three independent experiments** conducted on the test set.

---

## 4. Usage
(The full source code, including model definitions and training scripts, will be released upon formal acceptance of the manuscript.)

---

## 5. Weights
(The trained model weights of our CoRe-SCD can be downloaded from the following link:
https://drive.google.com/file/d/1LLK8LytVlD6lq6ea8dRi_JUhxhyOGHBV/view?usp=drive_link)

---


The code is mainly developed based on the public resources shared by Professor Lei Ding (https://github.com/DingLei14/SCanNet), which provided important references for this work, and we sincerely thank him for his generous support.




