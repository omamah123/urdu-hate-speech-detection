# Urdu Hate Speech Detection

Implementation of multiple machine learning and deep learning approaches for detecting hate speech in Urdu text.

> **Paper:** Hussain et al. (2025) *"Enhancing Urdu hate speech detection through differential transfer learning and adaptive loss functions"*, Scientific Reports.  
> DOI: https://doi.org/10.1038/s41598-025-21306-w

---

## Dataset

**NUHONS (Urdu Combined)** — `Urdu_combined.csv`

| Label | Count |
|-------|-------|
| Normal | 7,891 |
| Offensive | 6,199 |
| Hate | 2,247 |
| **Total** | **~16,337** (after cleaning) |

The dataset contains Urdu social media comments labeled as `normal`, `offensive`, or `hate`.

---

## Models Implemented

| Category | Models |
|----------|--------|
| Classical ML | Logistic Regression, SVM, Random Forest |
| Naive Bayes | Multinomial NB, Complement NB, Bernoulli NB |
| Deep Learning | RNN, LSTM (with Attention) |
| Transformer | XLM-RoBERTa |

---

## Project Structure

```
├── Urdu_Hate_Speech_Detection.ipynb   # Main notebook
├── Urdu_combined.csv                  # Dataset
└── README.md
```

---

## Pipeline

The notebook is organized into phases:

- **Phase 0** — Environment setup & imports
- **Phase 1** — Data loading & exploration
- **Phase 2** — Urdu text preprocessing pipeline
- **Phase 3** — Train/Val/Test split & TF-IDF features
- **Phase 4** — Classical ML & Naive Bayes classifiers
- **Phase 5** — Deep Learning: RNN & LSTM
- **Phase 6** — Transformer: XLM-RoBERTa

All models are evaluated with Accuracy, Precision, Recall, and F1-score (weighted).

---

## Requirements

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib torch transformers datasets
```

- Python 3.8+
- GPU recommended for XLM-RoBERTa training (CUDA supported)

---

## Usage

1. Clone the repository:
   ```bash
 git clone https://github.com/omamah123/urdu-hate-speech-detection.git
 cd urdu-hate-speech-detection
   ```

2. Install dependencies:
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn joblib torch transformers datasets
   ```

3. Open the notebook:
   ```bash
   jupyter notebook Urdu_Hate_Speech_Detection.ipynb
   ```

4. Run all cells in order (Phase 0 → Phase 6).

---

## Citation


```
Hussain et al. (2025). Enhancing Urdu hate speech detection through differential 
transfer learning and adaptive loss functions. Scientific Reports. 
https://doi.org/10.1038/s41598-025-21306-w
```
