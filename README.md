# BAT 3305 - Colazo: Support Vector Machines Kernel Lab

An interactive Streamlit teaching demo for undergraduate machine-learning instruction.

## What students can explore

- Linear, polynomial, and RBF/Gaussian SVM kernels
- Four synthetic datasets: linear clusters, two moons, concentric circles, and XOR
- Hyperparameters: `C`, `gamma`, polynomial degree, and `coef0`
- Noise level, sample size, train/test split, and random seed
- Decision boundary and the SVM ±1 margin levels
- Support vectors highlighted directly on the plot
- Held-out accuracy, precision, recall, and F1
- Confusion matrix and support-vector share
- Kernel-intuition plots for RBF gamma and polynomial degree
- Five guided classroom experiments and exit questions

## Files

- `app.py` — Streamlit interface
- `svm_utils.py` — dataset, model, metrics, and kernel helper functions
- `requirements.txt` — deployment dependencies
- `.streamlit/config.toml` — minimal Streamlit configuration

## Run locally

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the contents of this folder to the repository root.
3. In Streamlit Community Cloud, choose **Create app**.
4. Select the repository and branch.
5. Set the main file path to `app.py`.
6. Deploy.

No secrets, API keys, databases, or external files are required.

## Suggested classroom sequence

1. Start with **Concentric circles + Linear kernel** and establish that C cannot fix the wrong boundary family.
2. Switch to **RBF** and vary gamma from low to high.
3. Hold gamma fixed and vary C to separate the concepts of locality and error penalty.
4. Use **XOR + Polynomial** to explore feature interactions and degree.
5. Add noise and compare model complexity with held-out performance.

## Instructor note

The app standardizes the two predictor variables before fitting the SVM. This is intentional: SVMs are sensitive to feature scale, so the demo models a sound default workflow rather than fitting directly to unscaled features.
