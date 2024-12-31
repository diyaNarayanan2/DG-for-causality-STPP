# DG for Causality in STPP

Domain Generalisation for Causal Learning in Spatio Temporal Point Processes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diyaNarayanan2/DG-for-causality-STPP/blob/main/DG4Causal.ipynb)


Copyright: 2024, Diya Narayanan, [Data Science & it's Applications Lab](https://datasciapps.de/),  [German Research Centre for Artificial Intelligence](https://www.dfki.de/en/web) 

# Overview 

We aim to implement Causal Invariant Learning to the inference process of learning a Hawkes process using Maximum Likelihood estimation. This is done by integrating Domain Generalization algorithms into the learning process during the Expectation Maximization algorithm. The Domain Generalization algorithm incentivizes the model to learn invariant predictors across different environments

# Usage
### Option 1: Using Python

1. **Install Python**: Ensure you have Python installed on your system (version 3.7 or higher).
2. **Install `pip`**: Confirm that `pip` is installed (it usually comes with Python). You can check by running:
   ```bash
   python -m pip --version
   ```
3. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Scripts**:
   - Generate synthetic data:
     ```bash
     python SyntheticDataGen.py
     ```
   - Run the main script:
     ```bash
     python Main.py
     ```

---

### Option 2: Using Jupyter Notebook

1. **Open Jupyter Notebook**:
   - Launch Jupyter Notebook on your local machine and navigate to the project directory.
   - Open the file `DG4Causal.ipynb`.
2. **Run on Google Colab** :
   - Upload the file `DG4Causal.ipynb` to your Google Drive.
   - Open the notebook in Google Colab or click on the _open in colab_ button above.

---

### Option 3: Using a Conda Environment

1. **Install Conda**: Ensure Conda is installed on your system.
2. **Create a Conda Environment**:
   ```bash
   conda create --name myenv python=3.8
   ```
3. **Activate the Environment**:
   ```bash
   conda activate myenv
   ```
4. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the Scripts**:
   - Generate synthetic data:
     ```bash
     python SyntheticDataGen.py
     ```
   - Run the main script:
     ```bash
     python Main.py
     ```
