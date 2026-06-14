# Flight Price Prediction: A Framework Comparison (PyTorch vs. TensorFlow)

An end-to-end Deep Learning regression project that predicts airline ticket prices by processing large-scale structured data logs. This repository provides a direct architectural comparison by building, training, and evaluating equivalent Artificial Neural Network (ANN) regression models across both major deep learning frameworks: **PyTorch** and **TensorFlow / Keras**.

## 🚀 Project Overview
Dynamic ticket pricing is highly dependent on multi-class parameters such as seat tiers, route nodes, and carrier scheduling. This project handles price prediction as a pure regression challenge. It features a complete pipeline that integrates data cleansing, multi-collinearity checks via Variance Inflation Factor (VIF), and standard scaling transformations before routing features through parallel deep learning workflows.

## 📂 Repository Structure
* **`python---PyTorch__TensorFlow.ipynb`**: The master Jupyter Notebook containing data integration pipelines, VIF-driven feature filtering, and individual comparative neural network workflows for both PyTorch and TensorFlow.

## 📊 Dataset & Pipeline Breakdown
The workflow processes massive structured flight logs combining over 300,000 baseline records from separate commercial tiers (`business.csv` and `economy.csv`).

### 1. Data Cleaning & Engineering
* **Temporal Parsing**: Converts raw object features into standard datetime sequences (`date`, `dep_time`).
* **Duration Parsing**: Extracts hours and minutes using regular expressions (`regex`) to map flight duration segments into standard quantitative intervals (Total Minutes).
* **Target Isolation**: Strips native currency tags and formatting markers from price objects to map them into true integer regression values.

### 2. Feature Pruning via VIF Matrix
* Categorical parameters are tracked using dummy matrix extensions (`pd.get_dummies`).
* To prevent regression inflation faults, a Variance Inflation Factor (VIF) matrix is generated to audit multicollinearity across all 66 encoded features.
* Low-variance parameters (such as `class_economy`, selective source/destination hubs, and structural identifiers) are isolated to feed the deep learning networks.

### 3. Dual Framework Implementations
The network structures match across both libraries to maintain comparison validity:
* **Hidden Layers**: Input mapped to a 50-neuron dense block, stepping down into a 30-neuron hidden layer, using **ReLU** activations.
* **Output Node**: A single target layer mapping regression price predictions.
* **Optimization & Loss**: Regulated using the **Adam** optimizer and calculated using Mean Squared Error (MSE).
  * **TensorFlow**: Implemented cleanly using Keras `Sequential` API modules.
  * **PyTorch**: Subclassed via `nn.Module`, wrapping features explicitly into Torch tensors (`torch.float32`) for manual training loop execution.

## 🛠️ Tech Stack & Requirements
* **Core Language:** Python
* **Deep Learning Libraries:** PyTorch (`torch`, `torch.nn`), TensorFlow / Keras
* **Statistical Diagnostics:** Statsmodels (Variance Inflation Factor)
* **Data Wrangling:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME