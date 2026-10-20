<div align="center">

# 🧠 ADV — Advanced Programming for Data Science

### TP02 · Object-Oriented Data Pipelines in Python

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Git](https://img.shields.io/badge/Git-Version%20Control-F05032?style=for-the-badge&logo=git&logoColor=white)](https://git-scm.com/)
[![License](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)](#)

*A clean, reproducible implementation of data-processing pipelines using Object-Oriented Programming principles.*

[📌 Overview](#-project-overview) ·
[🗂️ Structure](#️-project-structure) ·
[📊 Dataset](#-dataset--data-pipeline) ·
[📓 Notebook](#-notebook-breakdown-tp2ipynb) ·
[⚙️ Setup](#️-setup--installation) ·
[▶️ Usage](#️-how-to-run) ·
[📈 Results](#-results--outputs)

</div>

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Project Structure](#️-project-structure)
3. [Dataset & Data Pipeline](#-dataset--data-pipeline)
4. [Notebook Breakdown (`TP2.ipynb`)](#-notebook-breakdown-tp2ipynb)
5. [Setup & Installation](#️-setup--installation)
6. [How to Run](#️-how-to-run)
7. [Results & Outputs](#-results--outputs)
8. [Tech Stack](#-tech-stack)
9. [Author & License](#-author--license)

---

## 📌 Project Overview

**ADV (Advanced Programming for Data Science)** is a hands-on course repository. This assignment, **TP02**, applies **Object-Oriented Programming (OOP)** in Python to build modular, reusable, and testable data-science workflows.

Instead of relying on one-off scripts, the project encapsulates data loading, cleaning, and transformation logic inside well-structured classes. This mirrors how production-grade data pipelines are designed.

### 🎯 Objectives

| # | Objective | Description |
|:-:|-----------|-------------|
| 1 | 🧱 **Apply OOP** | Design classes with encapsulation, clear responsibilities, and reusable methods |
| 2 | 🔄 **Build a pipeline** | Load raw data → clean → transform → export a processed dataset |
| 3 | 📊 **Analyze data** | Explore, summarize, and visualize the dataset with Pandas & NumPy |
| 4 | ♻️ **Ensure reproducibility** | Provide clear setup steps so results can be re-generated anywhere |
| 5 | 🧪 **Document outputs** | Save results to the `output/` directory for review |

> [!NOTE]
> This repository is part of an academic course. Assignment briefs and PDFs are intentionally excluded from version control via `.gitignore`.

---

## 🗂️ Project Structure

```text
ADV/
│
├── 📁 dataset/
│   ├── 📄 sample_data.csv               # Raw input dataset
│   └── 📄 transformed_sample_data.csv   # Processed dataset (pipeline output)
│
├── 📁 TP/
│   ├── 📓 TP2.ipynb                     # Main analysis & OOP implementation
│   └── 📁 output/                       # Generated results (tables, figures, logs)
│
├── 🚫 .gitignore                        # Excludes Docs/ (PDFs & assignment briefs)
└── 📘 README.md                         # You are here
```

> [!TIP]
> The `Docs/` folder (assignment briefs, PDFs) exists locally only. It is listed in `.gitignore`, so it will not appear in the remote repository.

---

## 📊 Dataset & Data Pipeline

### 🔁 Pipeline at a Glance

```text
┌──────────────────┐     ┌────────────────────┐     ┌───────────────────────────┐     ┌──────────────┐
│ sample_data.csv  │ ──▶ │  Load & Inspect    │ ──▶ │  Clean & Transform (OOP)  │ ──▶ │ transformed_ │
│   (raw input)    │     │  (Pandas)          │     │  (custom classes)         │     │ sample_data  │
└──────────────────┘     └────────────────────┘     └───────────────────────────┘     └──────────────┘
```

### 📦 Files Overview

| File | Location | Role | Stage |
|------|----------|------|:-----:|
| `sample_data.csv` | `dataset/` | Raw source data | 🟡 Input |
| `transformed_sample_data.csv` | `dataset/` | Cleaned / feature-engineered data | 🟢 Output |

<details>
<summary><b>📋 Dataset Summary (click to expand)</b></summary>

<br>

| Property | Details |
|----------|---------|
| **Format** | CSV (comma-separated values) |
| **Raw file** | `dataset/sample_data.csv` |
| **Processed file** | `dataset/transformed_sample_data.csv` |
| **Rows × Columns** | `<N> × <M>` *(update after running the notebook)* |
| **Missing values** | `<describe handling strategy>` |
| **Target variable** | `<if applicable>` |

**Feature overview** *(edit to match your data)*:

| Column | Type | Description |
|--------|:----:|-------------|
| `<column_1>` | `int` | `<description>` |
| `<column_2>` | `float` | `<description>` |
| `<column_3>` | `str` | `<description>` |
| `<column_4>` | `category` | `<description>` |

**Typical transformations applied:**

- ✅ Missing-value handling (imputation or removal)
- ✅ Data-type casting and column standardization
- ✅ Encoding of categorical variables
- ✅ Scaling / normalization of numerical features
- ✅ Export of the final dataset to `transformed_sample_data.csv`

</details>

---

## 📓 Notebook Breakdown (`TP2.ipynb`)

The main notebook is located at `TP/TP2.ipynb` and is organized into logical, sequential sections.

| Section | Title | What it does |
|:-------:|-------|--------------|
| **1** | 📥 Imports & Configuration | Loads libraries and defines paths |
| **2** | 🧱 OOP Class Design | Defines classes for loading, cleaning, and transforming data |
| **3** | 🔍 Exploratory Data Analysis | Summary statistics, distributions, and missing-value checks |
| **4** | 🔄 Data Transformation | Applies class methods to build the processed dataset |
| **5** | 💾 Export | Writes `transformed_sample_data.csv` and saves artifacts to `TP/output/` |
| **6** | 📈 Results & Discussion | Summarizes findings and observations |

<details>
<summary><b>🧩 OOP Concepts Demonstrated (click to expand)</b></summary>

<br>

| Concept | Where it appears |
|---------|------------------|
| **Encapsulation** | Data and processing logic bundled inside classes |
| **Abstraction** | Simple public methods hide cleaning/transform details |
| **Reusability** | The same class can process any CSV with a similar schema |
| **Modularity** | Each class has a single, clear responsibility |
| **Method chaining / pipelines** | Sequential transformations applied in a defined order |

**Illustrative skeleton:**

```python
import pandas as pd

class DataProcessor:
    """Loads, cleans, and transforms a tabular dataset."""

    def __init__(self, path: str):
        self.path = path
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.path)
        return self

    def clean(self):
        self.df = self.df.drop_duplicates().dropna()
        return self

    def transform(self):
        # feature engineering, encoding, scaling...
        return self

    def save(self, out_path: str):
        self.df.to_csv(out_path, index=False)
```

</details>

---

## ⚙️ Setup & Installation

> [!IMPORTANT]
> **Prerequisites:** Python **3.10+**, **Git**, and **pip** must be installed on your machine.

<details>
<summary><b>🛠️ Step-by-step installation (click to expand)</b></summary>

<br>

**1️⃣ Clone the repository**

```bash
git clone https://github.com/<your-username>/ADV.git
cd ADV
```

**2️⃣ Create a virtual environment**

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**3️⃣ Install dependencies**

```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

**4️⃣ Verify the installation**

```bash
python -c "import pandas, numpy, sklearn; print('✅ Environment ready')"
```

</details>

---

## ▶️ How to Run

<details>
<summary><b>🚀 Execution instructions (click to expand)</b></summary>

<br>

**1️⃣ Launch Jupyter**

```bash
jupyter notebook
```

**2️⃣ Open the notebook**

Navigate to `TP/TP2.ipynb`.

**3️⃣ Run all cells**

From the menu: **Kernel → Restart & Run All**.

**4️⃣ Check the outputs**

- Processed dataset: `dataset/transformed_sample_data.csv`
- Generated artifacts: `TP/output/`

</details>

> [!TIP]
> Run the notebook from the `TP/` directory (or use relative paths such as `../dataset/sample_data.csv`) so file paths resolve correctly.

> [!WARNING]
> Running the notebook will **overwrite** `transformed_sample_data.csv` and any files in `TP/output/` with the same name.

---

## 📈 Results & Outputs

All generated artifacts are stored in `TP/output/`.

| Output | Location | Description |
|--------|----------|-------------|
| 📄 Transformed dataset | `dataset/transformed_sample_data.csv` | Final cleaned and engineered data |
| 📊 Figures / plots | `TP/output/` | Visualizations from the analysis |
| 🧾 Logs / summaries | `TP/output/` | Execution logs and statistics |

### ✨ Key Highlights

- 🧱 A modular, class-based pipeline replaces ad-hoc scripting
- 🔄 End-to-end flow from raw CSV to analysis-ready dataset
- ♻️ Fully reproducible with the setup steps above
- 📊 Findings documented directly in the notebook

<details>
<summary><b>🗒️ Output Logs & Detailed Results (click to expand)</b></summary>

<br>

```text
[INFO] Loading dataset ........... OK
[INFO] Cleaning data ............. OK
[INFO] Applying transformations .. OK
[INFO] Saving output ............. OK
```

> Replace this block with real console output or key metrics from your run.

</details>

---

## 🧰 Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Environment** | Jupyter Notebook |
| **Data** | Pandas, NumPy |
| **Machine Learning** | scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Version Control** | Git & GitHub |

---

## 👤 Author & License

**Author:** `<Your Name>` · [GitHub](https://github.com/<your-username>) · [LinkedIn](https://linkedin.com/in/<your-handle>)

**Course:** Advanced Programming for Data Science (ADV)

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

⭐ *If you found this repository helpful, consider giving it a star!* ⭐

</div>