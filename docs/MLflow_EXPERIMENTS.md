# MLflow Experiment Tracking Report

## 1. Overview and Objective

The **Resume–Job Description Matcher** uses Machine Learning (TF-IDF vectorization and cosine similarity) alongside a boundary-aware technical skill extraction engine to measure the textual and technical alignment between candidate resumes and job postings.

To evaluate model configurations objectively, we track hyperparameter variants and performance metrics using **MLflow** under the experiment:
```text
Resume-Job-Description-Matcher
```

---

## 2. Evaluation Dataset (`evaluation_pairs.csv`)

### Dataset Characteristics
- **Total Samples:** 12 curated resume–job description pairs.
- **Data Source:** Synthetic evaluation dataset (`dataset_version: synthetic_v1.0`).
- **Ground Truth Target (`expected_score`):** Expert reference alignment score scaled from 0 to 100.
- **Coverage Categories:**
  1. **High Match:** Direct stack alignment (e.g., Senior Python/FastAPI backend, Machine Learning Engineer, React Frontend). Expected scores: 85–90.
  2. **Medium-High Match:** Broad backend alignment with minor stack gaps (Fullstack, QA automation). Expected scores: 70–80.
  3. **Medium Match:** Overlapping foundation but missing critical tools (e.g., Data Analyst missing Power BI). Expected scores: 55–65.
  4. **Low Match:** Stack divergence (e.g., Java Enterprise to React Frontend, iOS Swift to ML). Expected scores: 15–25.
  5. **Edge Cases:** Non-technical background applying to full-stack engineering (score ~8) and job postings with generic language containing zero recognized technical keywords (score ~30).

> **Important Disclosure:** This evaluation dataset is synthetic and designed strictly to benchmark NLP text similarity and skill coverage calculations. The scores measure textual and technical keyword alignment, not actual hiring decisions, candidate competency, or job performance.

---

## 3. Experiment Matrix

We executed three controlled experiments with a fixed random seed (`seed = 42`) for complete reproducibility:

| Experiment | TF-IDF N-gram Range | Skill Weight ($\lambda$) | TF-IDF Config | Dataset Identifier |
| :--- | :---: | :---: | :--- | :--- |
| **Experiment A** | `(1, 1)` (Unigrams) | `0.4` | `lowercase=True, stop_words=english` | `synthetic_v1.0` |
| **Experiment B** | `(1, 2)` (Unigrams + Bigrams) | `0.4` | `lowercase=True, stop_words=english` | `synthetic_v1.0` |
| **Experiment C** | `(1, 2)` (Unigrams + Bigrams) | `0.5` | `lowercase=True, stop_words=english` | `synthetic_v1.0` |

### Scoring Formula
$$\text{Final Score} = (1 - \lambda) \times \text{Text Similarity} + \lambda \times \text{Skill Coverage}$$
where $\lambda$ represents the `skill_weight` (between 0.0 and 1.0).

---

## 4. Evaluation Metrics & Actual Run Results

### Definitions
- **Mean Absolute Error (MAE):** $\frac{1}{N} \sum_{i=1}^N |\text{Predicted}_i - \text{Expected}_i|$
- **Spearman Rank Correlation ($\rho$):** Measures monotonicity between predicted rankings and expected ground-truth ranking without assuming linear relationships.
- **Mean Text Similarity (%):** Average cosine similarity percentage across test pairs.
- **Mean Skill Coverage (%):** Average technical skill coverage percentage across test pairs.

### Actual Run Results

```text
========================================================================================================
                                 Experiment N-Gram Range  Skill Weight     MAE  Spearman Corr  Mean Text Sim (%)  Mean Skill Cov (%)
        Experiment A (Unigrams, Weight 0.4)       (1, 1)           0.4 19.4467         0.8757              18.43               54.58
Experiment B (Unigrams+Bigrams, Weight 0.4)       (1, 2)           0.4 24.3092         0.8616              10.32               54.58
Experiment C (Unigrams+Bigrams, Weight 0.5)       (1, 2)           0.5 19.8842         0.8616              10.32               54.58
========================================================================================================
```

### Analysis and Insights
1. **Unigrams vs. Bigrams (Exp A vs. Exp B):**
   - Unigram TF-IDF vectorization (Exp A) yielded higher raw text similarity (18.43% mean vs. 10.32% in Exp B) because bigram feature spaces are sparser.
   - Both models demonstrated high Spearman rank correlation (> 0.86), confirming that the rank order of candidate alignment matches expected ground truth closely.
2. **Skill Weight Impact (Exp B vs. Exp C):**
   - Increasing the skill weight from 0.4 to 0.5 in Experiment C reduced MAE from 24.31 to 19.88 points because explicit technical skills are the dominant factor in role alignment.
3. **Artifact Generation:**
   - Every run generated an individual evaluation CSV file (`evaluation_results_*.csv`), a run configuration file (`config.json`), and a run summary report (`report.txt`).

---

## 5. How to Reproduce Experiments

### Running the Experiment Script
Run the experiment pipeline from the project root:
```bash
python experiments.py
```

Output artifacts will be saved to `./experiment_artifacts/` and MLflow metadata will be logged to `./mlruns/`.

### Viewing the MLflow UI Dashboard
To launch the interactive MLflow web dashboard:
```bash
# Windows PowerShell
$env:MLFLOW_ALLOW_FILE_STORE="true"
mlflow ui --port 5000

# macOS / Linux / Bash
export MLFLOW_ALLOW_FILE_STORE=true
mlflow ui --port 5000
```

Open your browser at:
```text
http://localhost:5000
```

### Navigating the MLflow Dashboard
1. **Experiment Selection:** Select `Resume-Job-Description-Matcher` in the left sidebar.
2. **Run Comparison:**
   - Check the boxes next to `Experiment A`, `Experiment B`, and `Experiment C`.
   - Click **Compare** to inspect side-by-side bar charts and parallel coordinates plots of `MAE`, `Spearman_Rank_Correlation`, and `skill_weight`.
3. **Inspecting Artifacts:**
   - Click into any individual run.
   - Navigate to the **Artifacts** tab to view:
     - `evaluation/`: The full per-sample prediction CSV.
     - `configuration/`: The `config.json` containing exact run settings.
     - `reports/`: Text report detailing sample count and summary statistics.
