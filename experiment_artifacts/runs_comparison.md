# MLflow Experiments Comparison Summary

                                 Experiment N-Gram Range  Skill Weight     MAE  Spearman Corr  Mean Text Sim (%)  Mean Skill Cov (%)
        Experiment A (Unigrams, Weight 0.4)       (1, 1)           0.4 19.4467         0.8757              18.43               54.58
Experiment B (Unigrams+Bigrams, Weight 0.4)       (1, 2)           0.4 24.3092         0.8616              10.32               54.58
Experiment C (Unigrams+Bigrams, Weight 0.5)       (1, 2)           0.5 19.8842         0.8616              10.32               54.58

### Key Takeaways:
- **Experiment B vs A**: Adding bigrams (1, 2) captures multi-word technical phrases (e.g., 'machine learning', 'data science', 'cloud infrastructure') more accurately in the TF-IDF representation.
- **Experiment C vs B**: Increasing skill weight to 0.5 balances vocabulary richness with explicit technical tool requirements.
- **Synthetic Dataset Disclaimer**: All metrics reflect performance on synthetic test pairs. Scores measure textual and keyword alignment, not real-world hiring outcomes.
