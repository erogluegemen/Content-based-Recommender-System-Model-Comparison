# Movie Recommendation System and Model Comparison

This project explores the construction and evaluation of multiple content-based recommender systems using the Tag Genome dataset. The analysis was divided into two primary paths:

1.  **Qualitative Recommendation:** Generating the most relevant, high-quality list of recommendations for a target user (User 999999).

2.  **Quantitative Prediction:** Building and comparing advanced Machine Learning models (XGBoost, CatBoost, etc.) to accurately predict a user's 1- to 5-star rating for a movie.

## Key Findings

* **Best Qualitative Model:** The **Word2Vec (NLP)** model provided the most relevant and specific recommendations. It successfully identified the user's niche preference for "Disney Channel Original Movies" (DCOMs).

* **Best Predictive Model:** The **CatBoost Regressor** was the clear champion in our offline evaluation, achieving the lowest Mean Absolute Error (MAE) of **0.9301**.

* **Core Conclusion:** The project definitively proved that treating the 1-5 star rating as a **Regression** problem (predicting a continuous value) is superior to a **Classification** problem (predicting a discrete class).

---

## Part 1: Qualitative Recommendation Generation

We built a series of progressively more complex models to generate a "Top 10" list for our target user.

### Model 1: TF-IDF (The Starting Point)

* **Problem:** Our first model, based on user reviews, was "polluted" by generic words (`good`, `best`, `original`). This led to poor and random recommendations.

* **Solution:** The most critical step was **iterative feature engineering**. We used a bar chart of word frequencies to build a custom, aggressive "stop-word" list, which was essential for finding a clean signal.

### Model 2: LSI/SVD (Topic Modeling)

* **Result:** By applying LSI to our clean TF-IDF matrix, we "denoised" the data. This model correctly identified the user's preference for the broad **"Comedy" topic**, recommending films like `Minions` and `The Muppets`.

### Model 3: Word2Vec (Semantic Meaning)

* **Result:** This was our most successful model. By learning the *semantic meaning* of words, it went beyond general topics to find the user's specific **"niche."** It correctly identified a strong preference for "Disney Channel Original Movies" (e.g., `Halloweentown II`, `Zenon: Z3`).
  
---

## Part 2: User Rating Prediction (The "Bake-off")

We held a "bake-off" between numerous ML models to see which could most accurately predict the 1-5 star rating from the `processed/10folds` data.

### Regression vs. Classification

We first tested whether to frame the problem as regression (predicting 4.8) or classification (predicting 5). Regression was the clear winner.

* **Optimized XGBoost Regressor MAE:** 0.9631
* **Optimized XGBoost Classifier MAE:** 0.9900

### Final Model Showdown

We then compared all our regression models. The `CatBoost Regressor` was the definitive champion, achieving the lowest error without complex tuning.

| Rank | Model | Model Type | MAE (Lower is better) |
| :--- | :--- | :--- | :--- |
| 1. | **CatBoost Regressor** | Gradient Boosting | **0.9301** |
| 2. | **XGBoost Regressor (Tuned)** | Gradient Boosting | **0.9631** |
| 3. | **XGBoost Regressor (Original)** | Gradient Boosting | **0.9771** |
| 4. | **Random Forest Regressor** | Bagging | **0.9848** |
| 5. | **XGBoost Classifier (Tuned)** | Gradient Boosting | **0.9900** |
| 6. | **XGBoost Classifier (Original)** | GradientBoosting | **1.0299** |
| 7. | **AdaBoost Regressor** | Boosting (Classic) | **1.0718** |
| 8. | **Linear Regression (Baseline)** | Linear | **1.0872** |

---

## Final Conclusion

This project successfully demonstrated three key findings:

1.  **For generating high-quality recommendations**, `Word2Vec` is superior because it understands the semantic *niche* of the content, not just keywords.

2.  **For predicting user ratings**, the `CatBoost Regressor` is the most accurate model, validating the power of modern gradient boosting.

3.  Finally, **feature engineering** (like our stop-word list) and **correct problem-framing** (Regression > Classification) proved to be more important than any individual algorithm.
