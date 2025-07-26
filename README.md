 Edu Performance Predictor

A Streamlit-based web app that predicts student performance using activity data, provides explainability via SHAP, and offers personalized learning recommendations.



 Features

-  Predicts student final grade using real interaction data from the [OULAD dataset](https://analyse.kmi.open.ac.uk/open-dataset/)
-  Built with XGBoost + SHAP for interpretable AI
-  Visual SHAP plots to explain individual predictions
-  Personalized suggestions to improve performance
-  Deployed as an interactive web app with Streamlit

---

 Technologies Used

- Python
- XGBoost
- SHAP
- scikit-learn
- pandas
- Streamlit
- matplotlib

---

 Dataset (OULAD)   https://analyse.kmi.open.ac.uk/open-dataset

We use the `studentVle.csv` from the OULAD dataset.

 The CSV is not included in the repo due to GitHub size limits.  
To fetch it automatically, run:

bash
python download_data.py
