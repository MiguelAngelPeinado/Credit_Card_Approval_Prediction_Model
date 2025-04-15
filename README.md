# 💳 Credit Card Approval – Streamlit Dashboard

This project presents a complete analysis and interactive dashboard focused on predicting whether a credit card application should be **approved or rejected** using classification models. It highlights how to balance accuracy with **real business impact**.

## 📁 Structure

Credit_card_approval/ 
├── App/ # Streamlit app files 
│	├── app.py # Main Streamlit app 
│ 	├── df_model.csv # Final dataset for the app 
│ 	├── model_final.pkl # Trained model 
│ 	├── y_test.pkl # Test target 
│ 	├── y_pred_xgb.pkl # Predictions - XGBoost 
│ 	├── y_pred_smote.pkl # Predictions - XGBoost + SMOTE 
│ 
├── Images/ # Visuals used in the dashboard 
├── Data/ # Original raw data from Kaggle 
├── Python/ # Jupyter notebooks for EDA & model training 
├── requirements.txt 
└── README.md


## 📚 Dataset Source

The dataset was obtained from [Kaggle – Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction), consisting of:

- `application_record.csv`: Static client features (income, job, family, housing, etc.)
- `credit_record.csv`: Monthly credit payment behavior used to define the target variable.


## 📊 Project Highlights

- ⚖️ **Imbalanced classification** using XGBoost and SMOTE.
- 📈 **Model evaluation** with classification reports and confusion matrices.
- 💰 **Economic impact analysis** comparing risk costs with and without a model.
- 💡 **Interactive Streamlit dashboard** to explore and explain insights.

## 🚀 Deployment

1. Clone the repository:
```bash
git clone https://github.com/MiguelAngelPeinado/Credit_Card_Approval_Prediction_Model.git


🙋 Author
Miguel Ángel Peinado
📧 miguel.ang.peinado@gmail.com
🔗 https://www.linkedin.com/in/miguel-angel-peinado-peinado/
💻 https://github.com/MiguelAngelPeinado
