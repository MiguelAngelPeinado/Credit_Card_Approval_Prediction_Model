import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pathlib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier
from matplotlib.ticker import FuncFormatter


# Load Data
# Obtener la ruta del archivo actual (app.py)
file_dir = pathlib.Path(__file__).parent

# Load Data
y_test = joblib.load(file_dir / "y_test.pkl")
y_pred_xgb = joblib.load(file_dir / "y_pred_xgb.pkl")
y_pred_smote = joblib.load(file_dir / "y_pred_smote.pkl")

st.set_page_config(layout="wide")

# 👉 TÍTULO GLOBAL DEL DASHBOARD
st.title("Credit Card Approval – Project Overview")

# Tabs
tabs = st.tabs(
    [
        "Overview",
        "Model Comparison",
        "Model Balance: Precision vs Business Impact",
        "About",
    ]
)



# --- TAB 1: OVERVIEW ---
with tabs[0]:
    st.markdown(
        """
        ### 🧾 Project Summary

        The goal of this project is to predict whether a credit card application should be **approved or rejected** based on a combination of **demographic, financial, and employment-related variables**.

        ---
        #### 📚 Source Data
        - `application_record.csv`: Contains static client features (e.g. income, family, housing, car).
        - `credit_record.csv`: Contains monthly credit behavior (`STATUS`) used to derive the target variable.

        After merging and cleaning both datasets, we created a new variable `TARGET` that indicates whether a customer should be considered **risky (0)** or **eligible (1)** for a credit card.

        #### 🎯 Objective
        - Build a **classification model** to identify potentially risky clients.
        - Handle a highly **imbalanced class distribution**.
        - Evaluate **business impact** of false approvals vs missed opportunities.
        - Provide a clear and interactive dashboard to explain and visualize the process.

        ---
        """
    )

    st.markdown("### 🔍 Key Dataset Insights")

    # Load dataset
    df = pd.read_csv(file_dir / "df_model.csv")

    # Definir formateador justo aquí (antes de col1, col2, col3)
    hundred_k_formatter = FuncFormatter(lambda x, _: f"{int(x/1_000):,}K")

    # 3 columns layout
    col1, col2, col3 = st.columns(3)

    # Gráfico 1 - Class Balance
    with col1:
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        sns.countplot(x="TARGET", data=df, ax=ax1, palette=["#ff9999", "#8fd9b6"])
        ax1.set_title("Class Balance")
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(["Rejected (0)", "Approved (1)"])
        st.pyplot(fig1)
        st.markdown(
            "<div style='font-size: 1rem; text-align: center;'>The dataset is extremely imbalanced – most clients are approved, while risky clients are very rare.</div>",
            unsafe_allow_html=True,
        )

    # Gráfico 2 - Income Distribution
    with col2:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.histplot(df["AMT_INCOME_TOTAL"], bins=40, color="#91b9e6", ax=ax2, kde=True)
        ax2.set_title("Income Distribution (filtered up to 700K)")
        ax2.set_xlabel("Total Income")
        ax2.set_xlim(0, 700_000)
        ax2.xaxis.set_major_formatter(hundred_k_formatter)
        st.pyplot(fig2)
        st.markdown(
            "<div style='font-size: 1rem; text-align: center;'>Most clients report incomes under 700K. Using a more intuitive 100K-based scale clarifies the concentration.</div>",
            unsafe_allow_html=True,
        )

    # Gráfico 3 - Income by Gender
    with col3:
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        sns.boxplot(
            x="CODE_GENDER", y="AMT_INCOME_TOTAL", data=df, ax=ax3, palette="pastel"
        )
        ax3.set_title("Income by Gender")
        ax3.set_xlabel("Gender")
        ax3.set_ylabel("Total Income")
        ax3.yaxis.set_major_formatter(hundred_k_formatter)
        st.pyplot(fig3)
        st.markdown(
            "<div style='font-size: 1rem; text-align: center;'>There is a slight variation in income distribution by gender, but both groups show similar medians.</div>",
            unsafe_allow_html=True,
        )


# --- TAB 2: MODEL COMPARATION ---
with tabs[1]:
    st.header("Model Comparison")
    st.markdown(
        "We compare two modeling approaches: **Standard XGBoost** vs **XGBoost with SMOTE**."
    )

    col1, spacer1, col2, spacer2, col3 = st.columns([2, 0.5, 2, 0.5, 3])

    # --- Colores suaves y etiquetas personalizadas ---
    labels = [0, 1]
    cmap = sns.color_palette("BuGn", as_cmap=True)

    # --- Matriz para modelo XGBoost estándar ---
    with col1:
        st.subheader("Standard XGBoost")

        fig1, ax1 = plt.subplots()
        disp1 = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred_xgb,
            display_labels=[0, 1],
            cmap="Blues",
            ax=ax1,
            colorbar=False,
        )
        ax1.set_title("Confusion Matrix – XGBoost", fontsize=12)
        ax1.tick_params(labelsize=10)

        # Cambiar tamaño del texto de los números
        for text in ax1.texts:
            text.set_fontsize(16)

        # 🔴 Marcar la celda con alto número de falsos negativos (clase 0 mal predicha como 1)
        ax1.add_patch(plt.Circle((0, 1), 0.3, color="red", fill=False, linewidth=2))

        st.pyplot(fig1)

    # --- Matriz para modelo con SMOTE ---
    with col2:
        st.subheader("XGBoost + SMOTE")
        fig2, ax2 = plt.subplots()
        disp2 = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred_smote,
            display_labels=labels,
            cmap=cmap,
            ax=ax2,
            colorbar=False,
        )
        ax2.set_title("Confusion Matrix – SMOTE", fontsize=12)
        ax2.tick_params(labelsize=10)

        for text in ax2.texts:
            text.set_fontsize(16)

        st.pyplot(fig2)

    # --- Reports dinámicos ---
    with col3:
        st.subheader("📊 Classification Reports – Comparison")

        # Diccionarios con métricas clave
        metrics_xgb = {
            "Accuracy": 0.9515,
            "Precision (class 0)": 0.1728,
            "Recall (class 0)": 0.4959,
            "F1-score (class 0)": 0.2563,
            "Precision (class 1)": 0.9911,
            "Recall (class 1)": 0.9593,
            "F1-score (class 1)": 0.9749,
            "ROC AUC": 0.7663,
        }

        metrics_smote = {
            "Accuracy": 0.9782,
            "Precision (class 0)": 0.3448,
            "Recall (class 0)": 0.3252,
            "F1-score (class 0)": 0.3347,
            "Precision (class 1)": 0.9884,
            "Recall (class 1)": 0.9894,
            "F1-score (class 1)": 0.9889,
            "ROC AUC": 0.7842,
        }

        # Comparativa con formato visual
        st.markdown("#### 📈 Metric Comparison (Standard XGBoost vs XGBoost + SMOTE)")
        st.markdown(
            """
        <style>
        .green-bold {
            color: #006400;
            font-weight: bold;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        comparison_text = "| Metric | Standard XGBoost | XGBoost + SMOTE |\n|--------|------------------|------------------|\n"

        for metric in metrics_xgb.keys():
            xgb_val = metrics_xgb[metric]
            smote_val = metrics_smote[metric]
            smote_str = f"{smote_val:.4f}"
            if smote_val > xgb_val:
                smote_str = f"<span class='green-bold'>{smote_str}</span>"
            comparison_text += f"| {metric} | {xgb_val:.4f} | {smote_str} |\n"

        st.markdown(comparison_text, unsafe_allow_html=True)

        # --- Insights finales de la comparativa ---
    st.markdown("---")
    st.markdown("### 📌 Insights & Considerations", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: justify; font-size: 1.1em;'>
        The comparison reveals significant improvements when using <strong>SMOTE balancing</strong>:
    
        - The standard model struggles to detect rejected applicants (class <code>0</code>) due to strong class imbalance.
        - With SMOTE, the recall for class 0 improves substantially, helping to identify risky profiles more effectively.
        - Despite a small drop in precision, the trade-off is acceptable in a real-world scenario where rejecting a risky client is more important than rejecting a safe one.
        - The ROC AUC Score also increases with SMOTE, indicating better model discrimination.
        <br><br>
    
        After applying techniques like <strong>SMOTE</strong> to balance the classes, we observe a general improvement in global metrics (<code>ROC AUC</code>, <code>accuracy</code>, <code>f1-score</code>). However, it's important to keep the following in mind:
        <br><br>
        We have reduced the number of lost clients due to unfounded suspicion, but at the same time, we’ve slightly opened the door to mistakenly approving some risky clients.
        <br><br>
        This is a key insight in real-world projects: <strong>not all model errors have the same economic impact</strong>.
        <br><br>
        <u>Types of Errors and Their Implications:</u><br>
        <table style="border-collapse: collapse; width: 100%; font-size: 0.95em;">
            <thead>
                <tr style="background-color: #f0f0f0;">
                    <th style="border: 1px solid #ccc; padding: 8px;">Type of Error</th>
                    <th style="border: 1px solid #ccc; padding: 8px;">Interpretation</th>
                    <th style="border: 1px solid #ccc; padding: 8px;">Expected Economic Impact</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="border: 1px solid #ccc; padding: 8px;">False Positive (FP)</td>
                    <td style="border: 1px solid #ccc; padding: 8px;">A risky client is approved</td>
                    <td style="border: 1px solid #ccc; padding: 8px;">Default risk / direct loss</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ccc; padding: 8px;">False Negative (FN)</td>
                    <td style="border: 1px solid #ccc; padding: 8px;">A good client is rejected</td>
                    <td style="border: 1px solid #ccc; padding: 8px;">Lost potential revenue</td>
                </tr>
            </tbody>
        </table>
        <br>
        Therefore, beyond evaluating the model with standard classification metrics, it is essential to include a <strong>cost-effectiveness analysis</strong> that considers these two error scenarios and their economic weight in the business strategy.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- TAB 3: MODEL BALANCE ---
with tabs[2]:
    st.header("Model Balance: Precision vs Business Impact")
    st.markdown("---")

    # --- Layout principal ---
    col1, spacer, col2 = st.columns([5, 0.5, 5])

    with col1:
        st.subheader("💼 Hypothesis and Economic Assumptions")
        st.markdown(
            """
        - **Clients simulated**: 100,000  
        - **Estimated % of risky clients** (`TARGET=0`): 1.69%  
        - **Cost of False Positive (FP)**: €3,000  
        <span style='font-size:0.85em; color:gray;'>Estimated default cost per risky client mistakenly approved.</span>

        - **Cost of False Negative (FN)**: €240  
        <span style='font-size:0.85em; color:gray;'>Estimated profit loss per good client mistakenly rejected.</span>

        - **SMOTE-based model performance** (on 7,292 test samples):  
            - FP: 83  
            - FN: 75  
        """,
            unsafe_allow_html=True,
        )

        # --- Cálculos ---
        total_clients = 100_000
        risk_pct = 0.0169
        cost_fp = 3000
        cost_fn = 240

        # Sin modelo
        risky_clients = int(total_clients * risk_pct)
        no_model_fp_cost = risky_clients * cost_fp

        # Con modelo - escalamos los FP/FN del modelo real
        test_samples = 7292
        fp_ratio = 83 / test_samples
        fn_ratio = 75 / test_samples
        est_fp = int(total_clients * fp_ratio)
        est_fn = int(total_clients * fn_ratio)

        model_fp_cost = est_fp * cost_fp
        model_fn_cost = est_fn * cost_fn
        model_total_cost = model_fp_cost + model_fn_cost

        savings = no_model_fp_cost - model_total_cost
        savings_pct = (savings / no_model_fp_cost) * 100

        st.markdown(
            f"""
        ### 📉 Estimated Costs Comparison
        - **Without model**:
            - Risky clients approved: **{risky_clients:,}**
            - Total FP cost: **€{no_model_fp_cost:,.0f}**
            - FN cost: **€0**
        
        - **With model**:
            - FP: **{est_fp:,}** → €{model_fp_cost:,.0f}  
            - FN: **{est_fn:,}** → €{model_fn_cost:,.0f}  
            - **Total cost**: €{model_total_cost:,.0f}

        ### ✅ Savings by using the model: **→ €{savings:,.0f}**
        <span style='font-size:1em;'>Applying the model allows an estimated economic saving of <strong>{savings_pct:.1f}%</strong> compared to approving all applications without risk evaluation, avoiding losses of over <strong>€1.4 million</strong> for every 100,000 processed applications.</span>
        """,
            unsafe_allow_html=True,
        )

        with st.expander("📄 Details of cost_fn Calculation"):
            st.markdown(
                """
                - Estimated monthly credit limit: €3,000  
                - Average credit utilization: 40% → approx. €1,200  
                - Average interest rate (APR): 18%  
                - Additional fees and margins: 2%  
                - Estimated monthly profit: approx. €20  
                - **Expected annual profit from an approved client: €240**
                """
            )

    with col2:
        st.subheader("📊 Visual Comparison")

        categories = ["False Positives", "False Negatives"]
        no_model = [no_model_fp_cost, 0]
        with_model = [model_fp_cost, model_fn_cost]

        bar_width = 0.35
        x = np.arange(len(categories))

        fig, ax = plt.subplots(figsize=(6, 4))

        # 🎨 Colores personalizados por barra
        colors_no_model = ["#fad7a0", "#fad7a0"]  # naranja claro
        colors_with_model = ["#a2d9ce", "#a2d9ce"]  # verde oscuro

        # Dibujar barras individuales con colores personalizados
        bars1 = [
            ax.bar(
                x[i] - bar_width / 2,
                no_model[i],
                bar_width,
                color=colors_no_model[i],
                label="No Model" if i == 0 else "",
            )
            for i in range(len(no_model))
        ]

        bars2 = [
            ax.bar(
                x[i] + bar_width / 2,
                with_model[i],
                bar_width,
                color=colors_with_model[i],
                label="With Model" if i == 0 else "",
            )
            for i in range(len(with_model))
        ]

        # Etiquetas: tamaño, posición y estilo
        for i in range(len(categories)):
            for bar_group in [bars1, bars2]:
                for bar in bar_group:
                    height = bar[0].get_height()
                    ax.text(
                        bar[0].get_x() + bar[0].get_width() / 2,
                        height - (height * 0.15),  # ajustar posición arriba
                        f"€{int(height):,}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                        color="grey",
                    )

        ax.set_ylabel("Estimated Economic Cost (€)")
        ax.set_title("Cost Comparison: No Model vs SMOTE Model")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        st.pyplot(fig)

        st.markdown("---")
        st.markdown(
            """
        > This analysis simulates the economic impact of applying the model to 100,000 new applications.  
        > Despite a slight increase in good clients being rejected (FN), the cost of avoiding risky approvals (FP) justifies model usage.
        """
        )

with tabs[3]:
    st.header("📘 About this Project")
    col1, col2 = st.columns([5, 2])

    with col1:
        st.subheader("📦 About the Dataset")
        st.markdown(
            """
            - **Source files**:
                - `application_record.csv`: static client data (e.g. gender, income, housing).
                - `credit_record.csv`: monthly credit payment records.
            - **Key Columns**:
                - `AMT_INCOME_TOTAL`: Declared total income.
                - `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `OCCUPATION_TYPE`: Socioeconomic details.
                - `STATUS`: Payment behavior derived from credit history.
                - `TARGET`: Engineered label – 1 if the client is *good*, 0 if *risky*.

            - **Data Processing Steps**:
                1. Merge both datasets on `ID`.
                2. Clean duplicates and null values.
                3. Engineer `TARGET` from the most delayed credit behavior.
                4. Encode categorical features.
                5. Split into train/test for modeling.

            - **📎 Data Source**:  
              Original dataset available on Kaggle –  
              [Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)
            """
        )

        st.subheader("🧠 Modeling Context")
        st.markdown(
            """
            This project explores the challenge of approving or rejecting credit card applications using a classification model.  

            We use **XGBoost** as the main classifier, comparing a standard model with one trained using **SMOTE** to address class imbalance.  

            Beyond performance metrics, this project emphasizes the **economic cost of misclassification** and highlights the relevance of cost-based decision-making in real business contexts.
            """
        )

    with col2:
        st.subheader("📬 Contact")
        st.markdown(
            """
            **Miguel Ángel Peinado**  
            _Data Analytics & Business Strategy_  

            📧 miguel.ang.peinado@gmail.com  
            
            [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/miguel-angel-peinado-peinado/)  
            [![GitHub](https://img.shields.io/badge/GitHub-MiguelAngelPeinado-black?style=flat&logo=github)](https://github.com/MiguelAngelPeinado)
            """
        )
