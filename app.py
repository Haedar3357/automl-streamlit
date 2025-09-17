# app.py (معدّل)
import os
import streamlit as st
import pandas as pd

from pycaret.classification import (
    setup as clf_setup, compare_models as clf_compare_models, pull as clf_pull,
    save_model as clf_save_model, predict_model as clf_predict_model,
    get_config as clf_get_config, create_model as clf_create_model,
    finalize_model as clf_finalize_model
)
from pycaret.regression import (
    setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull,
    save_model as reg_save_model, predict_model as reg_predict_model,
    get_config as reg_get_config, create_model as reg_create_model,
    finalize_model as reg_finalize_model
)

from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI
from streamlit_chat import message

# ---------- Load dataset if exists ----------
df = None
if os.path.exists("dataset.csv"):
    try:
        df = pd.read_csv("dataset.csv", index_col=None)
    except Exception as e:
        st.warning(f"Warning: failed to read dataset.csv: {e}")

# ---------- Sidebar ----------
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-02.png")
    st.title("AutoML")
    choice = st.radio("Select the task", ["Upload", "Profiling", "Modeling", "Download", "AI Assistant"])
    st.info("This project application helps you build and explore your data")

# ---------- AI Assistant (secure API key handling) ----------
if choice == "AI Assistant":
    st.title("AI Assistant")

    # Read API key securely: first check Streamlit secrets, then environment variables
    API_KEY = st.secrets.get("OPENROUTER_API_KEY") if "OPENROUTER_API_KEY" in st.secrets else os.environ.get("OPENROUTER_API_KEY")
    BASE_URL = st.secrets.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if not API_KEY:
        st.warning("OpenRouter API key not found. Add OPENROUTER_API_KEY to `.streamlit/secrets.toml` (local) or to Streamlit Cloud Secrets, or set environment variable OPENROUTER_API_KEY.")
        st.info("After you add the secret, redeploy the app on Streamlit Cloud (or restart locally).")
        client = None
    else:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    # System prompt (assistant instructions)
    SYSTEM_PROMPT = """
You are a professional AI assistant integrated inside an AutoML Streamlit app.
Your main role is to help users navigate and use the application effectively.
The app is built using Python, Streamlit, PyCaret, Pandas Profiling, and visualization libraries (Matplotlib, Seaborn).

Project structure:
1. Upload: Users can upload datasets (CSV format).
2. Profiling: Automated EDA using ydata_profiling (summary stats, distributions, correlations).
3. Modeling:
   - Classification and Regression supported.
   - Algorithms include Random Forest, KNN, Naive Bayes, SVM, XGBoost, Decision Tree, Linear Regression, Ridge, Lasso, Gradient Boosting, Elastic Net.
   - Uses PyCaret for setup, model training, evaluation, and saving.
   - Provides metrics: Accuracy, F1 Score (classification), MSE, MAE, R² (regression).
   - Displays confusion matrices and performance comparison.
4. Download: Users can download the trained/best model.
5. AI Assistant: This interactive chat assistant (you).

Your responsibilities:
- Explain different sections of the app and how to use them.
- Guide users in choosing the right models and techniques.
- Provide advice on handling data (missing values, encoding, scaling).
- Help interpret model evaluation metrics and results.
- Offer best practices for improving performance.
- Always provide your answers in two parts:
  1. Arabic explanation (clear and simple).
  2. English explanation (professional and structured).
- If the user asks something outside the project scope, politely redirect them back to the context of the AutoML app.
"""

    # Initialize session messages (keep system prompt)
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # show previous messages (except system) using streamlit_chat
    for msg in st.session_state["messages"]:
        if msg["role"] != "system":
            message(msg["content"], is_user=(msg["role"] == "user"))

    # If API key missing, we don't show the text_input to prevent errors
    if client is None:
        st.markdown("**Chat is disabled until the OpenRouter API key is configured.**")
    else:
        user_input = st.text_input("Ask me anything:")
        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.spinner("⏳ Processing..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state["messages"],
                    )
                    reply = response.choices[0].message.content
                except Exception as e:
                    reply = f"Error while connecting to model: {e}"

            st.session_state["messages"].append({"role": "assistant", "content": reply})
            message(reply)

# ---------- Upload dataset ----------
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset (CSV)", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file, index_col=None)
            df.to_csv("dataset.csv", index=False)
            st.success("Dataset uploaded and saved as dataset.csv")
            st.dataframe(df.head(10))
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

# ---------- Profiling (Custom EDA) ----------
if choice == "Profiling":
    st.title("Exploratory Data Analysis")

    if df is None:
        st.error("No dataset loaded. Please upload a CSV file in the 'Upload' tab first.")
    else:
        st.subheader("📊 Basic Info")
        st.write("**Shape:**", df.shape)
        st.write("**Data Types:**")
        st.dataframe(df.dtypes)

        st.subheader("🔎 Missing Values")
        st.dataframe(df.isnull().sum().to_frame("Missing Values"))

        st.subheader("📈 Descriptive Statistics")
        st.dataframe(df.describe(include='all').transpose())

        st.subheader("⚖️ Correlation Matrix (Numeric Columns)")
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns for correlation heatmap.")

        st.subheader("📊 Distribution Plots")
        col_choice = st.multiselect("Select numeric columns to view distributions", numeric_cols)
        for col in col_choice:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

        st.subheader("🪄 Categorical Feature Analysis")
        cat_cols = df.select_dtypes(exclude=['number']).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                st.write(f"### {col}")
                st.bar_chart(df[col].value_counts())
        else:
            st.info("No categorical columns detected.")

        st.subheader("📐 Outlier Detection (Boxplots)")
        out_cols = st.multiselect("Select numeric columns for boxplots", numeric_cols)
        for col in out_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Boxplot of {col}")
            st.pyplot(fig)


# ---------- Modeling ----------
if choice == "Modeling":
    st.title("Modeling")
    if df is None:
        st.error("No dataset loaded. Please upload a CSV file in the 'Upload' tab first.")
    else:
        chosen_target = st.selectbox("Select the target column", df.columns)
        st.subheader("Choose the algorithm type")
        algorithm_type = st.radio("Select the algorithm type", ["Classification", "Regression"])

        st.subheader("Choose to run a specific model or all models")
        run_all_models = st.radio("Run Model(s)", ["All Models", "Specific Model"])

        if algorithm_type == "Classification":
            st.subheader("Classification Algorithms")
            model_options = ['Random Forest', 'K-Nearest Neighbors', 'Naive Bayes', 'SVM', 'Extreme Gradient Boosting', 'Decision Tree Classifier']
            setup = clf_setup
            compare_models = clf_compare_models
            create_model = clf_create_model
            finalize_model = clf_finalize_model
            save_model = clf_save_model
            predict_model = clf_predict_model
            get_config = clf_get_config
            pull = clf_pull
        else:
            st.subheader("Regression Algorithms")
            model_options = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Random Forest Regressor', 'Gradient Boosting Regressor', 'Elastic Net']
            setup = reg_setup
            compare_models = reg_compare_models
            create_model = reg_create_model
            finalize_model = reg_finalize_model
            save_model = reg_save_model
            predict_model = reg_predict_model
            get_config = reg_get_config
            pull = reg_pull

        if run_all_models == "Specific Model":
            chosen_model = st.selectbox("Select the model to run", model_options)

        if st.button("Run Model(s)"):
            s = setup(data=df, target=chosen_target, normalize=True, verbose=False, html=False, session_id=123)

            if run_all_models == "All Models":
                best_model = compare_models()
                save_model(best_model, "best_model")
                model_to_use = best_model

                st.write(f"Best Model: {model_to_use}")
                st.write("Model Performance (Training Data):")
                metrics_df = pull()
                st.dataframe(metrics_df)
            else:
                # map chosen_model to pycaret id
                if chosen_model == 'Random Forest':
                    model_to_use = create_model('rf')
                elif chosen_model == 'K-Nearest Neighbors':
                    model_to_use = create_model('knn')
                elif chosen_model == 'Naive Bayes':
                    model_to_use = create_model('nb')
                elif chosen_model == 'SVM':
                    model_to_use = create_model('svm')
                elif chosen_model == 'Extreme Gradient Boosting':
                    model_to_use = create_model('xgboost')
                elif chosen_model == 'Decision Tree Classifier':
                    model_to_use = create_model('dt')
                elif chosen_model == 'Linear Regression':
                    model_to_use = create_model('lr')
                elif chosen_model == 'Ridge Regression':
                    model_to_use = create_model('ridge')
                elif chosen_model == 'Lasso Regression':
                    model_to_use = create_model('lasso')
                elif chosen_model == 'Random Forest Regressor':
                    model_to_use = create_model('rf')
                elif chosen_model == 'Gradient Boosting Regressor':
                    model_to_use = create_model('gbr')
                elif chosen_model == 'Elastic Net':
                    model_to_use = create_model('en')

                model_to_use = finalize_model(model_to_use)
                save_model(model_to_use, f"{chosen_model.replace(' ', '_')}_model")

            # Post processing: metrics & confusion matrix (same as before)
            X_train = get_config('X_train')
            y_train = get_config('y_train')
            X_test = get_config('X_test')
            y_test = get_config('y_test')

            train_predictions = predict_model(model_to_use, data=X_train)
            test_predictions = predict_model(model_to_use, data=X_test)

            label_col = 'Label' if 'Label' in test_predictions.columns else 'prediction_label'

            if label_col not in test_predictions.columns:
                st.error("Error: Predicted label column not found in predictions.")
                st.stop()

            if algorithm_type == "Classification":
                train_metrics = pd.DataFrame({
                    'Accuracy': [accuracy_score(y_train, train_predictions[label_col])],
                    'F1 Score': [f1_score(y_train, train_predictions[label_col], average='weighted')],
                })
                st.write("Train Data Performance Metrics:")
                st.dataframe(train_metrics)

                test_metrics = pd.DataFrame({
                    'Accuracy': [accuracy_score(y_test, test_predictions[label_col])],
                    'F1 Score': [f1_score(y_test, test_predictions[label_col], average='weighted')],
                })
                st.write("Test Data Performance Metrics:")
                st.dataframe(test_metrics)

                if run_all_models != "All Models":
                    cm_test = confusion_matrix(y_test, test_predictions[label_col])
                    st.write("Confusion Matrix (Test Data):")
                    fig, ax = plt.subplots()
                    sns.heatmap(cm_test, annot=True, cmap="Blues", fmt="d", ax=ax)
                    ax.set_title("Confusion Matrix (Test Data)")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)

            else:  # Regression
                train_metrics = pd.DataFrame({
                    'MSE': [mean_squared_error(y_train, train_predictions[label_col])],
                    'MAE': [mean_absolute_error(y_train, train_predictions[label_col])],
                    'R2 Score': [r2_score(y_train, train_predictions[label_col])],
                })
                st.write("Train Data Performance Metrics:")
                st.dataframe(train_metrics)

                test_metrics = pd.DataFrame({
                    'MSE': [mean_squared_error(y_test, test_predictions[label_col])],
                    'MAE': [mean_absolute_error(y_test, test_predictions[label_col])],
                    'R2 Score': [r2_score(y_test, test_predictions[label_col])],
                })
                st.write("Test Data Performance Metrics:")
                st.dataframe(test_metrics)

# ---------- Custom style ----------
st.markdown("""
    <style>
    .custom-container {
        background-color: orange;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Download model ----------
if choice == "Download":
    if os.path.exists("best_model.pkl"):
        with open('best_model.pkl', 'rb') as f:
            st.markdown('<div class="custom-container">', unsafe_allow_html=True)
            st.download_button('⬇️ Download Model', f, file_name='best_model.pkl')
            st.markdown('</div>', unsafe_allow_html=True)


