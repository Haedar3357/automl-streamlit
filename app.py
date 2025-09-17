import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from openai import OpenAI
from streamlit_chat import message
import joblib

# ---------- Load dataset if exists ----------
df = None
if os.path.exists("dataset.csv"):
    try:
        df = pd.read_csv("dataset.csv")
    except Exception as e:
        st.warning(f"Warning: failed to read dataset.csv: {e}")

# ---------- Sidebar ----------
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-02.png")
    st.title("AutoML")
    choice = st.radio("Select the task", ["Upload", "Profiling", "Modeling", "Download", "AI Assistant"])
    st.info("This project application helps you build and explore your data")

# ---------- AI Assistant ----------
if choice == "AI Assistant":
    st.title("AI Assistant")
    API_KEY = st.secrets.get("OPENROUTER_API_KEY") if "OPENROUTER_API_KEY" in st.secrets else os.environ.get("OPENROUTER_API_KEY")
    BASE_URL = st.secrets.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if not API_KEY:
        st.warning("OpenRouter API key not found.")
        client = None
    else:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    SYSTEM_PROMPT = """You are a professional AI assistant integrated inside an AutoML Streamlit app."""

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in st.session_state["messages"]:
        if msg["role"] != "system":
            message(msg["content"], is_user=(msg["role"] == "user"))

    if client is not None:
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

# ---------- Upload ----------
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset (CSV)", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
            df.to_csv("dataset.csv", index=False)
            st.success("Dataset uploaded and saved as dataset.csv")
            st.dataframe(df.head(10))
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

# ---------- Profiling ----------
if choice == "Profiling":
    st.title("Exploratory Data Analysis & Cleaning")
    
    if df is None:
        st.error("No dataset loaded.")
    else:
        # 1️⃣ إزالة الأعمدة المكررة
        df = df.loc[:,~df.columns.duplicated()]
        st.write("**Shape after removing duplicate columns:**", df.shape)
        
        st.subheader("Data Types")
        st.dataframe(df.dtypes)
        
        # 2️⃣ معالجة القيم الفارغة
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                col_std = abs(df[col].std())
                if col_std < 1:
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
        
        cat_cols = df.select_dtypes(exclude=['number']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        st.subheader("Missing Values After Cleaning")
        st.dataframe(df.isnull().sum().to_frame("Missing Values"))
        
        # 3️⃣ الإحصاءات الوصفية
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(include='all').transpose())
        
        # 4️⃣ مصفوفة الارتباط
        if len(numeric_cols) > 1:
            st.subheader("Correlation Matrix")
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        
        # 5️⃣ توزيعات البيانات
        st.subheader("Distribution Plots")
        col_choice = st.multiselect("Select numeric columns to view distributions", numeric_cols)
        for col in col_choice:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)
        
        # 6️⃣ تحليل القيم الشاذة
        st.subheader("Outlier Detection (IQR method)")
        outlier_summary = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
            count = outliers.shape[0]
            perc = (count / df.shape[0]) * 100
            outlier_summary.append({'Column': col, 'Outliers Count': count, 'Outliers %': perc})
        
        st.dataframe(pd.DataFrame(outlier_summary))
        
        # 7️⃣ تحليل البيانات الفئوية
        if len(cat_cols) > 0:
            st.subheader("Categorical Feature Analysis")
            for col in cat_cols:
                st.write(f"### {col}")
                st.bar_chart(df[col].value_counts())
        
        # 8️⃣ حفظ البيانات المعالجة في session_state لاستخدامها لاحقاً
        st.session_state['df_clean'] = df
        st.success("✅ Data cleaned and stored for Modeling.")



# ---------- Modeling Manual ----------
if choice == "Modeling":
        st.title("Modeling")
        
        if 'df_clean' in st.session_state:
            df_model = st.session_state['df_clean']
        elif df is not None:
            df_model = df
        else:
            st.error("No dataset loaded.")
            st.stop()
        
        target_col = st.selectbox("Select target column", df_model.columns)
        X = df_model.drop(columns=[target_col])
        y = df_model[target_col]
    
        # Encode categorical features
        for c in X.select_dtypes(include='object').columns:
            X[c] = LabelEncoder().fit_transform(X[c])
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        


        model_type = st.radio("Algorithm type", ["Classification", "Regression"])

        if model_type == "Classification":
            models = {
                'Random Forest': RandomForestClassifier(),
                'KNN': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB(),
                'SVM': SVC(probability=True),
                'Decision Tree': DecisionTreeClassifier(),
                'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            }
        else:
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'ElasticNet': ElasticNet(),
                'Random Forest Regressor': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'XGBoost Regressor': xgb.XGBRegressor()
            }

        chosen_model_name = st.selectbox("Select Model", list(models.keys()))
        run_model = st.button("Run Model")

        if run_model:
            model = models[chosen_model_name]
            model.fit(X_train, y_train)
            joblib.dump(model, f"{chosen_model_name.replace(' ', '_')}_model.pkl")

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            if model_type == "Classification":
                train_metrics = {
                    'Accuracy': accuracy_score(y_train, y_pred_train),
                    'F1 Score': f1_score(y_train, y_pred_train, average='weighted')
                }
                test_metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred_test),
                    'F1 Score': f1_score(y_test, y_pred_test, average='weighted')
                }
                st.subheader("Train Metrics")
                st.dataframe(pd.DataFrame([train_metrics]))
                st.subheader("Test Metrics")
                st.dataframe(pd.DataFrame([test_metrics]))

                cm = confusion_matrix(y_test, y_pred_test)
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)
            else:
                train_metrics = {
                    'MSE': mean_squared_error(y_train, y_pred_train),
                    'MAE': mean_absolute_error(y_train, y_pred_train),
                    'R2': r2_score(y_train, y_pred_train)
                }
                test_metrics = {
                    'MSE': mean_squared_error(y_test, y_pred_test),
                    'MAE': mean_absolute_error(y_test, y_pred_test),
                    'R2': r2_score(y_test, y_pred_test)
                }
                st.subheader("Train Metrics")
                st.dataframe(pd.DataFrame([train_metrics]))
                st.subheader("Test Metrics")
                st.dataframe(pd.DataFrame([test_metrics]))

# ---------- Download ----------
if choice == "Download":
    for file in os.listdir():
        if file.endswith("_model.pkl"):
            with open(file, 'rb') as f:
                st.download_button(f"Download {file}", f, file_name=file)


