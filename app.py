import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import lime
import lime.lime_tabular

# Load the trained model
model = joblib.load('MAAPRED.pkl')

# Load the label encoder
label_mapping = {'Alpha': 0, 'Beta': 1, 'Delta': 2, 'Gamma': 3, 'HCoV-HKU1': 4, 'MERS-CoV': 5, 'Normal': 6, 'omicron': 7}
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(list(label_mapping.keys()))

# App title
st.title('Protein Sequence Classification')

# Upload input file
uploaded_file = st.file_uploader(r"C:\Users\lokes\Desktop\classification-of-covid-varients-based-on-protein-sequences\protein sequence in numreical.xlsx", type="xlsx")

if uploaded_file:
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)
        
        # Check if 'class labels' column is present
        if 'class labels' not in df.columns:
            st.error("The input file must contain a 'class labels' column.")
            st.stop()

        # Data preprocessing
        scaler = RobustScaler()
        numerical_cols = df.columns[:-1]
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        X = df.drop(columns=['class labels'])
        y = df['class labels']
        y = label_encoder.transform(y)
        
        # Feature selection
        def select_features(X, y, method, k=10):
            if method == 'ANOVA':
                from sklearn.feature_selection import f_classif
                selector = SelectKBest(score_func=f_classif, k=k)
            elif method == 'Mutual Information':
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_new = selector.fit_transform(X, y)
            selected_features = selector.get_support(indices=True)
            feature_names = X.columns[selected_features]
            return X_new, selected_features, feature_names, selector

        method = st.selectbox("Feature selection method", ["ANOVA", "Mutual Information"])
        k = st.slider("Number of features", min_value=1, max_value=min(X.shape[1], 20), value=10)
        
        X_selected, selected_features, feature_names, _ = select_features(X, y, method=method, k=k)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
        
        # Model predictions
        y_pred = model.predict(X_test)
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        # Display results
        st.write("### Evaluation Metrics")
        st.write(f"Accuracy: {accuracy * 100:.2f}%")
        st.write(f"Precision: {precision * 100:.2f}%")
        st.write(f"Recall: {recall * 100:.2f}%")
        
        # Confusion matrix
        st.write("### Confusion Matrix")
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        st.pyplot(plt.gcf())
        
        # Random sample predictions
        st.write("### Random Sample Predictions")
        random_indices = np.random.choice(len(y_test), 5, replace=False)
        for i in random_indices:
            st.write(f"Sample {i+1}:")
            st.write(f"  Predicted Label: {label_encoder.inverse_transform([y_pred[i]])[0]}")
            st.write(f"  True Label: {label_encoder.inverse_transform([y_test[i]])[0]}")
            st.write("")

        # Lime explanation
        st.write("### Lime Explanation for a Sample")
        if st.button('Generate Lime Explanation for a Random Sample'):
            try:
                explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=label_encoder.classes_, verbose=True, mode='classification')
                exp = explainer.explain_instance(X_test[0], model.predict_proba, num_features=10)
                st.write(exp.as_list())
                fig = exp.as_pyplot_figure()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred while generating LIME explanation: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
