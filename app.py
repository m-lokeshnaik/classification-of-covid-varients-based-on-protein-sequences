import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2
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
            st.error("The input data must contain a 'class labels' column.")
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
                selector = SelectKBest(score_func=f_classif, k=k)
            elif method == 'Mutual Information':
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
            elif method == 'Information Gain':
                # Calculate entropy reduction for each feature
                base_entropy = -np.sum([(y == c).mean() * np.log2((y == c).mean() + 1e-10) for c in np.unique(y)])
                scores = []
                for col in X.columns:
                    # Discretize continuous features into 10 bins
                    bins = np.linspace(X[col].min(), X[col].max(), 11)
                    binned = np.digitize(X[col], bins)
                    # Calculate conditional entropy
                    cond_entropy = 0
                    for bin_val in np.unique(binned):
                        p_bin = (binned == bin_val).mean()
                        bin_dist = y[binned == bin_val]
                        if len(bin_dist) > 0:
                            p_classes = [(bin_dist == c).mean() for c in np.unique(y)]
                            p_classes = [p for p in p_classes if p > 0]
                            cond_entropy += p_bin * -np.sum([p * np.log2(p + 1e-10) for p in p_classes])
                    scores.append(base_entropy - cond_entropy)
                
                # Create a custom selector using information gain scores
                scores = np.array(scores)
                selected_features = np.argsort(scores)[-k:]
                selector = type('CustomSelector', (), {
                    'get_support': lambda self, indices=False: selected_features if indices else np.isin(range(len(scores)), selected_features),
                    'fit_transform': lambda self, X, y=None: X.iloc[:, selected_features],
                    'scores_': scores
                })()
                return X.iloc[:, selected_features], selected_features, X.columns[selected_features], selector
            elif method == 'Chi-Square':
                # Ensure data is non-negative for chi-square test
                X_non_neg = X - X.min()
                selector = SelectKBest(score_func=chi2, k=k)
                X_new = selector.fit_transform(X_non_neg, y)
                selected_features = selector.get_support(indices=True)
                feature_names = X.columns[selected_features]
                return X_new, selected_features, feature_names, selector

            X_new = selector.fit_transform(X, y)
            selected_features = selector.get_support(indices=True)
            feature_names = X.columns[selected_features]
            return X_new, selected_features, feature_names, selector

        method = st.selectbox("Feature selection method", 
                            ["ANOVA", "Mutual Information", "Information Gain", "Chi-Square"])
        k = 10  # Default number of features
        
        # Display feature selection method description
        if method == "ANOVA":
            st.info("ANOVA: Selects features based on the F-value between label/feature for regression tasks.")
        elif method == "Mutual Information":
            st.info("Mutual Information: Measures the mutual dependence between two variables, specifically the label and feature.")
        elif method == "Information Gain":
            st.info("Information Gain: Measures the reduction in entropy when splitting by a feature.")
        elif method == "Chi-Square":
            st.info("Chi-Square: Measures the dependence between features and labels using chi-square statistic.")
        
        X_selected, selected_features, feature_names, selector = select_features(X, y, method=method, k=k)
        
        # Display selected features and their scores
        st.write("### Selected Features")
        scores = pd.DataFrame({
            'Feature': feature_names,
            'Score': selector.scores_[selected_features] if hasattr(selector, 'scores_') else np.zeros(len(selected_features))
        }).sort_values('Score', ascending=False)
        st.dataframe(scores)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
        
        # Optimize model parameters
        model_params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'scale_pos_weight': 1,
            'eval_metric': 'Accuracy',
            'random_seed': 42
        }
        
        # Create and train a new model with optimized parameters
        optimized_model = CatBoostClassifier(**model_params)
        optimized_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        
        # Model predictions
        y_pred = optimized_model.predict(X_test)
        
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
                exp = explainer.explain_instance(X_test[0], optimized_model.predict_proba, num_features=10)
                st.write(exp.as_list())
                fig = exp.as_pyplot_figure()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred while generating LIME explanation: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
