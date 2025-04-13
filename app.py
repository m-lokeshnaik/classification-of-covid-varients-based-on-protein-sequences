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

# Mapping for 20 standard amino acids
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_int = {aa: idx+1 for idx, aa in enumerate(amino_acids)}

def sequence_to_numeric(seq):
    return [aa_to_int.get(aa.upper(), 0) for aa in seq]

def import_from_google_sheets(sheet_url):
    try:
        # Convert the Google Sheet URL to a CSV export link
        csv_export_url = sheet_url.replace("/edit?usp=drive_link", "/export?format=csv")
        
        # Read the Google Sheet as a pandas DataFrame
        df = pd.read_csv(csv_export_url)
        
        # Display success message and preview
        st.success("✅ Data imported successfully!")
        st.write("Preview of imported data:")
        st.dataframe(df.head())
        
        return df
    except Exception as e:
        st.error(f"❌ Error importing from Google Sheets: {e}")
        return None

# App title
st.title('Protein Sequence Classification')

# Input method selection
input_method = st.radio(
    "Choose input method:",
    ["Upload Excel File", "Enter Sequences Manually", "Import from Google Sheets"]
)

if input_method == "Upload Excel File":
    uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
elif input_method == "Enter Sequences Manually":
    num_seqs = st.number_input("How many sequences do you want to input?", min_value=1, value=1)
    data_rows = []
    
    for i in range(num_seqs):
        st.write(f"\nSequence {i+1}:")
        label = st.selectbox(f"Select class label for sequence {i+1}", list(label_mapping.keys()))
        sequence = st.text_input(f"Enter amino acid sequence {i+1} (only letters A-Y):").strip().upper()
        
        if sequence:
            numeric_seq = sequence_to_numeric(sequence)
            row = numeric_seq + [label]
            data_rows.append(row)
    
    if data_rows:
        max_len = max(len(row) - 1 for row in data_rows)
        columns = [f"pos_{i+1}" for i in range(max_len)] + ["class labels"]
        
        for row in data_rows:
            while len(row) < max_len + 1:
                row.insert(-1, 0)
        
        df = pd.DataFrame(data_rows, columns=columns)
elif input_method == "Import from Google Sheets":
    default_url = "https://docs.google.com/spreadsheets/d/1Qk8F46CATZAKqF0DYLPE88Mu9MyUTGzJC29_GMUtahA/edit?usp=drive_link"
    sheet_url = st.text_input("Google Sheets URL:", value=default_url)
    
    if st.button("Import Data"):
        df = import_from_google_sheets(sheet_url)
        if df is not None:
            # Save to Excel for consistency
            excel_path = "imported_data.xlsx"
            df.to_excel(excel_path, index=False)
            st.success(f"✅ Data saved to {excel_path}")

if 'df' in locals() and not df.empty:
    try:
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
