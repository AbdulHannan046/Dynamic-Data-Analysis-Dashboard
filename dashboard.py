import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# Streamlit App Title
st.title("Dynamic Data Analysis Dashboard")

# File Upload Section
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

# Initialize Hugging Face model (only once)
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

qa_pipeline = load_model()

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully!")
        
        # Display the dataset
        st.subheader("Preview of the Dataset")
        st.dataframe(df)

        # Dataset Summary
        st.subheader("Dataset Summary")
        st.write("Shape of the dataset:", df.shape)
        st.write("Columns in the dataset:", list(df.columns))
        st.write("Summary statistics:")
        st.write(df.describe())

        # Column Selection for Analysis
        st.subheader("Data Analysis & Visualization")
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        st.write("Select columns for visualization:")
        if numerical_columns:
            x_axis = st.selectbox("Select X-axis (Numerical)", options=numerical_columns)
            y_axis = st.selectbox("Select Y-axis (Numerical)", options=numerical_columns)
            plot_type = st.selectbox("Select Plot Type", options=["Scatter Plot", "Line Plot", "Bar Plot"])

            if st.button("Generate Plot"):
                st.subheader(f"{plot_type} of {x_axis} vs {y_axis}")
                fig, ax = plt.subplots()
                if plot_type == "Scatter Plot":
                    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
                elif plot_type == "Line Plot":
                    sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)
                elif plot_type == "Bar Plot":
                    sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax)
                st.pyplot(fig)

        if categorical_columns:
            st.write("Categorical Analysis")
            selected_category = st.selectbox("Select a Categorical Column", options=categorical_columns)
            if st.button("Show Value Counts"):
                st.write(df[selected_category].value_counts())

        # Correlation Heatmap
        st.subheader("Correlation Heatmap (Numerical Columns)")
        if numerical_columns:
            if st.button("Generate Heatmap"):
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[numerical_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
        else:
            st.write("No numerical columns available for correlation heatmap.")
        
        # -----------------------------------------------------------
        # 🆕 Chatbot Section
        st.subheader("📢 Ask Questions About Your Data")
        user_question = st.text_input("Enter your question:")

        if user_question:
            # Prepare a prompt with data sample
            sample_data = df.head(10).to_string(index=False)
            #prompt = f"Given the following data:\n{sample_data}\n\nAnswer the following question:\n{user_question}"
            prompt = f"""
            You are a data analyst. Your job is to answer questions based on the following table of data.

            Data Table:
            {sample_data}

            Rules:
            - Only use the provided data to answer.
            - If the answer is not directly found, say "I cannot determine from the provided data."
            - Be short and precise.

            Question:
            {user_question}

            Answer:
            """
            # Get model answer
            with st.spinner("Thinking..."):
                response = qa_pipeline(prompt, max_length=200)[0]['generated_text']
            
            st.success("Answer:")
            st.write(response)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a dataset to proceed.")
