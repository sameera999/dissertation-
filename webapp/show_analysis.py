import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

def load_data(model_name):
    # Load the feature importance data
    if model_name == 'KNN':
        return None
    
    feature_importances = pd.read_csv(f'C:/Users/Sameera/OneDrive - York St John University/MYPROJECT/processedData/{model_name}_feature_importances.csv', index_col=0)

    return feature_importances

def create_plot(feature_importances):
    if feature_importances is None:
        return None

    # Create a bar plot of feature importance
    plt.figure(figsize=(10, 5))
    plt.barh(feature_importances.index, feature_importances['importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    fig = plt.gcf()
    plt.close()
    return fig

def load_results(model_name):
    # Load the model results data
    results = pd.read_csv(f'C:/Users/Sameera/OneDrive - York St John University/MYPROJECT/processedData/{model_name}_model_results.csv')

    return results

def create_line_plot(results):
    # Select every 100th point for plotting
    results = results.iloc[::500, :]
    
    # Create a line plot of y_test and y_pred
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot y_test and y_pred over time on the first subplot
    axs[0].plot(results.index, results['y_test'], marker='o', label='Actual')
    axs[0].plot(results.index, results['y_pred'], marker='o', color='r', label='Predicted')
    axs[0].set_title('Actual and Predicted Over Time')
    axs[0].set_xlabel('Time Sequence')
    axs[0].set_ylabel('Value')
    axs[0].legend()

    # Plot y_test over time on the second subplot
    axs[1].plot(results.index, results['y_test'], marker='o', color='b', label='Actual')
    axs[1].set_title('Actual Over Time')
    axs[1].set_xlabel('Time Sequence')
    axs[1].set_ylabel('Value')
    axs[1].legend()

    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    return fig

def show_model_analysis(model_name):
    st.title(f'{model_name} Model Analysis')

    feature_importances = load_data(model_name)
    fig1 = create_plot(feature_importances)

    results = load_results(model_name)
    fig2 = create_line_plot(results)

    # Use Streamlit's columns feature to display plots side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Feature Importance')
        st.pyplot(fig1)

    with col2:
        st.subheader('Model Results')
        st.pyplot(fig2)

def show_analysis():    
    models = ['RandomForest', 'XGBoost','KNN']
    for model_name in models:
        show_model_analysis(model_name)
