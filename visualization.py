import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import streamlit as st

def plot_parent_correlations(df, target):
    features = [
        "Father's occupation",
        "Father's qualification",
        "Mother's occupation",
        "Mother's qualification"
    ]
    corrs = {feat: df[feat].corr(df[target]) for feat in features}
    corr_df = pd.DataFrame(list(corrs.items()), columns=['Feature', 'Correlation with Target'])
    corr_df = corr_df.sort_values(by='Correlation with Target', ascending=False)
    
    st.subheader("Parents Occupation & Qualification Correlations (Before Preprocessing)")
    st.dataframe(corr_df.style.format({"Correlation with Target": "{:.2f}"}))
    
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=corr_df, x='Feature', y='Correlation with Target', palette='pastel', ax=ax)
    ax.set_ylim(-1,1)
    plt.xticks(rotation=45)
    plt.title("Parents Occupation & Qualification Correlations")
    st.pyplot(fig)

def visualize_preprocessing_effects(df_before, df_after):
    target = 'Target'

    # 1. Ön İşleme Öncesi sütunlardan bazı oran ve bilgi mühendisliği sütunları (önceki df_before)
    df_before['sem_1_pass_rate'] = df_before["Curricular units 1st sem (approved)"] / df_before["Curricular units 1st sem (enrolled)"]
    df_before['sem_2_pass_rate'] = df_before["Curricular units 2nd sem (approved)"] / df_before["Curricular units 2nd sem (enrolled)"]

    df_before['father_knowledge'] = (df_before["Father's occupation"] * df_before["Father's qualification"]) / 2
    df_before['mother_knowledge'] = (df_before["Mother's occupation"] * df_before["Mother's qualification"]) / 2

    # 2. Ön İşleme Sonrası mühendislik özellikler df_after içinde zaten var (sem_1_pass_rate, sem_2_pass_rate, father_knowledge, mother_knowledge vb.)

    st.title("Ön İşleme Öncesi ve Sonrası Veri Karşılaştırması")

    # Korelasyonları karşılaştırma tablosu

    corr_before = {
        'sem_1_pass_rate_before': df_before['sem_1_pass_rate'].corr(df_before[target]),
        'sem_2_pass_rate_before': df_before['sem_2_pass_rate'].corr(df_before[target]),
        'father_knowledge_before': df_before['father_knowledge'].corr(df_before[target]),
        'mother_knowledge_before': df_before['mother_knowledge'].corr(df_before[target]),
    }

    corr_after = {
        'sem_1_pass_rate_after': df_after['sem_1_pass_rate'].corr(df_after[target]),
        'sem_2_pass_rate_after': df_after['sem_2_pass_rate'].corr(df_after[target]),
        'father_knowledge_after': df_after['father_knowledge'].corr(df_after[target]),
        'mother_knowledge_after': df_after['mother_knowledge'].corr(df_after[target]),
    }

    corr_df = pd.DataFrame([corr_before, corr_after], index=['Before Preprocessing', 'After Preprocessing']).T.reset_index()
    corr_df.columns = ['Feature', 'Before', 'After']

    st.subheader("Feature Correlations with Target Before and After Preprocessing")
    st.dataframe(corr_df.style.format({"Before": "{:.2f}", "After": "{:.2f}"}))

    # Histogramları karşılaştır (Öncesi ve sonrası yan yana subplot)

    fig, axs = plt.subplots(4, 2, figsize=(14, 14))

    features = ['sem_1_pass_rate', 'sem_2_pass_rate', 'father_knowledge', 'mother_knowledge']
    colors_before = ['skyblue', 'orange', 'green', 'purple']
    colors_after = ['lightblue', 'gold', 'lightgreen', 'violet']

    for i, feat in enumerate(features):
        sns.histplot(df_before[feat].dropna(), bins=20, ax=axs[i, 0], color=colors_before[i])
        axs[i, 0].set_title(f'{feat} (Before Preprocessing)')

        sns.histplot(df_after[feat].dropna(), bins=20, ax=axs[i, 1], color=colors_after[i])
        axs[i, 1].set_title(f'{feat} (After Preprocessing)')

    plt.tight_layout()
    st.pyplot(fig)

    # Scatter plot örneği: Pass rate vs Target (önce ve sonra)

    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))

    sns.scatterplot(data=df_before, x='sem_1_pass_rate', y=target, ax=axs2[0,0], alpha=0.6, color='skyblue')
    axs2[0,0].set_title('1st Sem Pass Rate vs Target (Before)')

    sns.scatterplot(data=df_after, x='sem_1_pass_rate', y=target, ax=axs2[0,1], alpha=0.6, color='lightblue')
    axs2[0,1].set_title('1st Sem Pass Rate vs Target (After)')

    sns.scatterplot(data=df_before, x='father_knowledge', y=target, ax=axs2[1,0], alpha=0.6, color='green')
    axs2[1,0].set_title("Father's Knowledge vs Target (Before)")

    sns.scatterplot(data=df_after, x='father_knowledge', y=target, ax=axs2[1,1], alpha=0.6, color='lightgreen')
    axs2[1,1].set_title("Father's Knowledge vs Target (After)")

    plt.tight_layout()
    st.pyplot(fig2)

    # Anne Baba occupation & qualification korelasyonları (ön işleme öncesi df_before'dan)
    plot_parent_correlations(df_before, target)


def comp(df, processed_df):
    col = "Admission grade"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(df[col].dropna(), bins=20, color='skyblue', alpha=0.7)
    axes[0].set_title('Admission Grade Distribution (Before)')
    axes[0].set_xlabel('Admission Grade')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(processed_df[col].dropna(), bins=20, color='orange', alpha=0.7)
    axes[1].set_title('Admission Grade Distribution (After)')
    axes[1].set_xlabel('Admission Grade')
    axes[1].set_ylabel('Frequency')

    min_before = df[col].min()
    max_before = df[col].max()
    min_after = processed_df[col].min()
    max_after = processed_df[col].max()

    axes[2].bar(['Min Before', 'Max Before'], [min_before, max_before], color='skyblue', alpha=0.7)
    axes[2].bar(['Min After', 'Max After'], [min_after, max_after], color='orange', alpha=0.7)
    axes[2].set_title('Min-Max Values Before & After')
    axes[2].set_ylabel('Value')

    plt.tight_layout()
    st.pyplot(fig)

    missing_before = df[col].isnull().sum()
    missing_after = processed_df[col].isnull().sum()
    
def comparison_visualization(df, processed_df, col):
    

    st.subheader("Feature Distribution Comparison")
    st.write(f"{col} Column")
    original_data = df[col]
    processed_data = processed_df[col]

    data_to_plot = pd.DataFrame({
        "Original": original_data,
        "Processed": processed_data
    })

    st.bar_chart(data_to_plot)

def plot_roc_curve(model, X_test, y_test):
    # Predict probabilities for the positive class
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")

    st.pyplot(fig)

def show_model_performance(model, X_test, y_test):
    st.header("7. Model Performance")
    st.write("""
    The model's performance was evaluated using various metrics such as accuracy, precision, recall, and F1-score on a separate test set to ensure 
    an objective assessment of its generalization capability.

    Additionally, to better understand the model's performance across classes, visualization tools like the ROC curve have been employed. 
    The ROC curve demonstrates the trade-off between sensitivity and specificity for different threshold values.

    Below is the ROC curve for the primary model `lgbm_voting_clf`, which illustrates its ability to discriminate between classes effectively.
    """)

    plot_roc_curve(model, X_test, y_test)

    st.write("""
    The area under the curve (AUC) value provides a summary measure of the model’s discriminative ability. An AUC closer to 1 indicates excellent performance.
    """)

# Örnek kullanım (Streamlit uygulamasında):
# from your_model_module import lgbm_voting_clf, X_test, y_test
# show_model_performance(lgbm_voting_clf, X_test, y_test)
