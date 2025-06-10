import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import plotly.graph_objects as go
import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing import preprocess_data

non_processed_df, df, X_train, X_test, y_train, y_test = preprocess_data()

def plot_rf_roc_curve(y_true, y_pred_proba):
    """Random Forest için ROC eğrisi çizimi"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # ROC eğrisi
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'Random Forest (AUC = {roc_auc:.3f})',
        line=dict(color='darkgreen', width=3),
        hovertemplate='<b>FPR:</b> %{x:.3f}<br><b>TPR:</b> %{y:.3f}<br><b>Threshold:</b> %{text:.3f}<extra></extra>',
        text=thresholds
    ))
    
    # Random classifier çizgisi
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier (AUC = 0.500)',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve - Random Forest Model',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=500,
        showlegend=True,
        legend=dict(x=0.6, y=0.1),
        template='plotly_white'
    )
    
    # Grid ekle
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig, roc_auc

def calculate_rf_metrics(y_true, y_pred_proba):
    """Random Forest için AUC metrikleri"""
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    
    return {
        'auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'optimal_tpr': optimal_tpr,
        'optimal_fpr': optimal_fpr
    }

# Random Forest model eğitimi butonunuzdan sonra bu kodu ekleyin:

# Random Forest ROC Analizi
if 'rf_is_run' in st.session_state and st.session_state.rf_is_run:
    st.subheader("📈 Random Forest ROC Curve Analysis")
    
    # RF modelinin probability predictions
    if hasattr(st.session_state.rf_model, 'predict_proba'):
        y_pred_proba = st.session_state.rf_model.predict_proba(X_test)[:, 1]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # ROC eğrisi
            fig, roc_auc = plot_rf_roc_curve(y_test, y_pred_proba)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Metrikler
            metrics = calculate_rf_metrics(y_test, y_pred_proba)
            
            st.metric("🎯 AUC Score", f"{metrics['auc']:.3f}")
            st.metric("⚖️ Optimal Threshold", f"{metrics['optimal_threshold']:.3f}")
            st.metric("✅ True Positive Rate", f"{metrics['optimal_tpr']:.3f}")
            st.metric("❌ False Positive Rate", f"{metrics['optimal_fpr']:.3f}")
            
            # AUC performans yorumu
            st.subheader("📊 Model Performance")
            if metrics['auc'] >= 0.9:
                st.success("🎯 **Excellent** (0.9-1.0)")
                st.write("Model çok başarılı!")
            elif metrics['auc'] >= 0.8:
                st.success("✅ **Good** (0.8-0.9)")
                st.write("Model başarılı!")
            elif metrics['auc'] >= 0.7:
                st.warning("⚠️ **Fair** (0.7-0.8)")
                st.write("Model orta düzeyde.")
            elif metrics['auc'] >= 0.6:
                st.warning("🔶 **Poor** (0.6-0.7)")
                st.write("Model zayıf performans.")
            else:
                st.error("❌ **Fail** (0.5-0.6)")
                st.write("Model başarısız!")
        
        # AUC açıklaması
        st.info("""
        **AUC (Area Under Curve) Nedir?**
        
        - **1.0**: Mükemmel sınıflandırıcı
        - **0.9-1.0**: Mükemmel performans
        - **0.8-0.9**: İyi performans  
        - **0.7-0.8**: Orta performans
        - **0.6-0.7**: Zayıf performans
        - **0.5**: Rastgele tahmin (hiç faydalı değil)
        """)
    
    else:
        st.error("❌ Random Forest modeli probability prediction desteklemiyor!")
else:
    st.info("ℹ️ ROC eğrisini görmek için önce Random Forest modelini eğitin.")
    