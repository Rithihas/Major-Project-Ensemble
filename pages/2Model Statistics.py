import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, mean_absolute_error, mean_squared_error, r2_score, average_precision_score, classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import plotly.express as px

st.set_page_config(page_title='Model_Statistics', layout='wide')
reduce_header_height_style = """
    <style>
        div.block-container {padding-top:0rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)

def plot_confusion_matrix(true_labels,predicted_labels, class_labels=['AD','MCI','NC']):
    
    cm_df = pd.DataFrame(confusion_matrix(true_labels, predicted_labels), index=class_labels, columns=class_labels)
    fig = px.imshow(cm_df,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    x=class_labels,
                    y=class_labels,
                    text_auto=True)
    return fig

def classification_metrics(true_labels,predicted_labels, labels=None):
    metrics = dict()
    metrics['report'] = classification_report(true_labels, predicted_labels, output_dict=True, target_names=labels)
    
    lb = LabelBinarizer()
    true_m = lb.fit_transform(true_labels)
    pred_m = lb.transform(predicted_labels)
        
    metrics['Accuracy'] = np.round(accuracy_score(true_labels, predicted_labels)*100, decimals=2)
    metrics['Precision'] = np.round(precision_score(true_m, pred_m, average="macro")*100, decimals=2)
    metrics['Recall'] = np.round(recall_score(true_m, pred_m, average="macro")*100, decimals=2)
    metrics['F1'] = np.round(f1_score(true_m, pred_m, average="macro")*100, decimals=2)
    metrics['ROC_AUC'] = np.round(roc_auc_score(true_m, pred_m, multi_class="ovr")*100, decimals=2)
        
    cm = confusion_matrix(true_labels, predicted_labels)
    FP = np.sum(cm, axis=0) - np.diag(cm)
    TN = np.sum(cm) - np.sum(cm, axis=0) - np.sum(cm, axis=1) + np.diag(cm)

    total_FP = np.sum(FP)
    total_TN = np.sum(TN)
    
    metrics['False Positive Rate'] = np.round((total_FP / (total_FP + total_TN))*100, decimals=2)
        
    return metrics



df = pd.read_csv("preds.csv")
st.markdown("<h1 style='margin-top:35px;padding-top:10px;text-align: center'>Model Statistics</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 24px;text-align: center'>Our ensemble model achieved the following metrics when trained on 4562 MRI images, and tested on 1268 MRI images.</p>",unsafe_allow_html=True)
st.write("---")

true_labels, predicted_labels = df['y_true'], df['y_pred_ensemble']
metrics_ensemble = classification_metrics(true_labels, predicted_labels)
#
st.markdown("<h3 style='margin-top:0px;text-align: left; color: #4280FF;'>Ensemble Model Metrics:</h3>", unsafe_allow_html=True)
f1, precision, recall, roc, accuracy, fpr = st.columns(6)
f1.metric(label="F1", value=round(metrics_ensemble["F1"],2))
precision.metric(label="Precision", value=round(metrics_ensemble["Precision"],2))
recall.metric(label="Recall", value=round(metrics_ensemble["Recall"],2))
roc.metric(label="ROC_AUC", value=round(metrics_ensemble["ROC_AUC"],2))
accuracy.metric(label="Accuracy", value=round(metrics_ensemble["Accuracy"],2))
        # tpr.metric(label="True Positive Rate", value=round(metrics["True Positive Rate"],2), delta= str(round(metrics["True Positive Rate"] - benchmark_metrics[model]["True Positive Rate"],2)))
fpr.metric(label="False Positive Rate", value=round(metrics_ensemble["False Positive Rate"],2))
st.write("---")
st.markdown("<h3 style='margin-top:0px;text-align: left; color: #4280FF;'>Comparison with other models:</h3>", unsafe_allow_html=True)

for i in ["Suhuai Luo's Alzheimer's Detection Model", 'VGG-16']:
    
    predicted_labels = df[i]
    metrics = classification_metrics(true_labels, predicted_labels)
    st.subheader(f"{i} :")
    f1, precision, recall, roc, accuracy, fpr = st.columns(6)
    f1.metric(label="F1", value=round(metrics["F1"],2), delta= str(round(metrics["F1"] - metrics_ensemble["F1"],2)))
    precision.metric(label="Precision", value=round(metrics["Precision"],2), delta= str(round(metrics["Precision"] - metrics_ensemble["Precision"],2)))
    recall.metric(label="Recall", value=round(metrics["Recall"],2), delta= str(round(metrics["Recall"] - metrics_ensemble["Recall"],2)))
    roc.metric(label="ROC_AUC", value=round(metrics["ROC_AUC"],2), delta= str(round(metrics["ROC_AUC"] - metrics_ensemble["ROC_AUC"],2)))
    accuracy.metric(label="Accuracy", value=round(metrics["Accuracy"],2), delta= str(round(metrics["Accuracy"] - metrics_ensemble["Accuracy"],2)))
        # tpr.metric(label="True Positive Rate", value=round(metrics["True Positive Rate"],2), delta= str(round(metrics["True Positive Rate"] - benchmark_metrics[model]["True Positive Rate"],2)))
    fpr.metric(label="False Positive Rate", value=round(metrics["False Positive Rate"],2), delta= str(round(metrics_ensemble["False Positive Rate"] - metrics["False Positive Rate"], 2)))

st.write("---")
st.markdown("<h3 style='margin-top:0px;text-align: left; color: #4280FF;'>Confusion Matrix:</h3>", unsafe_allow_html=True)
ens, luo, vgg = st.columns(3)
with ens:
    predicted_labels = df['y_pred_ensemble']
    figure_conf = plot_confusion_matrix(true_labels,predicted_labels)
    figure_conf.update_layout(
                title=f'Our Ensemble Model:',
                xaxis_title='Predicted Class',
                yaxis_title='True Class',
                            # title_font={"size": 20},
                xaxis_title_font={"size":16, "color":"white"},
                yaxis_title_font={"size":16, "color":"white"},
                width=400,
                height=400)
    figure_conf.update_xaxes(tickfont={"size":14, "color":"white"})
    figure_conf.update_yaxes(tickfont={"size":14, "color":"white"})
    st.plotly_chart(figure_conf)

with luo:
    predicted_labels = df["Suhuai Luo's Alzheimer's Detection Model"]
    figure_conf = plot_confusion_matrix(true_labels,predicted_labels)
    figure_conf.update_layout(
                title=f"Luo's Model:",
                xaxis_title='Predicted Class',
                yaxis_title='True Class',
                            # title_font={"size": 20},
                xaxis_title_font={"size":16, "color":"white"},
                yaxis_title_font={"size":16, "color":"white"},
                width=400,
                height=400)
    figure_conf.update_xaxes(tickfont={"size":14, "color":"white"})
    figure_conf.update_yaxes(tickfont={"size":14, "color":"white"})
    st.plotly_chart(figure_conf)

with vgg:
    predicted_labels = df['VGG-16']
    figure_conf = plot_confusion_matrix(true_labels,predicted_labels)
    figure_conf.update_layout(
                title=f'VGG-16:',
                xaxis_title='Predicted Class',
                yaxis_title='True Class',
                            # title_font={"size": 20},
                xaxis_title_font={"size":16, "color":"white"},
                yaxis_title_font={"size":16, "color":"white"},
                width=400,
                height=400)
    figure_conf.update_xaxes(tickfont={"size":14, "color":"white"})
    figure_conf.update_yaxes(tickfont={"size":14, "color":"white"})
    st.plotly_chart(figure_conf)
# for i in [,'y_pred_luo', 'y_pred_vgg']:
    
#     st.write("---")
metrics = classification_metrics(true_labels, predicted_labels)


