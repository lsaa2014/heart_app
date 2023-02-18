import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from PIL import Image
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, auc, f1_score
from xgboost import XGBClassifier, plot_importance, plot_tree
from sklearn.model_selection import train_test_split

def load_data(DATA_URL):
    data = pd.read_csv(DATA_URL)
    return data

def plotly_table(data):
    fig = go.Figure(data=[go.Table(columnwidth = [150, 50],
        header=dict(values=list(data.columns),
                    fill_color='royalblue',
                    align='left',
                    font=dict(color='white', size=14)),
        cells=dict(values=[data.Labels, data.Probability],
                   fill_color='lavender',
                   align='left'))])
    return fig

def plotly_prediction_piechart(predictions_df, showlegend=True, size=250):
    """Return piechart with predict_proba distributions for ClassifierExplainer
    Args:
        predictions_df (pd.DataFrame): generated with 
            ClassifierExplainer.prediction_summary_df(index)
        showlegend (bool, optional): Show the legend. Defaults to True.
        size (int): width and height of the plot
    Returns:
        plotly.Fig
    """

    data = [
        go.Pie(labels=predictions_df.label.values, 
                values=predictions_df.probability.values, 
                hole=0.3,
                sort=False)
    ]
    layout = dict(autosize=False, width=size, height=size, 
                    margin=dict(l=20, r=20, b=20, t=30, pad=4),
                    showlegend=showlegend)
    fig = go.Figure(data, layout)
    return fig

def select_classifier(selection, name=True):
    clfs = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(random_state=1, max_depth = 3, n_estimators = 30),
    'Gradient Boosting': GradientBoostingClassifier(max_depth = 2, n_estimators = 30),
    'XGBoost': XGBClassifier(max_depth = 3, n_estimators = 50),
     }

    return clfs[selection]


def error_metrics(y_true, y_pred):
    '''Metrics to use to validate the model'''
    results = {
    'Accuracy':round(accuracy_score(y_true, y_pred), 3), 
    'AUC':round(roc_auc_score(y_true, y_pred), 3),
    'Fscore':round(f1_score(y_true,y_pred), 3)
    }
    return results


def classif_model(classifier, data, target, features, testing = None):
    ''' Calculate metrics on test set from different classifiers'''
    ## Split the training
    X_train, X_val, y_train, y_val = \
    train_test_split(data[features], target,\
                     test_size=0.40, random_state = 22)

    print("Start training model!")
   
    clf = select_classifier(classifier)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    y_pred_proba = clf.predict_proba(X_val)
    metrics = error_metrics(y_val, y_pred)
    res = {metric:value for (metric, value) in metrics.items()}
    res['Model'] = clf.__class__.__name__
    result_metrics = pd.DataFrame([res])
    result_metrics = result_metrics.set_index('Model')

    ## table result
    res = [f'{val:.2%}' for val in y_pred_proba[:,1]]
    res_df = pd.DataFrame(list(zip(y_pred, res)), columns = ['Labels', 'Probability'])
    # st.write(res_df)
    
    if testing is not None:
        y_pred_test = clf.predict(testing)
        y_pred_proba = clf.predict_proba(testing)
        # st.write(y_pred_test)
        # st.write(y_pred_proba[:,1][0])
        st.write(f'Your prediction is {y_pred_test[0]} with {y_pred_proba[:,1][0]:.2%} of probability')	

    return result_metrics, res_df    

def write():
    """Writes content to the app"""
    df = load_data('heart_failure_clinical_records_dataset.csv')


    st.title("Predicting Heart Failure")
    st.markdown("Select a classifier")


    classifier = st.selectbox("Select a classifier Algorithm",   
                                     ['Logistic Regression', 
                                      'Random Forest', 'Gradient Boosting',
                                       'XGBoost'])                    
    st.subheader(f'''Prediction with {classifier}''')

    features = ['ejection_fraction','serum_creatinine', 'serum_sodium', 'time']
    col1, col2 = st.columns(2)

    time = col1.number_input('Follow-up period', min_value = df['time'].min(), value = 8, max_value = df['time'].max())
    ef = col1.number_input('Percentage of blood leaving', min_value = df['ejection_fraction'].min(), value = 38, max_value = df['ejection_fraction'].max())
    sc = col2.number_input("Level of creatinine in the blood (mg/dL)", min_value = df['serum_creatinine'].min(), value = 2.7, max_value = df['serum_creatinine'].max())
    ss = col2.number_input("Level of sodium in the blood (mEq/L)", min_value = df['serum_sodium'].min(), value = 116, max_value = df['serum_sodium'].max())

    new_df = pd.DataFrame([[ef, sc, ss, time]], columns = features)
    # st.write(new_df)

    if st.button('Predict'):
    	#st.subheader(f'''Prediction on a new test set''')    
    	st.subheader(f"**Result on your data**")
    	_, df_class = classif_model(classifier, df, df['DEATH_EVENT'], features, testing = new_df)

    if st.checkbox("Validation set results", False):
    	st.subheader(f'''Prediction on the validation set''')

    	metric_results, df_class = classif_model(classifier, df, df['DEATH_EVENT'], features, testing = None)
    		
    	selectidx = st.selectbox('Select an index', df.index.tolist())

    	st.write(f"The predicted label is {df_class.loc[selectidx, 'Labels']} \
    		  		      with {df_class.loc[selectidx, 'Probability']} of probability")
    		#st.table(df_class.loc[selectidx, :])

    	#show results
    	st.write(f"The metrics for on the validation set is")
    	st.table(metric_results)



if __name__ == "__main__":
    write()		




