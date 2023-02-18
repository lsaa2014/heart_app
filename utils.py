import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, auc, f1_score
from xgboost import XGBClassifier, plot_importance, plot_tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score


def load_data(DATA_URL):
    data = pd.read_csv(DATA_URL)
    return data


def load_data(DATA_URL):
    data = pd.read_csv(DATA_URL)
    return data

def plotly_importances_plot(importance_df, descriptions=None, round=3, 
            target="target" , units="", title=None, xaxis_title=None):
    """Return feature importance plot
    Args:
        importance_df (pd.DataFrame): generate with get_importance_df(...)
        descriptions (dict, optional): dict of descriptions of each feature. 
        round (int, optional): Rounding to apply to floats. Defaults to 3.
        target (str, optional): Name of target variable. Defaults to "target".
        units (str, optional): Units of target variable. Defaults to "".
        title (str, optional): Title for graph. Defaults to None.
        xaxis_title (str, optional): Title for x-axis Defaults to None.
    Returns:
        Plotly fig
    """
    
    importance_name = importance_df.columns[1] # can be "MEAN_ABS_SHAP", "Permutation Importance", etc
    if title is None:
        title = importance_name
    longest_feature_name = importance_df['Feature'].str.len().max()

    imp = importance_df.sort_values(importance_name)

    feature_names = [str(len(imp)-i)+". "+col 
            for i, col in enumerate(imp.iloc[:, 0].astype(str).values.tolist())]

    importance_values = imp.iloc[:,1]

    data = [go.Bar(
                y=feature_names,
                x=importance_values,
                #text=importance_values.round(round),
                text=descriptions[::-1] if descriptions is not None else None, #don't know why, but order needs to be reversed
                #textposition='inside',
                #insidetextanchor='end',
                hoverinfo="text",
                orientation='h')]

    layout = go.Layout(
        title=title,
        plot_bgcolor = '#fff',
        showlegend=False
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_yaxes(automargin=True)
    if xaxis_title is None:
        xaxis_title = units
    fig.update_xaxes(automargin=True, title=xaxis_title)

    left_margin = longest_feature_name*7
    if np.isnan(left_margin):
        left_margin = 100

    fig.update_layout(height=200+len(importance_df)*20,
                      margin=go.layout.Margin(
                                l=left_margin,
                                r=40,
                                b=40,
                                t=40,
                                pad=4
                            ))
    return fig 

def scoring_cv(model, X_train, y_train, n_folds, scoring):
    '''Calculate a metric with cross validation'''
    kf = KFold(n_folds, shuffle = True, random_state = 22)
    cv_scores = cross_val_score(model, X_train, y_train, scoring = scoring, cv = kf)
    return(cv_scores)

def modelfit_cv(clf, train, features, target, n_folds, performCV=True, printFeatureImportance=True):

    #split the data
    X_train, X_test, y_train, y_test = train_test_split(train[features], \
                                            train[target], test_size=0.20, random_state = 22)

    ## Fit the training
    clf.fit(X_train, y_train)
        
    #Predict testing set:
    pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, pred)

    #Perform cross-validation:
    if performCV:
        auc_cv = scoring_cv(clf, X_train, y_train, n_folds, scoring="roc_auc")

    #Print model report:
    print(f"\nModel Report with {clf.__class__.__name__}: ")
    print('AUC is {}'.format(auc))
    print('Accuracy is {}'.format(accuracy_score(y_test, pred)))
    print('AUC score error with cv : Mean - {:.4f} | Stdf - {:.4f}'.format(auc_cv.mean(), auc_cv.std()))

    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(clf.feature_importances_, X_train.columns).sort_values(ascending=True)
        colors = ['lightslategray',] * 12
        colors[11] = 'indianred'
        colors[10] = 'indianred'
        colors[9] = 'indianred'
        colors[8] = 'indianred'
        fig = px.bar(feat_imp, x = feat_imp.values, y = feat_imp.index, orientation = 'h', title = 'Feature Importances', width=550)
        fig.update_traces(marker_color = colors)
        fig.update_layout(xaxis_title="", yaxis_title="", plot_bgcolor = '#fff')
        fig.update_yaxes(tickfont=dict(color='black', size=8))
        st.write(fig)


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
   
    if testing is not None:
        y_pred_test = clf.predict(testing)
        y_pred_proba = clf.predict_proba(testing)
        st.write(f'Your prediction is {y_pred_test[0]} with {y_pred_proba[:,1][0]:.2%} of probability')	

    return result_metrics, res_df    
      
        
