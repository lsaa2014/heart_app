import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, auc, f1_score
from xgboost import XGBClassifier, plot_importance, plot_tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score

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

def write():
    df = load_data('heart_failure_clinical_records_dataset.csv')
    rf = RandomForestClassifier(random_state=1, max_depth = 3, n_estimators = 30)

    st.title("Predict heart failure using ML")
    st.subheader('Background')
    st.write('''Modeling survival for heart failure is still a challenge today,
        both in terms of reaching high prediction accuracy and 
        finding features that influence survival. Most of the models created for 
        this goal are only moderately accurate, and the predictive factors have limited interpretability.
        Recent models have shown improvements, particularly when the survival 
        outcome is combined with additional targets (for example, hospitalization). 
        Despite the fact that scientists have identified a large number 
        of predictors and indicators, there is no consensus on their 
        relative importance in predicting survival [1](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5#Sec8).
        \n The variables used are Age, Anaemia, High blood pressure, Creatinine phosphokinase, Diabetes, Ejection fraction,
        Sex, Platelets, Serum creatinine, Serum sodium, Smoking, Time and the target (death event).
        \n The authors find that Random forest is the best algorithm, which corroborates my results.
        For the final result, they used the best 2 features (ejection fraction, serum creatinine) based on the feature importance.
        Here are used the best 4 instead (ejection fraction, serum creatinine, serum sodium and time).
        You can find the feature importances plot using all features here below.''') 

    modelfit_cv(rf, df, [col for col in df.columns if col != 'DEATH_EVENT'], 'DEATH_EVENT', \
                5, performCV=True, printFeatureImportance=True)        

    st.text('''References  
           1. Chicco, Davide, and Giuseppe Jurman. "Machine learning can predict survival of patients 
             with heart failure from serum creatinine and ejection fraction alone.
             BMC medical informatics and decision making 20.1 (2020): 1-16.''')

if __name__ == "__main__":
    write() 