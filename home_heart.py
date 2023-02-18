from utils import *


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
