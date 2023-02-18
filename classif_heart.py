from utils import *


def write():
    """Writes content to the app"""
    df = load_data('heart_failure_clinical_records_dataset.csv')


    st.title("Predicting Heart Failure")
    st.markdown("Select a classifier")


    classifier = st.selectbox("Select a classifier Algorithm",   
                                     ['Logistic Regression', 
                                      'Random Forest',
                                       'XGBoost'])                    
    st.subheader(f'''Prediction with {classifier}''')

    features = ['ejection_fraction','serum_creatinine', 'serum_sodium', 'time']
    col1, col2 = st.beta_columns(2)

    time = col1.number_input('Follow-up period', min_value = df['time'].min(), value = 8, max_value = df['time'].max())
    ef = col1.number_input('Percentage of blood leaving', min_value = df['ejection_fraction'].min(), value = 38, max_value = df['ejection_fraction'].max())
    sc = col2.number_input("Level of creatinine in the blood (mg/dL)", min_value = df['serum_creatinine'].min(), value = 2.7, max_value = df['serum_creatinine'].max())
    ss = col2.number_input("Level of sodium in the blood (mEq/L)", min_value = df['serum_sodium'].min(), value = 116, max_value = df['serum_sodium'].max())

    new_df = pd.DataFrame([[ef, sc, ss, time]], columns = features)

    if st.button('Predict'): 
    	st.subheader(f"**Result on your data**")
    	_, df_class = classif_model(classifier, df, df['DEATH_EVENT'], features, testing = new_df)

    if st.checkbox("Validation set results", False):
    	st.subheader(f'''Prediction on the validation set''')

    	metric_results, df_class = classif_model(classifier, df, df['DEATH_EVENT'], features, testing = None)
    		
    	selectidx = st.selectbox('Select an index', df.index.tolist())

    	st.write(f"The predicted label is {df_class.loc[selectidx, 'Labels']} \
    		  		      with {df_class.loc[selectidx, 'Probability']} of probability")

    	#show results
    	st.write(f"The metrics for on the validation set is")
    	st.table(metric_results)



if __name__ == "__main__":
    write()		




