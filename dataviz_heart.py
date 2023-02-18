from utils import *

df = load_data('heart_failure_clinical_records_dataset.csv')


def write():
    """Writes content to the app"""
    st.title("Data visualization heart failure")
    st.subheader('Univariate')
    #st.table(df.describe())
    cat_var = ['anaemia', 'high_blood_pressure', 'sex', 'smoking', 'diabetes', 'DEATH_EVENT']
    num_var = [col for col in df.columns if col not in cat_var]

    select = st.selectbox('Select variable', num_var)

    uniplot = st.radio("Select your plot",
    	('histogram', 'boxplot', 'barplot'))

    if uniplot == 'histogram':
    	fig = px.histogram(df, x=select, title = f'Distribution of {select}')
    	st.write(fig)
    elif uniplot == 'boxplot':
    	fig = px.box(df, y=select, notched=True, title = f'Boxplot of {select}')
    	st.write(fig)	
    else:
    	select = st.selectbox('Select variable', cat_var)
    	fig = px.histogram(df, x = select, barmode='group', title = f'Count of {select}')
    	fig.update_layout(
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.05 # gap between bars of the same location coordinate.
        )
    	st.write(fig)


    st.subheader('Bivariate')
    biplot = st.radio("Select your plot",
    	('histogram', 'boxplot', 'scatterplot'))

    if biplot == 'histogram':
    	fig = px.histogram(df, x = select, color = 'DEATH_EVENT', title = f'Distribution of {select} by death event')
    	st.write(fig)

    elif biplot == 'boxplot':
        fig = px.box(df, x = select, color = 'DEATH_EVENT', notched=True, title = f'Boxplot of {select} by death event')
        st.write(fig)	

    else:
        df_new = df.copy()
        df_new['DEATH_EVENT'] = df_new['DEATH_EVENT'].astype('category')
        select1 = st.selectbox('Select the 2nd variable', num_var)
        fig = px.scatter(df_new, x = select, y = select1, color = 'DEATH_EVENT', title = f'{select} Vs. {select1} by death event')
        st.write(fig)

    st.subheader('Multivariate (Correlation plot)')
    cor = df[num_var].corr()
    fig = px.imshow(cor, color_continuous_scale='ice_r', width=690, height=600)
    st.write(fig)


if __name__ == "__main__":
    write()	
