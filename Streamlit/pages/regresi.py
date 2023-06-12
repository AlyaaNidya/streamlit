import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import pearsonr #untuk memasukkan fungsi korelasi pearson
import matplotlib.pyplot as plt #untuk memasukkan scatterplot
from sklearn import linear_model, metrics
from sklearn.datasets import * #untuk memasukkan sample dataset yang sudah tersedia di sklearn
from sklearn.metrics import r2_score #untuk memasukkan R^2
import Glejser_test #untuk melakukan uji residual identik
from Heteroscedasticity_test_general import Heteroscedasticity_tests
from statsmodels.stats.stattools import durbin_watson as dwtest #uji residual independen
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import ks_2samp #uji residual berdistribusi normal

def calc_corr(var1, var2): #mendefinisikan fungsi korelasi
    corr, p_val = pearsonr(var1, var2)
    return corr, p_val

def plot_regression_line(x, y, b, indep_var, dep_var):
    fig = plt.figure()
    #melakukan plotting pada scatter plot
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30)
    #membuat model regresi
    y_pred = b[0] + b[1]*x
    #membuat garis regresi
    plt.plot(x, y_pred, color = "g") 
    #membuat label
    plt.xlabel(indep_var)
    plt.ylabel(dep_var) 
    st.write(fig) #fungsi untuk menampilkan plot

def reg_analysis(var1, var2):
    var1 = var1.to_numpy().reshape(-1, 1) 
    var2 = var2.to_numpy().reshape(-1, 1) 
    reg = linear_model.LinearRegression()
    #melatih model
    reg.fit(var1, var2)
    rsq = reg.score(var1, var2)
    #koefisien regresi
    return [reg.intercept_[0], reg.coef_[0][0]], rsq


def get_data(data_choice):
    if data_choice == 'Age and BMI':
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df = df[['age','bmi']]
        X_var = df.age
        Y_var = df.bmi
    elif data_choice == 'Age and Blood Pressure':
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df = df[['age','bp']]
        X_var = df.age
        Y_var = df.bp
    elif data_choice == 'Petal length and Sepal length':
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df = df[['petal length (cm)','sepal length (cm)']]
        df.columns = ['petallength','sepallength']   
        X_var = df.petallength 
        Y_var = df.sepallength
    elif data_choice == 'Sepal width and Sepal length':
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df = df[['sepal width (cm)','sepal length (cm)']]
        df.columns = ['sepalwidth','sepallength']   
        X_var = df.sepalwidth 
        Y_var = df.sepallength
    elif data_choice == 'Median income and Average room count':
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df = df[['MedInc','AveRooms']]
        df.columns = ['medianincome','averageroomcount']   
        X_var = df.medianincome 
        Y_var = df.averageroomcount
    return df, X_var, Y_var

def get_dataset_details(data_choice):
    if data_choice == 'Age and BMI':
        data_used = 'data load_diabetes dari sklearn.datasets'
    elif data_choice == 'Age and Blood Pressure':
        data_used = 'data load_diabetes dari sklearn.datasets' 
    elif data_choice == 'Petal length and Sepal length':
        data_used = 'data load_iris darisklearn.datasets'
    elif data_choice == 'Sepal width and Sepal length':
        data_used = 'data load_iris darisklearn.datasets'
    elif data_choice == 'Median income and Average room count':
        data_used = 'data fetch_california_housing dari sklearn.datasets'
    return data_used

st.title("Correlation and Regression Analysis") #membuat judul

#membuat form
my_form_3 = st.form(key = "form3")
#membuat select box
user_input = my_form_3.selectbox('Data',('None', 'Age and BMI', 'Age and Blood Pressure', 'Petal length and Sepal length', 
                                         'Sepal width and Sepal length', 'Median income and Average room count'))
submit = my_form_3.form_submit_button(label = "Submit")

if user_input == 'None':
    st.write('tidak ada data yang terpilih')

if user_input != 'None':
    dataset_used = get_dataset_details(user_input)
    st.write('Dataset yang digunakan: ', dataset_used)
    df, X, Y = get_data(user_input)
    indep_var = df.columns[0]
    dep_var = df.columns[1]
    correlation, corr_p_val = calc_corr(X,Y)
    st.write('korelasi pearson: %.3f' % correlation)
    st.write('p value: %.3f' % corr_p_val)
    params, rsq = reg_analysis(X,Y)
    plot_regression_line(X,Y,params,indep_var, dep_var)
    md_results1 = f"model regresinya adalah **y** **=** **{params[0]:.2f}** **+** **{params[1]:.2f}** * x."
    st.markdown(md_results1)
    md_results2 = f"R^2 memiliki nilai sebesar **=** **{rsq:.2f}**."
    st.markdown(md_results2)
    md_results3 = f'Uji residual identiknya adalah **=** **{Glejser_test.Glejser_test(X, Y).glejser_test()}**.'
    st.markdown(md_results3)
    md_results4 = f'Uji residual independen adalah **=** **{dwtest(resids=np.array(params))}**.'
    st.markdown(md_results4)
    md_results5 = f'Uji residual berdistribusi normal:'
    md_results6 = f'**{ks_2samp(X, Y)}**.'
    st.markdown(md_results5)
    st.markdown(md_results6)




    #YAALLAH PUSING :( 