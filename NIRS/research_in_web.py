import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import matplotlib.pyplot as plt

@st.cache
def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('data/Admission_Predict_Ver1.1.txt', sep=",", nrows=500)
    return data

@st.cache
def preprocess_data(data_in):
    '''
    Масштабирование признаков, функция возвращает X и y для кросс-валидации
    '''
    data_out = data_in.copy()
    scale_cols = ['GRE Score', 'TOEFL Score','University Rating', 'Research', 'SOP', 'CGPA']
    new_cols = []
    sc1 = MinMaxScaler()
    sc1_data = sc1.fit_transform(data_out[scale_cols])
    for i in range(len(scale_cols)):
        col = scale_cols[i]
        new_col_name = col + '_scaled'
        new_cols.append(new_col_name)
        data_out[new_col_name] = sc1_data[:,i]
    return data_out[new_cols], data_out['Chance of Admit ']

st.sidebar.header('Метод ближайших соседей')
data = load_data()

# Слайдеры для выбора количества фолдов, шага для соседей и алгоритма
cv_slider = st.sidebar.slider('Количество фолдов:', min_value=3, max_value=10, value=3, step=1)

step_slider = st.sidebar.slider('Шаг для соседей:', min_value=1, max_value=50, value=10, step=1)

algorithm_options = {'auto', 'ball_tree', 'kd_tree', 'brute'}
selected_algorithm = st.sidebar.selectbox('Выбор алгоритма:', options=list(algorithm_options), index=0)

weight_options = {'uniform', 'distance'}
selected_weight = st.sidebar.selectbox('Веса:', options=list(weight_options), index=0)

p = st.sidebar.slider('Параметр мощности для метрики Минковского:', min_value=1, max_value=2, value=1, step=1)

# Подготовка данных
data_len = data.shape[0]
rows_in_one_fold = int(data_len / cv_slider)
allowed_knn = int(rows_in_one_fold * (cv_slider-1))
st.write('Количество строк в наборе данных - {}'.format(data_len))
st.write('Максимальное допустимое количество ближайших соседей с учетом выбранного количества фолдов - {}'.format(allowed_knn))

n_range_list = list(range(1, allowed_knn, step_slider))
n_range = np.array(n_range_list)
tuned_parameters = [{'n_neighbors': n_range.tolist()}]

data_X, data_y = preprocess_data(data)

clf_gs = GridSearchCV(KNeighborsRegressor(p=p, weights=selected_weight, algorithm=selected_algorithm), tuned_parameters, cv=cv_slider, scoring='neg_mean_squared_error')
clf_gs.fit(data_X, data_y)

st.subheader('Оценка качества модели')
st.write('Лучшее значение параметров - {}'.format(clf_gs.best_params_))

fig1 = plt.figure(figsize=(7,5))
ax = plt.plot(n_range, clf_gs.cv_results_['mean_test_score'])
st.pyplot(fig1)
