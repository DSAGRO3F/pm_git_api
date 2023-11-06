import flask
from flask import Flask, jsonify, request, render_template, redirect, url_for, app, redirect
import pandas as pd
import numpy as np
import pickle
import sklearn
import json
from datetime import datetime, date
from dateutil import parser


# Initialisation Flask
app = flask.Flask(__name__)
app.config["DEBUG"] = True


# Définition des chemins

#1. Chemin en local:
# path = '/Users/olivierdebeyssac/Python_project_predictive_maintenance/pm_git/pm_api/'

#2. Chemin sur git
path = '/'



# Chemin model en local
f_model_name = 'best_model.pkl'
# model_path = path + f_model_name

# Chemin model sur git
model_path = path + f_model_name


# Chemin data en local
f_df_name = 'df.pkl'
# df_path = path + f_df_name

# Chemin data sur git
df_path = path + f_df_name


# f_df_total_name = 'df_total.pkl'
# df_total_path = path + f_df_total_name
#
# f_X_total_name = 'X_total.pkl'
# X_total_path = path + f_X_total_name
#
# f_y_total_name = 'y_total.pkl'
# y_total_path = path + f_y_total_name


# Chargement des data
df_open = open(df_path, 'rb')
df = pickle.load(df_open)
df = df[0:730]

# df_total_open = open(df_total_path, 'rb')
# df_total = pickle.load(df_total_open)
#
# X_total_open = open(X_total_path, 'rb')
# X_total = pickle.load(X_total_open)
#
# y_total_open = open(y_total_path, 'rb')
# y_total = pickle.load(y_total_open)

# Construction fonction pour obtention dataframe df par app. dashbord
@app.route('/df', methods=['GET'])
def get_df():
       print('df: {}'.format(df[0:2]))
       print(df.columns)
       d_df = df.to_dict()
       jsonified = jsonify(d_df)
       # print(type(jsonified))
       # print('d_df: {}'.format(d_df['ID']))

       return jsonified




# Définition des features pour analyse temporelle des capteurs
sensor_feat = ['ID', 'DATE','S13', 'S15', 'S16', 'S17', 'S18', 'S19', 'S5', 'S8']

# Construction liste des Id équipements.
@app.route('/id', methods=['GET'])
def liste_id():
       ids = df.loc[:, 'ID'].values.tolist()
       unique_ids = np.unique(ids)
       unique_ids = unique_ids.tolist()
       return jsonify(unique_ids)


l_avg = []
l_min = []
l_max = []

df_basic = df.loc[:, ['ID', 'S13', 'S15', 'S16', 'S17', 'S18', 'S19', 'S5', 'S8']]
# print('df_info: {}'.format(df.info()))


# Construction valeurs spécifiques capteurs
@app.route('/sensors_data/<id>', methods=['GET'])
def get_sensors_data(id):
       # print('valeur_id: {}'.format(id))
       df['ID'] = df['ID'].astype('int')
       # print(df['ID'].dtypes)
       df_sensors_data = df[sensor_feat]
       # print(df_sensors_data['ID'].dtypes)
#      # print(df_sensors_data['DATE'].dtypes)

       # Sélection de l'équipement
       df_selected_eq = df_sensors_data[df_sensors_data['ID'] == int(id)]
       # print(df_selected_eq)


       # Pour chq capteur calculer moyenne, min, max, val. pic, rolling.mean()
       l_sensors = ['S13', 'S15', 'S16', 'S17', 'S18', 'S19', 'S5', 'S8']


       #4. Construction valeurs pic
       peak_features = ['peak_' + name for name in l_sensors]

       for i in range(len(peak_features)):
              feat_name = l_sensors[i]
              new_feat_name = peak_features[i]
              df_temp_1 = df_selected_eq.copy()
              avg_val = df_temp_1[feat_name].mean(axis=0)
              df_temp_1[new_feat_name] = df_temp_1[feat_name]/avg_val
              df_selected_eq = df_temp_1
       # print('df_selected_eq: {}'.format(df_selected_eq['peak_S13'][0:2]))


       #5. Rolling function:
       roll_features = ['roll_' + name for name in l_sensors]

       for i in range(len(roll_features)):
              feat_name = l_sensors[i]
              new_feat_name = roll_features[i]
              df_temp_1 = df_selected_eq.copy()
              df_temp_1[new_feat_name] = df_temp_1[feat_name].rolling(window=90).mean()
              df_selected_eq = df_temp_1

       # print('df_selected_eq: {}'.format(df_selected_eq.shape))

       # Conversion df en dict
       df_selected_eq = df_selected_eq.to_dict()

       return jsonify(df_selected_eq)


# Construction fonction qui renvoie X_test.
# Cette fonction

@app.route('/X_test/<id>', methods=['GET'])
def X_test_data(id, df=df):
       df['ID'] = df['ID'].astype('int')
       # print(df['ID'].dtypes)

       # Relevant features for building X_test
       training_features = ['ID',
                            'TIME_SINCE_START', 'TIME_TO_FAILURE', 'FAILURE_TGT',
                            'S15', 'S17', 'S13', 'S5', 'S16', 'S19', 'S18',
                            'S8', 'AGE_OF_EQUIPMENT', 'S15_mean_val', 'S15_med_val',
                            'S15_min_val', 'S15_max_val', 'S17_mean_val', 'S17_med_val',
                            'S17_min_val', 'S17_max_val', 'S13_mean_val', 'S13_med_val',
                            'S13_min_val', 'S13_max_val', 'S5_mean_val', 'S5_med_val',
                            'S5_min_val', 'S5_max_val', 'S16_mean_val', 'S16_med_val',
                            'S16_min_val', 'S16_max_val', 'S19_mean_val', 'S19_med_val',
                            'S19_min_val', 'S19_max_val', 'S18_mean_val', 'S18_med_val',
                            'S18_min_val', 'S18_max_val', 'S8_mean_val', 'S8_med_val',
                            'S8_min_val', 'S8_max_val', 'S15_peak', 'S17_peak', 'S13_peak',
                            'S5_peak', 'S16_peak', 'S19_peak', 'S18_peak', 'S8_peak']
       X_test = df[training_features]

       # Make predictions on X_Test.

       # 1. Load model.
       best_model = open(model_path, 'rb')
       best_model = pickle.load(best_model)

       # 2. Predictions.
       y_proba = best_model.predict_proba(X_test)
       y_pred = y_proba[:, 1]

       # 3. Concaténer y_pred à X_test.
       X_test['y_pred'] = y_pred

       # 4. Cut off
       cut_off = 0.5
       y_pred_cut_off = [0 if val < cut_off else 1 for val in y_pred]

       # 5. Concaténer y_pred_cut_off à X_test.
       X_test['y_pred_cutoff'] = y_pred_cut_off

       # 6. Concaténer 'DATE' et X_test
       X_test = X_test.sort_values(by=['ID'], ascending=True)
       df = df.sort_values(by=['ID'], ascending=True)
       dates = df['DATE'].to_list()
       X_test['DATE'] = dates
       X_test['DATE'] = X_test['DATE'].astype(str)
       #X_test['DATE'] = X_test['DATE'].apply(lambda x: datetime.strptime(x,"%a %b %d %Y %H:%M:%S %Z%z (IST)"))

       # 4. Sélectionner l'ensemble des prédictions pour l'équipement "id"
       X_test_selected_eq = X_test[X_test['ID'] == int(id)]

       # 5. Convertir en dictionnaire
       d_X_test = X_test_selected_eq.to_dict()

       # 6. Sérialisation.
       jsonified = jsonify(d_X_test)

       return jsonified





app.run()

# get_sensors_data()

# get_df()