import pandas as pd
import numpy as np
import streamlit as st
from flask import redirect, jsonify
import json, urllib3, time


MLFLOW_URI = 'http://127.0.0.1:5001/invocations'
path_server = 'http://127.0.0.1:5000/'

@st.cache_data
def load_data():
    http = urllib3.PoolManager()
    r = http.request('GET',path_server+'load-agg-data')
    resData = json.loads(r.data)
    # return resData['data'], resData['data_agg']
    return resData['data']

chosen_radio = st.sidebar.radio(
'Comment souhaitez vous entrer les valeurs du client?',("Index", "Valeur"))

if chosen_radio == "Index":

    # save chosen_radio
    st.session_state['chosen_radio'] = "Index"

    #Chargement des données
    # identifiant, data_agg = load_data()
    identifiant = load_data()
    identifiant = pd.DataFrame(identifiant)
    # data_agg = pd.DataFrame(data_agg)

    if 'data' not in st.session_state:
        st.session_state['data'] = identifiant

    latest_iteration = st.empty()
    latest_iteration.markdown("## :blue[Vous identifierez le client grâce à son numéro d'identification !]")
    
    #time.sleep(3)
    
    latest_iteration.markdown("## :blue[Choix de l'identifiant]")
    chosen_customer = st.selectbox(
    "Quel est le numéro d'identifiant du client?", identifiant['SK_ID_CURR'])

    if 'chosen_customer' not in st.session_state:
        st.session_state['chosen_customer'] = chosen_customer


    # tab_comparaison = data_agg[['info',	'Bon', 'Mauvais']].merge(
    #     identifiant[identifiant.SK_ID_CURR==chosen_customer].T\
    #     .reset_index().rename(columns = {'index':'info', 0:'Client selectionné'}),
    #     on = 'info',how='inner'
    #             )
    
    #st.table(tab_comparaison)

    # Bouton Score
    predict_btn = st.sidebar.button('Obtenir le score')

    if predict_btn:
        http = urllib3.PoolManager()
        r = http.request('GET', path_server+'predict/index'+'?idClient='+str(chosen_customer) )
        if r.status != 200:
            st.write('Calcul du Score non disponible')
        elif json.loads(r.data)['proba'] != None:
            resData = json.loads(r.data)
            if resData['proba'] <0.5:
                st.balloons()
                st.markdown("## :green[Prédit comme bon client]")
                st.markdown("La probabilité du client d'être un client à risque est de {:.2%}".format(resData['proba'])  )
                st.write(resData['decision'])
            else:
                st.markdown("## :red[Prédit comme client à risque]")
                st.markdown("La probabilité du client d'être un client à risque est de {:.2%}".format(resData['proba'])  )
                st.write(resData['decision'])
                st.snow()
        else:
            st.write('La base de données recus est vide')
    
