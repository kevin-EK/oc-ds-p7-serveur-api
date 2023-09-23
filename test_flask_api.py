import json
import joblib
import pandas as pd
import pytest
from flask import jsonify
from flask_api import app
import important_features

@pytest.fixture # indiq que la fct ci-dessous est 1 fixture ie elle fournit 1 objet/ressource à d'autres fct de test
#Les fixtures sont utilisées pour configurer l'état initial requis pour les tests.
def client():
    client = app.test_client()
    yield client


def test_helloworld(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome to my api!" in response.data # ou b"Welcome to my api!" == response.data.decode()

def test_get_agg_data(client):
    response = client.get('/load-agg-data')
    assert response.status_code == 200
    # Vous pouvez ajouter d'autres assertions en fonction de la réponse JSON que vous attendez.

def test_get_data(client):
    response = client.get('/load_data')
    assert response.status_code == 200
    # Vous pouvez ajouter d'autres assertions en fonction de la réponse JSON que vous attendez.

def test_index_predict(client):
    response = client.get('/predict/index?idClient=100002')
    assert response.status_code == 200
    # Vous pouvez ajouter d'autres assertions en fonction de la réponse JSON que vous attendez.
