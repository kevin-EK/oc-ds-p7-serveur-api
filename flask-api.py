from flask import Flask,  redirect, url_for ,request, jsonify
import urllib3
import pandas as pd
import joblib
import json
import certifi

app = Flask(__name__)

classify_mappings = {0: 'Bon client', 1: 'Client à risque'}
path_server = 'https://oc-ds-p7-kevin-el-ce8c86036717.herokuapp.com'

try:
    model = joblib.load("support/models/model.sav") # Load "model.pkl"
    print ('Model loaded')
except Exception as e:
        print(f"Une erreur s'est produite : {e}")

important_features = [
'NAME_INCOME_TYPE', 'AMT_REQ_CREDIT_BUREAU_MON', 'NAME_EDUCATION_TYPE',
'FONDKAPREMONT_MODE', 'OCCUPATION_TYPE', 'FLAG_OWN_CAR',
'REG_CITY_NOT_WORK_CITY', 'REG_REGION_NOT_LIVE_REGION', 'REGION_POPULATION_RELATIVE',
'AMT_REQ_CREDIT_BUREAU_WEEK', 'REG_REGION_NOT_WORK_REGION', 'AMT_ANNUITY',
'DAYS_REGISTRATION', 'REGION_RATING_CLIENT_W_CITY', 'FLAG_DOCUMENT_5',
'NAME_TYPE_SUITE', 'AMT_INCOME_TOTAL', 'FLAG_OWN_REALTY',
'FLAG_DOCUMENT_13', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'DAYS_ID_PUBLISH',
'AMT_REQ_CREDIT_BUREAU_HOUR', 'FLAG_WORK_PHONE', 'DAYS_BIRTH',
'DEF_30_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_3', 'EXT_SOURCE_2',
'DAYS_LAST_PHONE_CHANGE', 'FLAG_EMAIL', 'FLAG_DOCUMENT_9',
'CODE_GENDER', 'DAYS_EMPLOYED', 'REG_CITY_NOT_LIVE_CITY',
'AMT_GOODS_PRICE', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_11',
'OBS_30_CNT_SOCIAL_CIRCLE', 'FLAG_PHONE', 'FLAG_DOCUMENT_6'
]

@app.route("/")
def helloworld():
    return "<h1>Welcome to my api!</h1>"

@app.route("/testmodel")
def loadmodel():
    return model[-1].__class__.__name__

@app.route('/load-agg-data')
def get_agg_data():
    # amelioration création base de données
    try:                
        # load data
        data1 = joblib.load("support/data/application_train.sav")
        data1 = data1[important_features+['TARGET','SK_ID_CURR']]
        data2 = joblib.load("support/data/application_test.sav")
        data2 = data2[important_features+['SK_ID_CURR']]
    
        # feature engenering
        data = pd.concat([data1, data2],axis=0).drop_duplicates()
        data['ANNUITY_INCOME_PERC'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
        data = data[data['CODE_GENDER'] != 'XNA']
        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            data[bin_feature], uniques = pd.factorize(data[bin_feature])
        ## return jsonify( {'data':data.to_dict('records'), 'data_agg':data_agg.to_dict('records')} ) 
        return jsonify( {'data':data.to_dict('records')} ) 
    except Exception as e:
        return f"Aggregation data NOT available! \n\n Une erreur s'est produite : {e}"
    # http://127.0.0.1:5000/load-agg-data
    # https://oc-ds-p7-kevin-el-ce8c86036717.herokuapp.com/load-agg-data
    

@app.route('/load_data')
def get_data():
    # amelioration création base de données
    try:
        # load data
        df_train = joblib.load("support/data/application_train.sav")
        df_test = joblib.load("support/data/application_test.sav")
        list_index = list(set(df_train.SK_ID_CURR.to_list() + df_test.SK_ID_CURR.to_list())) # a tester
        #columns_to_keep = joblib.load("support/data/list_col_to_keep.joblib")
        columns_to_keep = important_features

        # concatened dataframe
        data = pd.concat([df_train,df_test], axis = 0)

        # Filtered columns
        data = data[['SK_ID_CURR','TARGET']+columns_to_keep]
        
        #feature ingenering
        data = data[data['CODE_GENDER'] != 'XNA']
        data['ANNUITY_INCOME_PERC'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            data[bin_feature], uniques = pd.factorize(data[bin_feature])
                
        # pour recuperer les resultats use open url
        return jsonify({'index':list_index, 'data':data.to_dict('records')})
    except Exception as e:
        return f"Data NOT available! \n\n Une erreur s'est produite : {e}"
    # http://127.0.0.1:5000/load_data
    # https://oc-ds-p7-kevin-el-ce8c86036717.herokuapp.com/load_data
    

@app.route('/predict/index',methods=['GET'])
def index_predict():
    # get index
    idClient = int(request.args.get('idClient'))
    # get data
    try:
        http = urllib3.PoolManager(ca_certs=certifi.where(),
                                   retries=False)
        r = http.request('GET',path_server+'/load_data')
        resData = json.loads(r.data)
    except Exception as e:
        return f"loading api not available \n\n Une erreur s'est produite : {e}"

    if idClient in resData['index']:
        # filtered data
        valideData = pd.DataFrame.from_dict(resData['data'])
        valideData = valideData.loc[valideData.SK_ID_CURR == idClient,:].drop(columns = ['SK_ID_CURR','TARGET'])
        return redirect(url_for('values_predict',values = valideData.to_json(orient="columns") ))
    else:
        return 'Id non reconnu'
    # http://127.0.0.1:5000/predict/index?idClient=274986
    # https://oc-ds-p7-kevin-el-ce8c86036717.herokuapp.com/predict/index?idClient=100002



#### test
@app.route('/predict/index_test',methods=['GET'])
def index_predict_test():
    # get index
    idClient = int(request.args.get('idClient'))
    # get data
    try:
        http = urllib3.PoolManager(ca_certs=certifi.where(),
                                   retries=False)
        r = http.request('GET',path_server+'/load_data')
        resData = json.loads(r.data)
    except Exception as e:
        return f"loading api not available \n\n Une erreur s'est produite : {e}"

    if idClient in resData['index']:
        # filtered data
        valideData = pd.DataFrame.from_dict(resData['data'])
        valideData = valideData.loc[valideData.SK_ID_CURR == idClient,:].drop(columns = ['SK_ID_CURR','TARGET'])
        return jsonify( {'data':valideData.to_json(orient="columns")})
    else:
        return 'Id non reconnu'




@app.route('/predict/values',methods=['GET'])
def values_predict():
    try:
        # get data
        dict_data = request.args.get('values')
        # restructuration de la données
        dict_data = pd.DataFrame(json.loads(dict_data) )
        # probability
        probability = model.predict_proba(dict_data)[0][1]
        #decision
        decision = model.predict( dict_data) 
        decision = int( decision )
    except Exception as e:
        return f"prediction NOT available! \n\n Une erreur s'est produite : {e}" 
    return jsonify({'proba':probability, 'decision': classify_mappings[decision]  })
    


if __name__ == '__main__':
    model = joblib.load("support/models/model.sav") # Load "model.pkl"
    print ('Model loaded')

    important_features = [
    'NAME_INCOME_TYPE', 'AMT_REQ_CREDIT_BUREAU_MON', 'NAME_EDUCATION_TYPE',
    'FONDKAPREMONT_MODE', 'OCCUPATION_TYPE', 'FLAG_OWN_CAR',
    'REG_CITY_NOT_WORK_CITY', 'REG_REGION_NOT_LIVE_REGION', 'REGION_POPULATION_RELATIVE',
    'AMT_REQ_CREDIT_BUREAU_WEEK', 'REG_REGION_NOT_WORK_REGION', 'AMT_ANNUITY',
    'DAYS_REGISTRATION', 'REGION_RATING_CLIENT_W_CITY', 'FLAG_DOCUMENT_5',
    'NAME_TYPE_SUITE', 'AMT_INCOME_TOTAL', 'FLAG_OWN_REALTY',
    'FLAG_DOCUMENT_13', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'DAYS_ID_PUBLISH',
    'AMT_REQ_CREDIT_BUREAU_HOUR', 'FLAG_WORK_PHONE', 'DAYS_BIRTH',
    'DEF_30_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_3', 'EXT_SOURCE_2',
    'DAYS_LAST_PHONE_CHANGE', 'FLAG_EMAIL', 'FLAG_DOCUMENT_9',
    'CODE_GENDER', 'DAYS_EMPLOYED', 'REG_CITY_NOT_LIVE_CITY',
    'AMT_GOODS_PRICE', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_11',
    'OBS_30_CNT_SOCIAL_CIRCLE', 'FLAG_PHONE', 'FLAG_DOCUMENT_6'
    ]

    
    app.run(port = 80, use_reloader = True,debug=False)#debug=True, 
