from flask import Flask,  redirect, url_for ,request, jsonify
import urllib3
import pandas as pd
import numpy as np
import joblib
import json
from important_features import important_features

app = Flask(__name__)

classify_mappings = {0: 'Bon client', 1: 'Client à risque'}
path_server = 'https://oc-ds-p7-kevin-el-ce8c86036717.herokuapp.com/'#'http://127.0.0.1:80/'

@app.route("/")
def helloworld():
    return "<h1>Welcome to my api!</h1>"


@app.route('/load-agg-data')
def get_agg_data():
    # amelioration création base de données
    try:                
        # load data
        data1 = joblib.load("support/data/application_train.sav")
        #data1 = data1[important_features+['TARGET','SK_ID_CURR']]
        data2 = joblib.load("support/data/application_test.sav")
        #data2 = data2[important_features+['SK_ID_CURR']]
    
        # feature engenering
        data = pd.concat([data1, data2],axis=0).drop_duplicates()
        data['ANNUITY_INCOME_PERC'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
        data = data[data['CODE_GENDER'] != 'XNA']
        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            data[bin_feature], uniques = pd.factorize(data[bin_feature])
        ## return jsonify( {'data':data.to_dict('records'), 'data_agg':data_agg.to_dict('records')} ) 
        return jsonify( {'data':data.to_dict('records')} ) 
    except:
        return 'Aggregation data NOT available!'
    # http://127.0.0.1:5000/load-agg-data
    

@app.route('/load_data')
def get_data():
    # amelioration création base de données
    try:
        # load data
        df_train = joblib.load("support/data/application_train.sav")
        df_test = joblib.load("support/data/application_test.sav")
        list_index = list(set(df_train.SK_ID_CURR.to_list() + df_test.SK_ID_CURR.to_list())) # a tester
        #columns_to_keep = joblib.load("support/data/list_col_to_keep.joblib")
        #columns_to_keep = important_features

        # concatened dataframe
        data = pd.concat([df_train,df_test], axis = 0)

        # Filtered columns
        #data = data[['SK_ID_CURR','TARGET']+columns_to_keep]
        
        #feature ingenering
        data = data[data['CODE_GENDER'] != 'XNA']
        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            data[bin_feature], uniques = pd.factorize(data[bin_feature])
        # Categorical features with One-Hot encode
        #df, cat_cols = one_hot_encoder(df, nan_as_category)

        # NaN values for DAYS_EMPLOYED: 365.243 -> nan
        data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
        
        # Some simple new features (percentages)
        data['PERC_CREDIT_GOODS_PRICE'] = data['AMT_CREDIT'] / data['AMT_GOODS_PRICE']
        data['INCOME_CREDIT_PERC'] = data['AMT_INCOME_TOTAL'] / data['AMT_CREDIT']
        data['PAYMENT_RATE'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']
        data['INCOME_PER_CHILDREN'] = data['AMT_INCOME_TOTAL'] / (data['CNT_CHILDREN']+1)# revenu par enfant
        data['ANNUITY_INCOME_PERC'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
        data['DEF_RATE'] = data['DEF_60_CNT_SOCIAL_CIRCLE']/data['DEF_30_CNT_SOCIAL_CIRCLE'] 
        data['CHIDREN_RATE'] = data['CNT_CHILDREN'] / data['CNT_FAM_MEMBERS']# ratio d'enfant

        # ratio defaillance
        data['DEF_30_RATE'] = data['DEF_30_CNT_SOCIAL_CIRCLE'] / data['OBS_30_CNT_SOCIAL_CIRCLE']
        data['DEF_60_RATE'] = data['DEF_60_CNT_SOCIAL_CIRCLE'] / data['OBS_60_CNT_SOCIAL_CIRCLE']
                
        # pour recuperer les resultats use open url
        return jsonify({'index':list_index, 'data':data.to_dict('records')})
    except:
        return 'Data NOT available!'
    # http://127.0.0.1:5000/load_data
    
    
@app.route('/predict/index',methods=['GET'])
def index_predict():
    # get index
    idClient = int(request.args.get('idClient'))
    # get data
    try:
        http = urllib3.PoolManager()
        r = http.request('GET',path_server+'load_data')
        resData = json.loads(r.data)
    except:
        return 'loading api not available'

    if idClient in resData['index']:
        # filtered data
        valideData = pd.DataFrame.from_dict(resData['data'])
        valideData = valideData.loc[valideData.SK_ID_CURR == idClient,:].drop(columns = ['SK_ID_CURR','TARGET'])
        return redirect(url_for('values_predict',values = valideData.to_json(orient="columns") ))
    else:
        return 'Id non reconnu'
    # http://127.0.0.1:80/predict/index?idClient=116014
    # https://bienvoyons.pythonanywhere.com/predict/index?idClient=100002



@app.route('/predict/values',methods=['GET'])
def values_predict():
    model = joblib.load("support/models/model.sav")
    # get data
    dict_data = request.args.get('values')
    # restructuration de la données
    dict_data = pd.DataFrame(json.loads(dict_data) )
    # probability
    probability = model.predict_proba(dict_data)[0][1]
    #decision
    decision = model.predict( dict_data) 
    decision = int( decision )
    return jsonify({'proba':probability, 'decision': classify_mappings[decision]  })
    # http://127.0.0.1:5000/predict/values?values=%7B%22AMT_ANNUITY%22%3A%7B%220%22%3A16713.0%7D,%22AMT_CREDIT%22%3A%7B%220%22%3A288873.0%7D,%22AMT_INCOME_TOTAL%22%3A%7B%220%22%3A67500.0%7D,%22ANNUITY_INCOME_PERC%22%3A%7B%220%22%3A0.2476%7D,%22CNT_CHILDREN%22%3A%7B%220%22%3A0%7D,%22CODE_GENDER%22%3A%7B%220%22%3A0%7D,%22DAYS_BIRTH%22%3A%7B%220%22%3A-16963%7D,%22DAYS_EMPLOYED%22%3A%7B%220%22%3A-1746%7D,%22EXT_SOURCE_2%22%3A%7B%220%22%3A0.657665461%7D,%22EXT_SOURCE_3%22%3A%7B%220%22%3A0.7091891097%7D,%22FLAG_OWN_CAR%22%3A%7B%220%22%3A0%7D,%22FLAG_OWN_REALTY%22%3A%7B%220%22%3A0%7D,%22INCOME_CREDIT_PERC%22%3A%7B%220%22%3A0.2336666978%7D,%22NAME_EDUCATION_TYPE%22%3A%7B%220%22%3A%22Secondary+%5C%2F+secondary+special%22%7D,%22NAME_FAMILY_STATUS%22%3A%7B%220%22%3A%22Married%22%7D,%22NAME_INCOME_TYPE%22%3A%7B%220%22%3A%22Working%22%7D,%22OCCUPATION_TYPE%22%3A%7B%220%22%3A%22Laborers%22%7D,%22ORGANIZATION_TYPE%22%3A%7B%220%22%3A%22Business+Entity+Type+3%22%7D,%22REG_CITY_NOT_LIVE_CITY%22%3A%7B%220%22%3A0%7D,%22REG_CITY_NOT_WORK_CITY%22%3A%7B%220%22%3A0%7D,%22REG_REGION_NOT_LIVE_REGION%22%3A%7B%220%22%3A0%7D,%22REG_REGION_NOT_WORK_REGION%22%3A%7B%220%22%3A0%7D,%22time_to_repay%22%3A%7B%220%22%3A17.2843295638%7D%7D
    # https://bienvoyons.pythonanywhere.com/predict/values?values=%7B%22AMT_ANNUITY%22%3A%7B%220%22%3A16713.0%7D,%22AMT_CREDIT%22%3A%7B%220%22%3A288873.0%7D,%22AMT_INCOME_TOTAL%22%3A%7B%220%22%3A67500.0%7D,%22ANNUITY_INCOME_PERC%22%3A%7B%220%22%3A0.2476%7D,%22CNT_CHILDREN%22%3A%7B%220%22%3A0%7D,%22CODE_GENDER%22%3A%7B%220%22%3A0%7D,%22DAYS_BIRTH%22%3A%7B%220%22%3A-16963%7D,%22DAYS_EMPLOYED%22%3A%7B%220%22%3A-1746%7D,%22EXT_SOURCE_2%22%3A%7B%220%22%3A0.657665461%7D,%22EXT_SOURCE_3%22%3A%7B%220%22%3A0.7091891097%7D,%22FLAG_OWN_CAR%22%3A%7B%220%22%3A0%7D,%22FLAG_OWN_REALTY%22%3A%7B%220%22%3A0%7D,%22INCOME_CREDIT_PERC%22%3A%7B%220%22%3A0.2336666978%7D,%22NAME_EDUCATION_TYPE%22%3A%7B%220%22%3A%22Secondary+%5C%2F+secondary+special%22%7D,%22NAME_FAMILY_STATUS%22%3A%7B%220%22%3A%22Married%22%7D,%22NAME_INCOME_TYPE%22%3A%7B%220%22%3A%22Working%22%7D,%22OCCUPATION_TYPE%22%3A%7B%220%22%3A%22Laborers%22%7D,%22ORGANIZATION_TYPE%22%3A%7B%220%22%3A%22Business+Entity+Type+3%22%7D,%22REG_CITY_NOT_LIVE_CITY%22%3A%7B%220%22%3A0%7D,%22REG_CITY_NOT_WORK_CITY%22%3A%7B%220%22%3A0%7D,%22REG_REGION_NOT_LIVE_REGION%22%3A%7B%220%22%3A0%7D,%22REG_REGION_NOT_WORK_REGION%22%3A%7B%220%22%3A0%7D,%22time_to_repay%22%3A%7B%220%22%3A17.2843295638%7D%7D

if __name__ == '__main__':
    model = joblib.load("support/models/model.sav") # Load "support/models/HistGradientBoostingClassifier_model.sav"
    print ('Model loaded')    
    app.run(port = 80, use_reloader = True,debug=False)#debug=True, 