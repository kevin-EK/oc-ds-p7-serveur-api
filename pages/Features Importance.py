import numpy as np
import pandas as pd
import joblib, time
import streamlit as st
import streamlit.components.v1 as components
import shap
from streamlit_shap import st_shap



entete = st.empty()
if "data" in st.session_state:
    with st.spinner('Loading data...'):
        time.sleep(2)
    entete.success('Done!')
    entete.write("la valeur de la page 2 a été récupéré")
    entete.empty()

#joblib.dump(st.session_state.data, "support/data/data_session.sav")

# def st_shap(plot, height=None):
#     shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
#     components.html(shap_html, height=height)

st.title("SHAP in Streamlit (SHapley Additive exPlanations)\n")

st.sidebar.markdown("## :blue[SHAP Values]")
st.sidebar.markdown("Les valeurs SHAP montrent comment et dans quelle mesure chaque facteur a contribué à la prédiction du modèle finale.") 
st.sidebar.markdown("L'importance de chaque feature par rapport aux autres et la dépendance du modèle à l'égard de l'interaction entre les features.")


st.sidebar.markdown("### :orange[Liaison entre la probabilité obtenu par le model et les valeurs SHAP pour une variable binaire]")
st.sidebar.markdown("Pour une probabilié p prédit par le modèle.\
                    Cela nous donne une probabilité logarithmique prévue de ")
#st.sidebar.latex(r'''f(x) = ln\left(\frac{p}{1-p}\right)''')

st.sidebar.latex('f(x) = E[f(x)] + SHAP values')
st.sidebar.latex(r'''ln\left(\frac{p}{1-p}\right) = E[ln\left(\frac{p}{1-p}\right)] + SHAP values''')
#st.sidebar.image('support\data\log proba binary target.png')
st.sidebar.markdown("E[f(x)] est la valeur attendue de la variable cible, ou en d'autres termes, la moyenne de toutes les prédictions : ")
st.sidebar.code( "mean(model.predict(X))) .")


st.sidebar.markdown("Les valeurs SHAP positives sont interprétées comme augmentant les probabilités. \
                    Par exemple pour l'individu 100002, ext_source_3 a augmenté la probabilité logarithmique de 0.85 .\
                    Cette fonctionnalité a augmenté la probabilité que le modèle prévoie que le client soit un mauvais payeur. \
                    De même, les valeurs négatives diminuent les probabilités.")

# train XGBoost model
model = joblib.load("support/models/model.sav")

# get feature name
feature_namestr = model[0].get_feature_names_out()
feature_namestr = [c.replace('num__','').replace('cat__','').replace('disc__','').replace('NAME_EDUCATION_TYPE_','N_EDUC_TYP_')\
                   .replace('ORGANIZATION_TYPE_','ORGA_TYP_').replace('Secondary / secondary special','Secondary special')  for c in feature_namestr]


# transform data pipeline
transformed_data = model[:-1].transform(st.session_state.data)
#transform_datatest = model[:-1].transform(datatest)
#st.write('Apply DataFrame')
#st.table(transformed_data[:5,:])

#st.write('Conversion DataFrame')
transformed_data = pd.DataFrame(transformed_data, columns = feature_namestr )
#st.table(transformed_data.head())

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
#shap.initjs()
explainer = shap.Explainer(model[-1],  )#transformed_data


#st.write("shap_values")
if 'shap_values' not in st.session_state:
    shap_values = explainer.shap_values( transformed_data  ) #id = 228316
    #shap_values = joblib.load('support\data\shap_values.sav')
    st.session_state['shap_values'] = shap_values
else:
    shap_values = st.session_state['shap_values']


# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)      
if st.session_state.chosen_radio == 'Index':
    st.write("L'identifiant choisi est " + str(st.session_state.chosen_customer) )
    # Force plot
    # st_shap( 
    # shap.force_plot(
    #     explainer.expected_value[0], 
    #     shap_values[st.session_state.data.SK_ID_CURR==st.session_state.chosen_customer,:], 
    #     transformed_data.loc[st.session_state.data.SK_ID_CURR==st.session_state.chosen_customer]
    #     ), height=300 , width=1000 
    #     )
    

    #Waterfall plot
    shap_valuesWaterfall = explainer(transformed_data.loc[st.session_state.data.SK_ID_CURR==st.session_state.chosen_customer])
    exp = shap.Explanation(shap_valuesWaterfall.values, 
                           shap_valuesWaterfall.base_values[0][0], 
                           transformed_data.loc[st.session_state.data.SK_ID_CURR==st.session_state.chosen_customer])
    st_shap(shap.plots.waterfall(exp[0], max_display = 20), height=1000, width=1000 )
    
    #st.write(explainer(transformed_data.loc[st.session_state.data.SK_ID_CURR==st.session_state.chosen_customer]).values)

elif st.session_state.chosen_radio == 'Valeur':
    #st.write( 'Pas encore actif' )
    #st.write(st.session_state['valuesClient'])
    data_manu = pd.DataFrame(model[:-1].transform(pd.DataFrame(st.session_state['valuesClient'], index=[0])),
                             columns= feature_namestr)
    shap_valuesWaterfall = explainer( data_manu ) 
    exp = shap.Explanation(shap_valuesWaterfall.values, 
                           shap_valuesWaterfall.base_values[0][0], 
                           data_manu#shap_valuesWaterfall.data
                           )
    st_shap( shap.plots.waterfall(exp[0], max_display = 20) , width=1000)

    # st_shap( 
    #     shap.force_plot(
    #         explainer.expected_value[0], 
    #         shap_values[st.session_state.data.SK_ID_CURR==st.session_state.chosen_customer,:], 
    #         st.session_state.chosen_radio.valuesClient
    #         ) 
    #         )

# visualize the training set predictions
#shap.force_plot(shap.force_plot(explainer.expected_value, shap_values, st.session_state.data), 400)