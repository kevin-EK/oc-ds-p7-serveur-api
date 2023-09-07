# import libraries
import streamlit as st
import os.path
import pandas as pd
# import numpy as np
import joblib
from evidently.report import Report
from evidently.metrics import DataDriftTable
from evidently.metrics import DatasetDriftMetric
from important_features import important_features

st.markdown("# Evidently html import ❄️")
st.sidebar.markdown("# DataDrift ❄️")

# page1.py
#if "shared" not in st.session_state:
#   st.session_state["shared"] = True
   
@st.cache_data
def load_2data():   
    # create ref and cur dataset for drift detection
    #important_features = joblib.load('support/data/list_col_to_keep.joblib')
    # data_ref = pd.read_csv("support/data/application_train.csv", usecols=important_features )
    # data_cur = pd.read_csv("support/data/application_test.csv",  usecols= important_features )
    data_ref = joblib.load("support/data/application_train.sav")
    data_ref = data_ref[important_features+['TARGET','SK_ID_CURR']]
    data_cur = joblib.load("support/data/application_test.sav")
    data_cur = data_cur[important_features+['SK_ID_CURR']]
    return data_ref,data_cur

ref_df , current_df = load_2data()
# dataset-level metrics
data_drift_dataset_report = Report(metrics=[
    DatasetDriftMetric(),
    DataDriftTable(),    
])

data_drift_dataset_report.run(reference_data = ref_df, current_data = current_df)

#report in a JSON format
#data_drift_dataset_report.save_html('report.html')


import streamlit.components.v1 as components

# >>> import plotly.express as px
# >>> fig = px.box(range(10))
# >>> fig.write_html('test.html')

st.header("evidently html import")

if os.path.isfile('report.html'):
    HtmlFile = open("report.html", 'r', encoding='utf-8')
else:
    data_drift_dataset_report.save_html('report.html')
    HtmlFile = open("report.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
print(source_code)
components.html(source_code,height = 2000,width = 800)