
import pandas as pd
import numpy as np

def pre_processing(input_path,output_path):
    df = pd.read_csv(input_path)
    df = df.drop(['Last_Reporting_Date','Female_Personnel', 'Male_Personnel'], axis=1) #TODO - Breaking during Nrmalization
    str_cols = ['ISOCode3', 'Contributing_Country', 'Mission_Acronym', 'Personnel_Type']
    for col in df.columns:
        if col in str_cols:
            replace_map = {k: v for v, k in enumerate( df[col].unique())}
            df[col] = df[col].map(replace_map)
    #print(df)
    df = df.astype(np.float64)        
    df.to_pickle(output_path)

    