def get_data(data):
    import pandas as pd
    import numpy as np
    import torch.nn.functional as F
    if data == 'compas':  # coding COMPAS
        # load the data from ProPublica
        compas = pd.read_csv('data/compas-scores-two-years.csv', index_col=0)
        # Following COMPAS ProPublica analysis, we will filter out rows where days_b_screening_arrest is over 30 or under -30
        compas = compas[(compas['days_b_screening_arrest'] <= 30) & (
            compas['days_b_screening_arrest'] >= -30)]
        # Only interested in Black and White
        compas = compas[(compas['race'] == 'African-American') | (compas['race'] == 'Caucasian')]
        # Coding Age
        bins = [0, 25, 45, 1000]
        labels = ["less_25", "between_25_45", "more_45"]  # 0, 1 and 2
        compas['age_cat'] = pd.cut(compas.age, bins=bins, right=False, labels=labels)
        age = pd.get_dummies(compas.age_cat)
        age_less_25 = np.array(age)[:,0]
        age_between_25_45 = np.array(age)[:,1]
        age_more_45 = np.array(age)[:,2]
        # Sex: Male is 1 Female is 0
        male = np.array(compas.sex.astype("category").cat.codes)
        
        X = 
        y = np.array(compas['v_score_text'].apply(
            lambda x: 0 if x == 'Low' else 1))
        return compas
    else:
        return None
