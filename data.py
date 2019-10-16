def get_data(data):
	import pandas as pd
	if data == 'compas':
		compas = pd.read_csv('data/compas-scores-two-years.csv', index_col=0)
		compas = compas[(compas['days_b_screening_arrest'] <= 30) & (compas['days_b_screening_arrest'] >= -30)]
		return compas
	else:
		return None
