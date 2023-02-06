import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def LoadData(path):
	input_data = pd.read_csv(path)

	shape_num = input_data.shape
	X_row = input_data.iloc[:, :shape_num[1]-1]
	Y_row = input_data.iloc[:, shape_num[1]-1]
	input_size = X_row.shape[1]

	train_x, val_test_x, train_y, val_test_y = train_test_split(X_row, Y_row, test_size = 0.3, shuffle = True)
	val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size = 0.5, shuffle = True)

	return input_size, train_x, val_test_x, train_y, val_test_y, val_x, test_x, val_y, test_y