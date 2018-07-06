import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import metrics, datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

data = pd.read_csv("austin_final.csv")

X = data.drop(['PrecipitationSumInches'], axis=1)
Y_temp = data['PrecipitationSumInches']
Y_temp = Y_temp.values.reshape(-1, 1)

'''
# Generate multiclass one-hot columns
Y = pd.DataFrame(columns=['No Rains', 'Drizzle', 'Moderate Rains', 'Heavy Rains'])
for i in range(Y_temp.size):
	if(Y_temp[i]<0.001):
		Y.loc[i] = [1, 0, 0, 0]
	elif(Y_temp[i]>=0.001 and Y_temp[i]<0.1):
		Y.loc[i] = [0, 1, 0, 0]
	elif(Y_temp[i]>=0.1 and Y_temp[i]<0.4):
		Y.loc[i] = [0, 0, 1, 0]
	else:
		Y.loc[i] = [0, 0, 0, 1]
'''
Y = []

x1 = pd.DataFrame(columns=X.columns.values)
x2 = pd.DataFrame(columns=X.columns.values)
x3 = pd.DataFrame(columns=X.columns.values)
x4 = pd.DataFrame(columns=X.columns.values)
for i in range(Y_temp.size):
	if(Y_temp[i]<0.001):
		Y.append(1)
		x1.loc[i] = X.loc[i]
	elif(Y_temp[i]>=0.001 and Y_temp[i]<0.1):
		Y.append(2)
		x2.loc[i] = X.loc[i]
	elif(Y_temp[i]>=0.1 and Y_temp[i]<1.2):
		Y.append(3)
		x3.loc[i] = X.loc[i]
	else:
		Y.append(4)
		x4.loc[i] = X.loc[i]

Y = np.array(Y).reshape(len(Y), )

logr = LogisticRegression(multi_class='ovr', solver='liblinear').fit(X, Y)

#input = np.array([[74], [60], [45], [67], [49], [43], [93], [75], [57], [29.68], [10], [7], [2], [20], [4], [31]])
input = np.array([[58], [43], [28], [37], [22], [18], [75], [49], [22], [30.35], [10], [10], [10], [14], [4], [21]])
input = input.reshape(1, -1)
classes = ['None', 'No Rain', 'Drizzles', 'Moderate Rains', 'Heavy Rains']
print(classes[int(logr.predict(input))])

x1 = x1.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH'], axis=1)
x2 = x2.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH'], axis=1)
x3 = x3.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH'], axis=1)
x4 = x4.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH'], axis=1)


for i in range(6):
	plt.subplot(3,2,i+1)
	plt.scatter(x1.index.values, x1[x1.columns.values[i]], color='b')
	plt.scatter(x2.index.values, x2[x2.columns.values[i]], color='r')
	plt.scatter(x3.index.values, x3[x3.columns.values[i]], color='g')
	plt.scatter(x4.index.values, x4[x4.columns.values[i]], color='y')
	plt.title(x1.columns.values[i])

blue_patch = mpatches.Patch(color='blue', label='No rains')
red_patch = mpatches.Patch(color='red', label='Drizzles')
green_patch = mpatches.Patch(color='green', label='Moderate rains')
yellow_patch = mpatches.Patch(color='yellow', label='Heavy rains')
plt.legend(handles=[blue_patch, red_patch, green_patch, yellow_patch], bbox_to_anchor=(1.05, 2), loc=2, borderaxespad=0.)

plt.show()
