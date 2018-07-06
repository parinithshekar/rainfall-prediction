import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("austin_final.csv")

X = data.drop(['PrecipitationSumInches'], axis=1)
Y = data['PrecipitationSumInches']
Y = Y.values.reshape(-1, 1)

day_index = 798
days = [i for i in range(Y.size)]

clf = LinearRegression()
clf.fit(X, Y)

input = np.array([[74], [60], [45], [67], [49], [43], [93], [75], [57], [29.68], [10], [7], [2], [20], [4], [31]])
input = input.reshape(1, -1)
print(clf.predict(input))

plt.scatter(days, Y)
plt.scatter(days[day_index], Y[day_index], color='r')
plt.title("Precipitation level")
plt.xlabel("Days")
plt.ylabel("Precipitation in inches")
'''
for i, txt in enumerate(days):
    plt.annotate(txt, (days[i],Y[i]))
'''
plt.show()
x_vis = X.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH'], axis=1)

for i in range(x_vis.columns.size):
	plt.subplot(3,2,i+1)
	plt.scatter(days, x_vis[x_vis.columns.values[i]])
	plt.scatter(days[day_index], x_vis[x_vis.columns.values[i]][day_index], color='r')
	plt.title(x_vis.columns.values[i])

plt.show()
