This repository builds a Linear as well as a Logistic model to predict rainfalls in Austin, Texas.

The following dataset constitutes 3.5 years worth of weather data, including temperature, humidity, dewpoints, etc: https://www.kaggle.com/grubenm/austin-weather

Execute the files **linearRegression.py** and **logisticRegression.py** to obtain predictions for an arbitrary day with hardcoded input parameters.

![5b3f34740ce87](https://i.loli.net/2018/07/06/5b3f34740ce87.png)

A day (in red) having a precipitation of about 2 inches is tracked across multiple parameters.

![5b3f34511cde8](https://i.loli.net/2018/07/06/5b3f34511cde8.png)

Manually classifying the precipitation levels into 4 different classes as follows:

- No Rain: precipitation<0.001

- Drizzle: 0.001<=precipitation<0.1
- Moderate Rains: 0.1<=precipitation< 1.2
- Heavy Rains: precipitation>=1.2

The graphs we obtain after classifying express various trends which tie rainfall and humidity, visibility and temperature together.

![5b3f36bd69c75](https://i.loli.net/2018/07/06/5b3f36bd69c75.png)






