#below line is used use for data manupulation
import pandas as pd
import numpy as np
#below line(3,4)are used for model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# to read the comma separated file
df =pd.read_csv("C:\\Users\\utuku\\OneDrive\\Desktop\\complete learnings\\online\\(ml)project 1\\real_estate\\__MACOSX\\Real_Estate.csv")
print(df.head()) 

#model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train )

df.dropna(inplace=True)  #remove missing rows
df=pd.get_dummies(df,columns= ['Location'],drop_first=True) #convert location to numbers
X=df.drop('House price of unit area' , axis = 1)
y = df['House price of unit area']

#testing the model
y_pred=model.predict(X_test)
from sklearn.metrics import mean_squared_error,r2_score
print("Mean Squared Error:", mean_squared_error(y_test,y_pred))
print("R^2 Score:", r2_score(y_test,y_pred))


new_data = np.array([[1400, 3, 8, 1]])  # Example input after encoding
print("Predicted Price:", model.predict(new_data))
