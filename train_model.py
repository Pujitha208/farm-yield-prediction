 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Create sample dataset
data = {
    "rainfall": [100,120,80,90,110,95,105,115,85,75],
    "temperature": [25,27,23,24,26,22,28,29,21,20],
    "fertilizer": [30,35,25,20,40,30,45,50,28,18],
    "yield": [200,220,180,190,210,195,230,240,175,160]
}

df = pd.DataFrame(data)

X = df[["rainfall","temperature","fertilizer"]]
y = df["yield"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LinearRegression()
model.fit(X_train,y_train)

# Save model
pickle.dump(model, open("model.pkl","wb"))

print("Model saved successfully")
