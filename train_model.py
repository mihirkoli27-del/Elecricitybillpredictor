import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
# Load dataset
data=pd.read_csv("dataset/electricity_data.csv")
# Convert Season text to number
le=LabelEncoder()
data['Season']=le.fit_transform(data['Season'])
# Features and target
X=data[['Units','Appliances','Hours','Season','Rate']]
y=data['Bill']

# Split data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42
)
# Train model
model=LinearRegression()
model.fit(X_train,y_train)
# Save model
with open("model.pkl","wb")as f:pickle.dump(model,f)
print("Model trained and saved successfully!")