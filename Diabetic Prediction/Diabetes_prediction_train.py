import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
from sklearn.ensemble import IsolationForest

home_dir = os.path.expanduser("~")
file_path = os.path.join(home_dir, "downloads", "diabetes_unbalanced.csv")
df = pd.read_csv(file_path)

model = IsolationForest()
model.fit(df)
df['anomaly']= model.predict(df)
df.drop(df[df['anomaly']==-1].index,inplace = True)

# Selecting the variables with strong correlation
strong_corr_vars = ['GenHlth', 'HighBP', 'DiffWalk', 'BMI', 'HighChol', 'Age', 'HeartDiseaseorAttack', 'PhysHlth', 'PhysActivity', 'Education', 'Income', 'Diabetes_binary']

# Filtering the dataset based on the selected variables
df_filtered = df[strong_corr_vars]

# Splitting the data into input features (X) and target variable (y)
X = df_filtered.drop('Diabetes_binary', axis=1)
y = df_filtered['Diabetes_binary']

# Split the filtered data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Model 4: Gradient Boosting
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train_scaled, y_train)
gb_y_pred = gb_model.predict(X_test_scaled)

print("Gradient Boosting Results:")
print('Accuracy:', accuracy_score(y_test, gb_y_pred))

# Save the model
with open('gb_model.pkl', 'wb') as file:
    pickle.dump(gb_model, file)

# Save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Save the isolation forest model
with open('isolation_forest_model.pkl', 'wb') as file:
    pickle.dump(model, file)

