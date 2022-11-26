import numpy as np
import sklearn
import streamlit as st
import pickle
import pandas as pd

df = pd.read_csv('heart-disease.csv')
y = df['target']
X = df.drop('target', axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

pickle.dump(model, open('HeartClf.pkl', 'wb'))
