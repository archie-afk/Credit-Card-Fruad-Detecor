import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Shape
print("Shape:", df.shape)

# Preview
print("\nFirst 5 rows:")
print(df.head())

# Column info
print("\nColumn types:")
print(df.info())

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Class balance
print("\nClass distribution:")
print(df['Class'].value_counts())
print(df['Class'].value_counts(normalize=True) * 100)