#!/usr/bin/env python
# coding: utf-8

# # NOTEBOOK SETUP

# In[49]:


import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # DATASET OVERVIEW

# In[32]:


pokemon = pd.read_csv("Downloads/pokemon_data.csv")


# In[33]:


pokemon.columns


# In[34]:


pokemon.head()


# In[35]:


#Getting basic statistics of the dataset 
pokemon.describe()


# In[36]:


print(pokemon.info()) 


# In[37]:


pokemon.columns


# In[38]:


# Adding 'Total' feature
pokemon['Total'] = pokemon[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].sum(axis=1)


# In[39]:


# Type-wise stats
type_stats = pokemon.groupby('Type 1').agg({
    'HP': 'mean',
    'Attack': 'mean',
    'Defense': 'mean',
    'Total': 'mean'
}).sort_values(by='Total', ascending=False)
print(type_stats)


# # DATA CLEANING

# In[40]:


#check how many null values in the dataset
print(pokemon.isnull().sum())


# In[41]:


# Handling missing values
pokemon['Type 2'].fillna('None', inplace=True)


# In[44]:


# Check for duplicate entries and drop them if found
duplicates = df.duplicated().sum()
print(f"Number of duplicate entries: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()


# # DESCRIPTIVE ANALYSIS 

# In[42]:


# Legendary vs Non-Legendary statistics
print(pokemon.groupby('Legendary').mean())


# In[45]:


# Descriptive statistics for primary types
type1_grouped = df.groupby('Type 1').mean()
print("\nAverage stats by Type 1:")
print(type1_grouped[['HP', 'Attack', 'Defense', 'Speed']])


# In[43]:


#DESCRIPTIVE AND COMPARATIVE DATA ANALYSIS
## Average stats by Type 1
avg_stats_by_type = df.groupby('Type 1').mean()
avg_stats_by_type[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]


# In[46]:


# Analyze the count distribution of each primary type
type_counts = df['Type 1'].value_counts()
print("\nCount of each primary type:")
print(type_counts)


# # DATA VISUALIZATION

# In[50]:


# Distribution of Pokémon primary types
plt.figure(figsize=(10, 6))
sns.countplot(x='Type 1', data=df, palette='viridis', order=df['Type 1'].value_counts().index)
plt.title('Distribution of Pokémon by Primary Type')
plt.xlabel('Primary Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[51]:


# Boxplot comparison for Legendary vs. Non-Legendary Pokémon by stats
plt.figure(figsize=(10, 6))
sns.boxplot(x='Legendary', y='Attack', data=df, palette='coolwarm')
plt.title('Attack Comparison between Legendary and Non-Legendary Pokémon')
plt.show()


# In[52]:


# Pair plot for the stats columns to analyze relationships
stats_columns = ['HP', 'Attack', 'Defense', 'Speed']
sns.pairplot(df[stats_columns])
plt.suptitle("Pair Plot of Pokémon Stats", y=1.02)
plt.show()


# In[53]:


# Heatmap for correlation between stats
plt.figure(figsize=(8, 6))
correlation_matrix = df[stats_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='Blues')
plt.title('Correlation Heatmap of Pokémon Stats')
plt.show()


# # ADVANCED ANALYSIS AND FEATURE EXPLORATION

# In[54]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Feature selection for clustering
features = df[['HP', 'Attack', 'Defense', 'Speed']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# In[55]:


# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)


# In[56]:


# Plot clusters in a 2D space (Attack vs Defense as example)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Attack', y='Defense', hue='Cluster', data=df, palette='Set2')
plt.title('K-Means Clustering of Pokémon based on Attack and Defense')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.legend(title='Cluster')
plt.show()


# In[57]:


# Example feature engineering: calculating Total Power
df['Total Power'] = df[['HP', 'Attack', 'Defense', 'Speed']].sum(axis=1)

# Analyzing Total Power distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Total Power'], kde=True, color='purple')
plt.title('Distribution of Total Power')
plt.xlabel('Total Power')
plt.ylabel('Frequency')
plt.show()


# In[ ]:




