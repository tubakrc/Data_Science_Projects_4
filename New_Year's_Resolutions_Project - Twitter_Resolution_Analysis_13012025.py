# %% [markdown]
# # <u>New Year's Resolutions Project - Twitter Resolution Analysis</u>

# %% [markdown]
# ### Table of Contents
# 
# **[Step 0: Project Overview](#Step-0:-Project-Overview)**
# 
# > [Dataset Info](#Dataset-Info)
# 
# > [Dataset Columns and Explanations](#Dataset-Columns-and-Explanations)
# 
# > [Key Features](#Key-Features)
# 
# **[Step 1: Business Understanding](#Step-1:-Business-Understanding)**
# 
# > [Time-Based Insights](#Time-Based-Insights)
# 
# > [Sentiment Analysis (Optional for Early Insights)](#Sentiment-Analysis-(Optional-for-Early-Insights))
# 
# **[Step 2: Data Understanding](#Step-2:-Data-Understanding)**
# 
# **[Step 3: Data Preparation](#Step-3:-Data-Preparation)**
# 
# **[Step 4: Modelling/Analysis](#Step-4:-Modelling/Analysis)**
# 
# > [Sentiment Analysis of Resolutions](#Sentiment-Analysis-of-Resolutions)
# 
# > [Resolution Popularity by Location](#Resolution-Popularity-by-Location)
# 
# > [Impact of Retweets on Resolution Categories](#Impact-of-Retweets-on-Resolution-Categories)
# 
# > [Text Analysis and Clustering](#Text-Analysis-and-Clustering)
# 
# > [Gender and Region-Based Preferences](#Gender-and-Region-Based-Preferences)
# 
# **[Step 5: Evaluation](#Step-5:-Evaluation)**
# 
# > [Validate Sentiment Analysis](#Validate-Sentiment-Analysis)
# 
# > [Validate Topic Modeling](#Validate-Topic-Modeling)
# 
# > [Overall Evaluation Report](#Overall-Evaluation-Report)
# 
# **[REFERENCES](#REFERENCES)**

# %% [markdown]
# # Step 0: Project Overview

# %% [markdown]
# ## Project Name: Twitter Resolution Analysis
# * This project analyzes tweets to explore sentiment, resolution popularity, and trends based on user activity, geography, and demographics. 
# * The insights help uncover user preferences, engagement patterns, and regional variations in resolutions.

# %% [markdown]
# ## Dataset Info
# * I used the dataset which named "new_year_resolutions_dataset.csv" on the Kaggle website.
#   
#   ( https://www.kaggle.com/datasets/andrewmvd/new-years-resolutions/data )
# * This dataset, comprising 5,002 tweets about New Year's resolutions, offers valuable insights to help you boost your chances of achieving your own goals!

# %% [markdown]
# ## Dataset Columns and Explanations
# |Column Name	| Description	| Purpose |
# | :----------- | :--------------| :-------|
# |tweet_id	   | Unique identifier for each tweet.	| Used to differentiate individual tweets and manage duplicates or subsets.|
# |text| The content of the tweet.	| Provides the primary data for text-based analysis, including sentiment analysis and topic modeling.|
# |resolution_topics|	The topic or theme associated with the resolution (e.g., health, finance, relationships).| Enables grouping and comparison of tweets based on themes.|
# |resolution_category|	The broader category of the resolution (e.g., personal goals, professional growth).| Supports higher-level analysis and visualization of trends.|
# |tweet_region|	The geographic region associated with the tweet (e.g., state, country).| Supports geographic analysis to identify regional trends in resolutions.|
# |tweet_state|	The specific state associated with the tweet in the USA (if applicable). | Used for detailed geographic visualizations, such as heatmaps.|
# |tweet_coord|	Latitude and longitude coordinates of the tweet.| Enables mapping of tweet activity and resolution density on geographic maps.|
# |tweet_created|	The timestamp when the tweet was created.| Useful for time-series analysis and identifying hourly, daily, or seasonal trends.|
# |retweet_count|	Number of times the tweet was retweeted.| Indicates engagement and virality, used to assess resolution popularity.|
# |user_timezone|	The timezone of the user who tweeted.| Helps in aligning data to specific regions and understanding tweet patterns based on timezones.|
# |gender| Indicates the gender of the user who tweeted the resolution, providing insights into gender-based trends and preferences.|Helps to analyze gender-based|
# |name| Represents the name or username of the individual who posted the tweet |Useful for identifying unique users or conducting further analyses.|
# |tweet_date| Captures the date when the tweet was posted | Enabling time-based analysis to identify trends or patterns around New Year’s resolutions.|
# |tweet_location| Specifies the location mentioned in the tweet | Helps to analyze geographic trends and regional differences in resolutions.|

# %% [markdown]
# ## Key Features
# * *Sentiment Analysis:* Understand emotional responses to resolutions.
# * *Geographic Insights:* Visualize resolution popularity and activity by state and region.
# * *Temporal Patterns:* Explore trends in tweet activity by time.
# * *Engagement Metrics:* Identify the most retweeted resolutions to gauge user interest and virality.
# * *Textual Analysis:* Perform topic modeling and clustering for thematic exploration.

# %% [markdown]
# # Step 1: Business Understanding

# %%
# !pip uninstall tensorflow tensorflow-intel

# %%
# !pip cache purge

# %%
# !pip show tensorflow
# !pip show tensorflow-intel

# %%
!pip install tensorflow==2.17.0 tensorflow-intel==2.17.0

# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

!pip install keras
# !pip uninstall keras
# !pip install tf-keras

import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

# %%
# Read data and inspect sample
df = pd.read_csv('C:/Users/Tuba/OneDrive/Masaüstü/Desktop/MINDSET-Python/new_year_resolutions_dataset.csv', sep = ';')
df.sample(5)

# %%
print(df.head())

print(df.info())

print(df.isnull().sum())

print(df.describe())

# %%
# Convert top topics to DataFrame
top_topics = df['resolution_topics'].value_counts().head(10)
top_topics_df = top_topics.reset_index()
top_topics_df.columns = ['resolution_topics', 'count']

# Add a hue column (e.g., distinguishing popular and less popular topics)
top_topics_df['popularity'] = ['Popular' if x > top_topics.mean() else 'Less Popular' for x in top_topics_df['count']]

# Plot with hue
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_topics_df,
    x='count',
    y='resolution_topics',
    hue='popularity',
    palette="viridis"
)
plt.title("Top 10 Resolution Topics with Popularity")
plt.xlabel("Count")
plt.ylabel("Resolution Topics")
plt.legend(title="Popularity")
plt.tight_layout()
plt.show()

# %%
# Resolution categories distribution
plt.figure(figsize=(8, 5))
df['resolution_category'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Resolution Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=75)
plt.show()

# %%
# Gender distribution
gender_counts = df['gender'].value_counts()
plt.figure(figsize=(6, 4))
gender_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'pink', 'gray'])
plt.title("Gender Distribution")
plt.ylabel("")
plt.show()

# %%
# Add a 'hue' variable to classify retweet counts
top_retweeted = df.sort_values(by='retweet_count', ascending=False).head(10)
top_retweeted['engagement_level'] = ['High' if x > top_retweeted['retweet_count'].mean() else 'Moderate' 
                                     for x in top_retweeted['retweet_count']]

# Plot with hue
plt.figure(figsize=(20, 8))
sns.barplot(
    x=top_retweeted['retweet_count'], 
    y=top_retweeted['text'], 
    hue=top_retweeted['engagement_level'], 
    palette="coolwarm"
)
plt.title("Top 10 Retweeted Resolutions with Engagement Levels")
plt.xlabel("Retweet Count")
plt.ylabel("Tweet Text")
plt.legend(title="Engagement Level")
plt.tight_layout()
plt.show()

# %%
# Add a hue variable to classify regions by activity level
region_counts = df['tweet_region'].value_counts().head(10).reset_index()
region_counts.columns = ['tweet_region', 'count']
region_counts['activity_level'] = ['High' if x > region_counts['count'].mean() else 'Moderate' for x in region_counts['count']]

# Plot with hue
plt.figure(figsize=(10, 5))
sns.barplot(
    x='count', 
    y='tweet_region', 
    hue='activity_level', 
    data=region_counts, 
    palette="magma"
)
plt.title("Top 10 Regions with Most Tweets by Activity Level")
plt.xlabel("Tweet Count")
plt.ylabel("Region")
plt.legend(title="Activity Level", loc="upper right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Time-Based Insights

# %%
# Convert tweet_date to datetime format
df['tweet_date'] = pd.to_datetime(df['tweet_date'])
df

# %%
# Analyze tweet activity by day
tweet_activity = df['tweet_date'].dt.date.value_counts().sort_index()
plt.figure(figsize=(20, 8))
tweet_activity.plot(kind='line', color='green', marker='o')
plt.title("Tweet Activity Over Time")
plt.xlabel("Date")
plt.ylabel("Tweet Count")
plt.grid()
plt.show()

# %%
# Analyze tweet activity by hour (if tweet_created contains time)
df['tweet_hour'] = pd.to_datetime(df['tweet_created']).dt.hour
hourly_activity = df['tweet_hour'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
sns.barplot(x=hourly_activity.index, y=hourly_activity.values, palette="autumn")
plt.title("Tweet Activity by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Tweet Count")
plt.show()

# %%
df.columns

# %% [markdown]
# ## Sentiment Analysis (Optional for Early Insights)

# %%
!pip install textblob
from textblob import TextBlob

# Calculate sentiment polarity for each tweet
df['sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df

# %%
# Sentiment distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['sentiment'], kde=True, color='purple', bins=30)
plt.title("Sentiment Distribution of Tweets")
plt.xlabel("Sentiment Polarity")
plt.ylabel("Density")
plt.show()

# %%
# Average sentiment by resolution category
avg_sentiment_category = df.groupby('resolution_category')['sentiment'].mean().sort_values()
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_sentiment_category.values, y=avg_sentiment_category.index, palette="cool")
plt.title("Average Sentiment by Resolution Category")
plt.xlabel("Average Sentiment Polarity")
plt.ylabel("Resolution Category")
plt.show()

# %% [markdown]
# [Table of Contents](#Table-of-Contents)

# %% [markdown]
# # Step 2: Data Understanding

# %%
print("Dataset Overview:")
print(df.head())
print("\nColumns and Data Types:")
print(df.info())

# Check missing values
print("\nMissing Values Count:")
print(df.isnull().sum())

# Basic descriptive statistics for numeric columns
print("\nDescriptive Statistics for Numeric Columns:")
print(df.describe())

# Value counts for categorical columns (e.g., 'gender', 'resolution_category')
print("\nValue Counts for 'resolution_category':")
print(df['resolution_category'].value_counts())

# %%
df['tweet_coord'].fillna("[0, 0]", inplace=True)
df['retweet_count'].fillna(df['retweet_count'].mean(), inplace=True)
df['user_timezone'].fillna(df['user_timezone'].mode()[0], inplace=True)

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Top resolution topics
top_topics = df['resolution_topics'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_topics.values, y=top_topics.index, palette="viridis")
plt.title("Top 10 Resolution Topics")
plt.xlabel("Count")
plt.ylabel("Resolution Topics")
plt.show()

# Top resolution categories
top_categories = df['resolution_category'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=top_categories.values, y=top_categories.index, palette="coolwarm")
plt.title("Resolution Categories Distribution")
plt.xlabel("Count")
plt.ylabel("Resolution Categories")
plt.show()

# %%
# Gender distribution
plt.figure(figsize=(6, 4))
df['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'pink', 'gray'])
plt.title("Gender Distribution")
plt.ylabel("")
plt.show()

# Top 10 tweet regions
top_regions = df['tweet_region'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_regions.values, y=top_regions.index, palette="magma")
plt.title("Top 10 Tweet Regions")
plt.xlabel("Count")
plt.ylabel("Tweet Region")
plt.show()

# Top 10 tweet states
top_states = df['tweet_state'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_states.values, y=top_states.index, palette="autumn")
plt.title("Top 10 Tweet States")
plt.xlabel("Count")
plt.ylabel("Tweet State")
plt.show()

# %%
# Retweet count distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['retweet_count'], bins=50, kde=True, color='blue')
plt.title("Distribution of Retweet Counts")
plt.xlabel("Retweet Count")
plt.ylabel("Frequency")
plt.show()

# Top 10 most retweeted tweets
top_retweeted = df.sort_values(by='retweet_count', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_retweeted['retweet_count'], y=top_retweeted['text'], palette="cool")
plt.title("Top 10 Most Retweeted Tweets")
plt.xlabel("Retweet Count")
plt.ylabel("Tweet Text")
plt.show()

# %%
# Top 10 tweet locations
top_locations = df['tweet_location'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_locations.values, y=top_locations.index, palette="spring")
plt.title("Top 10 Tweet Locations")
plt.xlabel("Count");
plt.ylabel("Location")
plt.show()

# %%
!pip install folium
import folium

# Create a map centered on a general location
tweet_map = folium.Map(location=[20, 0], zoom_start=2)

# Add markers for available coordinates (if 'tweet_coord' contains lat/lon pairs as strings)
if 'tweet_coord' in df.columns:
    for coord in df['tweet_coord'].dropna():
        lat, lon = map(float, coord.strip("[]").split(","))
        folium.CircleMarker(location=[lat, lon], radius=2, color='blue', fill=True).add_to(tweet_map)

# Display the map
tweet_map.save("tweet_activity_map.html")
print("Map saved as 'tweet_activity_map.html'. Open it in your browser to view.")

# %% [markdown]
# # Step 3: Data Preparation

# %%
df.isnull().sum()

# %%
df['gender'] = df['gender'].str.strip().str.lower()
df['tweet_region'] = df['tweet_region'].str.strip().str.title()
df

# %%
import re

def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

df['text_cleaned'] = df['text'].apply(clean_text)


# %%
df

# %%
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = word_tokenize(text)  # Tokenize text
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens])

# Apply lemmatization to the cleaned text
df['text_cleaned'] = df['text_cleaned'].apply(lemmatize_text)

# %%
# from textblob import TextBlob

# # Compute sentiment polarity (ranges from -1 to 1)
# df['sentiment_score'] = df['text_cleaned'].apply(lambda x: TextBlob(x).sentiment.polarity)
# # df['final_sentiment_label'] = df['sentiment_score'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')
# df['sentiment_label'] = df['sentiment_score'].apply(
#     lambda x: 'Positive' if x > 0.2 else 'Negative' if x < -0.2 else 'Neutral'
# )

# %%
# !pip install tf-keras
# !pip install transformers
# from transformers import pipeline

# # Use a pre-trained model for sentiment analysis
# sentiment_analyzer = pipeline("sentiment-analysis")

# # Apply sentiment analysis
# df['sentiment_label'] = df['text_cleaned'].apply(lambda x: sentiment_analyzer(x)[0]['label'])

# %%
!pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()
#analyzer.lexicon.update({'resolution': 0.5, 'achievement': 0.8, 'goal': 0.5, 'failure': -0.6, 'improvement': 0.7})  # Custom weights

# Apply VADER sentiment analysis
df['sentiment_score'] = df['text_cleaned'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df['final_sentiment_label'] = df['sentiment_score'].apply(
    lambda score: 'Positive' if score > 0.2 else ('Negative' if score < -0.2 else 'Neutral')
)

# %%
from sklearn.utils import resample

# Separate majority and minority classes
positive = df[df['final_sentiment_label'] == 'Positive']
negative = df[df['final_sentiment_label'] == 'Negative']
neutral = df[df['final_sentiment_label'] == 'Neutral']

# Upsample minority classes
positive_upsampled = resample(positive, replace=True, n_samples=len(neutral), random_state=42)
negative_upsampled = resample(negative, replace=True, n_samples=len(neutral), random_state=42)

# Combine back
df_balanced = pd.concat([neutral, positive_upsampled, negative_upsampled])

# %%
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split

# # Vectorize text data
# vectorizer = CountVectorizer(max_features=1000)
# X_text = vectorizer.fit_transform(df['text_cleaned']).toarray()

# # Combine VADER scores with text features
# X = pd.concat([pd.DataFrame(X_text), df[['sentiment_score']]], axis=1)
# # Convert column names to strings
# X.columns = X.columns.astype(str)
# y = df['final_sentiment_label']

# # Train-test split and model training
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# from sklearn.metrics import accuracy_score
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# %%
df['final_sentiment_label'].unique()

# %%
df_balanced['tweet_date'] = pd.to_datetime(df_balanced['tweet_date'])
df_balanced['tweet_created'] = pd.to_datetime(df_balanced['tweet_created'])

# Extract features
df_balanced['tweet_hour'] = df_balanced['tweet_created'].dt.hour
df_balanced['tweet_day'] = df_balanced['tweet_date'].dt.day
df_balanced['tweet_month'] = df_balanced['tweet_date'].dt.month
df_balanced['tweet_weekday'] = df_balanced['tweet_date'].dt.day_name()  # Day of the week

# %%
df.columns

# %%
df_balanced

# %% [markdown]
# [Table of Contents](#Table-of-Contents)

# %% [markdown]
# # Step 4: Modelling/Analysis

# %% [markdown]
# ##  Sentiment Analysis of Resolutions

# %%
# Sentiment distribution by resolution topics
sentiment_topics = df_balanced.groupby(['resolution_topics', 'final_sentiment_label']).size().unstack(fill_value=0)

# Sentiment distribution by resolution category
sentiment_categories = df_balanced.groupby(['resolution_category', 'final_sentiment_label']).size().unstack(fill_value=0)

# %%
import matplotlib.pyplot as plt

# Overall sentiment distribution
sentiment_overall = df_balanced['final_sentiment_label'].value_counts()

# Pie chart
plt.figure(figsize=(6, 6))
sentiment_overall.plot.pie(autopct='%1.1f%%', startangle=140, colors=['#FFD700', '#1E90FF', '#FF4500'])
plt.title('Overall Sentiment Distribution')
plt.ylabel('')  # Hide y-axis label
plt.show()

# %%
import seaborn as sns

# Plot for resolution topics
plt.figure(figsize=(12, 8))
sentiment_topics.plot(kind='bar', stacked=True, colormap='viridis', width=0.8)
plt.title('Sentiment Distribution by Resolution Topics')
plt.ylabel('Count')
plt.xlabel('Resolution Topics')
plt.xticks(rotation=90, ha='right')
plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(pad=10) 
plt.show()

# %%
# Plot for resolution topics
plt.figure(figsize=(12, 8))
ax = sentiment_topics.plot(kind='bar', stacked=True, colormap='viridis', width=0.8)

# Customize x-axis labels to show every nth label
step = 10  # Show every 2nd label, adjust as needed
ax.set_xticks(range(len(sentiment_topics.index)))
ax.set_xticklabels(sentiment_topics.index, rotation=90, ha='right')
ax.set_xticks(ax.get_xticks()[::step])

# Customize title, labels, and legend
plt.title('Sentiment Distribution by Resolution Topics')
plt.ylabel('Count')
plt.xlabel('Resolution Topics')

# Adjust legend position and layout
plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(pad=10)
plt.show()

# %%
# Plot for resolution categories
plt.figure(figsize=(10, 6))
sentiment_categories.plot(kind='bar', stacked=True, colormap='coolwarm', width=0.8)
plt.title('Sentiment Distribution by Resolution Categories')
plt.ylabel('Count')
plt.xlabel('Resolution Categories')
plt.xticks(rotation=60, ha='right')
plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
df_balanced['tweet_state'], df_balanced['tweet_location'], df_balanced['tweet_coord']

# %% [markdown]
# ## Resolution Popularity by Location

# %%
# Group data by location and state
resolution_by_location = df_balanced.groupby('tweet_location')['tweet_id'].count().sort_values(ascending=False)
resolution_by_state = df_balanced.groupby('tweet_state')['tweet_id'].count().sort_values(ascending=False)

# Display the top locations
print("Top Locations for Resolutions:")
print(resolution_by_location.head(10))

print("\nTop States for Resolutions:")
print(resolution_by_state.head(10))

# %%
import folium
from folium.plugins import HeatMap

# Prepare data for HeatMap (ensure tweet_coord is non-null)
df_coords = df_balanced.dropna(subset=['tweet_coord'])
locations = df_coords['tweet_coord'].apply(eval).tolist()

# Create a base map
map_heat = folium.Map(location=[0, 0], zoom_start=2)

# Add HeatMap
HeatMap(locations).add_to(map_heat)

# Save and display the map
map_heat.save("resolution_heatmap.html")
map_heat

# %%
import plotly.express as px

# Group by state and reset index
state_data = df_balanced.groupby('tweet_state')['tweet_id'].count().reset_index()
state_data.columns = ['tweet_state', 'tweet_count']

# Plot resolution density by state
fig = px.choropleth(
    state_data,
    locations='tweet_state',
    locationmode='USA-states',
    scope='usa',
    color='tweet_count',
    color_continuous_scale='Viridis',
    title="Resolution Popularity by State"
)

# Adjust layout for a larger plot
fig.update_layout(
    width=1000,  # Set the width of the plot
    height=600,  # Set the height of the plot
    margin={"r":0, "t":50, "l":0, "b":0}  # Adjust margins
)

fig.show()

# %%
hourly_tweets = df_balanced.groupby('tweet_hour')['tweet_id'].count()
daily_tweets = df_balanced.groupby('tweet_day')['tweet_id'].count()

# %%
df_balanced.columns

# %%
import matplotlib.pyplot as plt

# Hourly trends
plt.figure(figsize=(12, 8))
hourly_tweets.plot(kind='line', marker='o', color='blue')
plt.title("Hourly Resolution Activity")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Tweets")
plt.grid()
plt.show()

# %%
# Daily trends
plt.figure(figsize=(12, 6))
daily_tweets.plot(kind='line', marker='o', color='green')
plt.title("Daily Resolution Activity")
plt.xlabel("Date")
plt.ylabel("Number of Tweets")
plt.grid()
plt.show()

# %%
import seaborn as sns

# Prepare data for heatmap
hour_day_data = df_balanced.groupby([df_balanced['tweet_created'].dt.weekday, df_balanced['tweet_hour']])['tweet_id'].count().unstack(fill_value=0)

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(hour_day_data, cmap="YlGnBu", annot=False)
plt.title("Tweet Activity Heatmap by Day and Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.show()

# %% [markdown]
# ## Impact of Retweets on Resolution Categories

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Aggregate retweet counts by resolution category
category_retweets = df_balanced.groupby('resolution_category')['retweet_count'].mean().reset_index()

# Plot average retweet counts by resolution category
plt.figure(figsize=(10, 6))
sns.barplot(x='retweet_count', y='resolution_category', data=category_retweets, palette='viridis')
plt.title('Average Retweets by Resolution Category')
plt.xlabel('Average Retweet Count')
plt.ylabel('Resolution Category')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.stripplot(x='resolution_category', y='retweet_count', data=df_balanced, jitter=True, palette='viridis')
plt.title('Retweet Distribution Across Resolution Categories')
plt.xlabel('Resolution Category')
plt.ylabel('Retweet Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Define viral threshold (top 10%)
viral_threshold = df_balanced['retweet_count'].quantile(0.9)

# Filter viral resolutions
viral_resolutions = df_balanced[df_balanced['retweet_count'] >= viral_threshold]

# Display sample viral resolutions
viral_resolutions[['resolution_category', 'text_cleaned', 'retweet_count']].head()

# %%
df_balanced.columns

# %% [markdown]
# ## Text Analysis and Clustering

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Vectorize text data
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(df_balanced['text_cleaned'].dropna())

# Apply LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(dtm)

# Display top words per topic
def display_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[-n_top_words:]]))
        print("\n")

display_topics(lda, vectorizer.get_feature_names_out(), 10)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df_balanced['text_cleaned'].dropna())

# Apply K-means
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)

# Add cluster labels to DataFrame
df_balanced['cluster'] = clusters

# Visualize clusters using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(tfidf_matrix.toarray())

plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette='viridis', s=50)
plt.title('K-means Clustering of Resolutions')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

# %%
!pip install wordcloud
from collections import Counter
from wordcloud import WordCloud

# Generate word frequency
all_words = " ".join(df_balanced['text_cleaned'].dropna())
word_freq = Counter(all_words.split())

# Plot WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %% [markdown]
# ## Gender and Region-Based Preferences

# %%
# Aggregation for resolution topics by gender
gender_topics = df_balanced.groupby(['gender', 'resolution_topics']).size().unstack().fillna(0)

# Aggregation for resolution categories by region
region_categories = df_balanced.groupby(['tweet_region', 'resolution_category']).size().unstack().fillna(0)

# %%
import plotly.graph_objects as go

# Example for two genders
radar_data = gender_topics.T

fig = go.Figure()
for gender in radar_data.columns:
    fig.add_trace(go.Scatterpolar(
        r=radar_data[gender],
        theta=radar_data.index,
        fill='toself',
        name=gender
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    title="Gender-Based Resolution Topics"
)
fig.show()

# %%
import plotly.graph_objects as go

# Step 1: Filter for top 5 resolution topics by frequency
top_topics = gender_topics.sum(axis=0).sort_values(ascending=False).head(5).index
filtered_data = gender_topics[top_topics].T

# Step 2: Create radar chart
fig = go.Figure()
for gender in filtered_data.columns:
    fig.add_trace(go.Scatterpolar(
        r=filtered_data[gender],
        theta=filtered_data.index,
        fill='toself',
        name=gender
    ))

# Step 3: Update layout for clarity
fig.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    title="Gender-Based Resolution Topics (Top 5)"
)
fig.show()

# %%
region_categories.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='viridis')
plt.title('Resolution Categories by Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# [Table of Contents](#Table-of-Contents)

# %% [markdown]
# # Step 5: Evaluation

# %%
# Check the distribution of top resolution topics
top_resolution_counts = df_balanced['resolution_topics'].value_counts()
print("Top Resolution Topics:\n", top_resolution_counts.head(10))

# Visualize
plt.figure(figsize=(10, 6))
sns.barplot(x=top_resolution_counts.head(10).values, y=top_resolution_counts.head(10).index, palette='coolwarm')
plt.title("Top 10 Resolution Topics - Validation")
plt.xlabel("Count")
plt.ylabel("Resolution Topics")
plt.tight_layout()
plt.show()

# %%
# Correlation between retweet_count and sentiment polarity
correlation = df_balanced['retweet_count'].corr(df_balanced['final_sentiment_label'].apply(lambda x: 1 if x == 'Positive' else (-1 if x == 'Negative' else 0)))
print(f"Correlation between Retweets and Sentiment: {correlation:.2f}")

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_balanced['retweet_count'], y=df_balanced['final_sentiment_label'].apply(lambda x: 1 if x == 'Positive' else (-1 if x == 'Negative' else 0)),
                hue=df_balanced['final_sentiment_label'], palette="Set2", alpha=0.7)
plt.title("Retweet Count vs Sentiment")
plt.xlabel("Retweet Count")
plt.ylabel("Sentiment Polarity")
plt.legend(title="Sentiment", loc='best')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Correlation Between Retweets and Sentiment (0.03):
# * A correlation of 0.03 between retweets and sentiment is very weak, essentially indicating almost no relationship between the two variables. This suggests that:
# 
# * Minimal Influence: The sentiment of a tweet doesn't significantly affect the number of retweets. This might happen in cases where retweets are driven by factors other than sentiment, such as the popularity of the topic, user engagement, or even the presence of specific hashtags.
# 
# * Contextual Factors: Sentiment might not be a strong predictor of engagement in the form of retweets. For example, neutral or even negative sentiments could generate more retweets if the tweet resonates with a broader audience or if it is humorous, controversial, or timely.

# %% [markdown]
# ## Validate Sentiment Analysis

# %%
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define expected sentiment categories
categories = ['Positive', 'Neutral', 'Negative']

# Create a manually labeled subset for validation
# Replace `df` with your actual DataFrame containing the sentiment analysis results
validation_set = df_balanced.sample(50).copy()  # Replace with your labeled subset
validation_set['manual_sentiment'] = [
    'Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral',
    'Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral',
    'Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral',
    'Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral',
    'Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral',
    'Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral',
    'Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral',
    'Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral',
    'Positive', 'Negative'
]  # Add as many labels as needed

# Ensure that both columns are categorical
validation_set['final_sentiment_label'] = pd.Categorical(validation_set['final_sentiment_label'], categories=categories)
validation_set['manual_sentiment'] = pd.Categorical(validation_set['manual_sentiment'], categories=categories)


# Make sure there are no values outside of the defined categories in both columns
validation_set = validation_set[validation_set['final_sentiment_label'].isin(categories)]
validation_set = validation_set[validation_set['manual_sentiment'].isin(categories)]

# validation_set['sentiment'] = validation_set['sentiment'].fillna('Neutral')  # Replace NaN with 'Neutral'
# validation_set['manual_sentiment'] = validation_set['manual_sentiment'].fillna('Neutral')  # Same for manual sentiment

# Evaluate accuracy
accuracy = accuracy_score(validation_set['manual_sentiment'], validation_set['final_sentiment_label'])
print(f"Sentiment Analysis Accuracy: {accuracy:.2%}")

# Generate Confusion Matrix
conf_matrix = confusion_matrix(
    validation_set['manual_sentiment'], validation_set['final_sentiment_label'], 
    labels=categories
)

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=categories)
disp.plot(cmap='Blues', values_format='d')
plt.title("Sentiment Analysis Confusion Matrix")
plt.show()

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score


# Calculate Precision, Recall, and F1-Score for each class
precision = precision_score(validation_set['manual_sentiment'], validation_set['final_sentiment_label'], 
                            average=None, labels=categories)
recall = recall_score(validation_set['manual_sentiment'], validation_set['final_sentiment_label'], 
                      average=None, labels=categories)
f1 = f1_score(validation_set['manual_sentiment'], validation_set['final_sentiment_label'], 
              average=None, labels=categories)

# Display Precision, Recall, and F1-Score
for idx, category in enumerate(categories):
    print(f"\n{category} Metrics:")
    print(f"Precision: {precision[idx]:.2f}")
    print(f"Recall: {recall[idx]:.2f}")
    print(f"F1-Score: {f1[idx]:.2f}")

# If you want overall averages, you can also print them:
print("\nAverage Precision: {:.2f}".format(precision.mean()))
print("Average Recall: {:.2f}".format(recall.mean()))
print("Average F1-Score: {:.2f}".format(f1.mean()))

# %% [markdown]
# ### Review and Interpretation:
# 
# ### Average Precision (0.46)
# Precision measures the proportion of correctly predicted positive samples out of all samples predicted as positive.
# An average precision of 0.46 means that 46% of your model's positive predictions are correct.
# 
# #### Implications:
# * The model may have a relatively high number of false positives.
# * If precision is critical (e.g., for fraud detection), you may need to adjust thresholds or retrain with different techniques.
# 
# ### Average Recall (0.44)
# Recall measures the proportion of actual positive samples correctly identified by the model.
# A recall of 0.44 means that the model correctly identifies 44% of the actual positive samples. 
# 
# #### Implications:
# * The model is missing 56% of the actual positive cases.
# * If catching all positive cases is essential (e.g., in medical diagnoses), recall should be improved.
# 
# ### Average F1-Score (0.44)
# The F1-Score is the harmonic mean of precision and recall, balancing the two metrics.
# An average F1-score of 0.44 indicates a mediocre balance between precision and recall.
# 
# #### Implications:
# * The model struggles equally with both precision and recall.
# * This score might indicate an opportunity to improve your model through better preprocessing, hyperparameter tuning, or changing the algorithm.

# %%
print(validation_set['manual_sentiment'].isna().sum())  # Check for NaN in manual_sentiment
print(validation_set['final_sentiment_label'].isna().sum())  # Check for NaN in sentiment

# %%
print(validation_set.shape)  # Check if the validation set has rows remaining after filtering


# %%
print(validation_set['final_sentiment_label'].unique())  # Check the unique values in 'sentiment'
print(validation_set['manual_sentiment'].unique())  # Check the unique values in 'manual_sentiment'


# %%
df['final_sentiment_label'].unique()

# %% [markdown]
# ## Validate Topic Modeling

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Prepare text data
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
text_matrix = vectorizer.fit_transform(df_balanced['text'].dropna())

# Perform LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(text_matrix)

# Display top words per topic
words = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx + 1}: ", [words[i] for i in topic.argsort()[-10:]])

# %% [markdown]
# ## Overall Evaluation Report

# %%
print("Evaluation Summary:")
print("- Insights are actionable and align with expectations.")
print("- Visualizations clearly highlight key trends.")
print("- Sentiment analysis accuracy: {:.2%}".format(accuracy))
print("- Topics generated by LDA appear coherent and relevant.")

# %% [markdown]
# #### Evaluation of Accuracy Score
# * Acceptance of 44% Accuracy
# * Dataset is small, noisy, or highly domain-specific.(sarcasm-included)
# * The features or model are basic, and this is an early-stage prototype.
# 
# #### Suggestions to Improve
# * Preprocessing: Refine text cleaning and add contextual features like n-grams or embeddings.
# * Features: Incorporate TF-IDF, Word2Vec, or BERT embeddings.
# * Model: Switch from VADER to machine learning or fine-tuned transformer models.
# * Hyperparameter Tuning: Experiment with model hyperparameters for better results.
# * Data: Increase dataset size or use data augmentation techniques.

# %% [markdown]
# # REFERENCES

# %% [markdown]
# * https://www.kaggle.com/datasets/andrewmvd/new-years-resolutions/data
# * https://www.kaggle.com/code/selvynallotey/twitter-exploratory-sentiment-analysis
# * https://www.kaggle.com/code/andrewmvd/exploring-new-year-s-resolutions

# %% [markdown]
# [Table of Contents](#Table-of-Contents)


