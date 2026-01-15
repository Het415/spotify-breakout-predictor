# spotify-breakout-predictor
🎵 Spotify Breakout Artist Predictor
Project Overview
This project uses Machine Learning to solve the "A&R Discovery" problem: identifying "breakout" artists before they become global superstars. While traditional metrics focus on current stream counts, this model leverages cross-platform signals (TikTok virality and Shazam curiosity) to predict future success on Spotify.

Key Findings
Predictive Power: The model achieves 90% Precision and 94% Recall in identifying breakout candidates.

Leading Indicators: TikTok engagement (Views, Likes, and Posts) is 10x more predictive of an upcoming breakout than traditional radio AirPlay or playlist reach.

Signal Independence: There is a very low correlation (0.04) between TikTok Views and Spotify Streams, proving that social virality is an independent "hidden" signal that requires ML to decode.

The Data Science Pipeline
Data Acquisition: Processed a dataset of the "Most Streamed Songs of 2024" containing 4,500+ tracks with metrics from Spotify, TikTok, YouTube, and Shazam.

Preprocessing & Cleaning:

Handled 1,000+ missing values in social media columns.

Engineered a Stream Velocity metric to normalize popularity against time since release.

Cleaned numeric data formatted as strings with commas.

Exploratory Data Analysis (EDA): Created a "Breakout Map" to visualize the 896 candidates with high social virality but low current Spotify popularity.

Modeling: Trained a Random Forest Classifier using stratified sampling to handle class imbalance.

Tech Stack
Language: Python

Environment: Google Colab

Libraries: Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn

Deployment: Streamlit Cloud

How to Use
Filter artists by their maximum current "Spotify Popularity" score.

The dashboard will display a "Watchlist" of artists with the highest Breakout Probability according to the model.
