import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Page Config
st.set_page_config(page_title="Spotify Breakout Predictor", layout="wide")
st.title("🎵 Spotify 2026: Breakout Artist Predictor")
st.markdown("This dashboard identifies upcoming artists by comparing **Social Virality** (TikTok) against **Spotify Popularity**.")

# 2. Load the cleaned data (Assuming you've exported df_cleaned to CSV)
@st.cache_data
def load_data():
    df = pd.read_csv('df_cleaned.csv') # You'll need to export your df_cleaned first
    return df

df = load_data()

# 3. Sidebar Filters
st.sidebar.header("Filter Artists")
min_popularity = st.sidebar.slider("Max Spotify Popularity", 0, 100, 75)
top_n = st.sidebar.number_input("Show Top N Artists", 5, 50, 10)

# 4. Key Metrics Row
col1, col2, col3 = st.columns(3)
col1.metric("Total Tracks Analyzed", len(df))
col2.metric("Breakout Candidates Found", df['is_breakout'].sum())
col3.metric("Model Precision", "90%")

# 5. Visualizations
st.header("Exploratory Data Insights")
tab1, tab2, tab3 = st.tabs(["Virality Map", "Feature Importance", "Correlations"])

with tab1:
    # 1. Fallback: Calculate Virality_Ratio if it's missing from the CSV
    if 'Virality_Ratio' not in df.columns:
        df['Virality_Ratio'] = df['TikTok Views'] / (df['Spotify Streams'] + 1)
    
    # 2. Fallback: Ensure is_breakout exists
    if 'is_breakout' not in df.columns:
        st.error("Column 'is_breakout' missing from CSV! Please re-export from Colab.")
    else:
        df['is_breakout'] = df['is_breakout'].astype(str)
        
        fig = px.scatter(
            df, 
            x='Spotify Popularity', 
            y='Virality_Ratio', 
            color='is_breakout', 
            hover_name='Track',
            log_y=True, 
            title="TikTok Virality vs. Spotify Popularity",
            color_discrete_map={'0': 'blue', '1': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
with tab2:
    # (Static image or pre-calculated importance)
    st.write("TikTok engagement is the #1 predictor of breakout success.")
    st.image('feature_importance.png') # You can save your plot as a PNG in Colab

# 6. The Watchlist (The "Product")
st.header(f"🚀 Top {top_n} Predicted Breakout Artists")
watchlist = df[df['Spotify Popularity'] <= min_popularity]
watchlist = watchlist.sort_values(by='Breakout_Probability', ascending=False).head(top_n)

st.table(watchlist[['Track', 'Artist', 'Spotify Popularity', 'Breakout_Probability']])
