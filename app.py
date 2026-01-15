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
    st.write("### What Drives a Breakout?")
    st.write("Our Random Forest model identified these as the strongest signals for predicting a breakout artist.")
    
    # These values come from your model's feature_importances_ results
    importance_data = pd.DataFrame({
        'Signal': ['TikTok Views', 'TikTok Likes', 'TikTok Posts', 'Spotify Playlist Reach', 'Shazam Counts'],
        'Importance': [0.42, 0.23, 0.11, 0.08, 0.02] 
    })
    
    fig_imp = px.bar(
        importance_data, 
        x='Importance', 
        y='Signal', 
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    st.plotly_chart(fig_imp, use_container_width=True)
with tab3:
    st.write("### Platform Correlation Heatmap")
    st.write("This heatmap shows how different platform signals relate to one another.")
    
    # 1. Select the numeric columns for correlation
    corr_cols = ['Spotify Streams', 'Spotify Popularity', 'TikTok Views', 
                 'Shazam Counts', 'YouTube Views', 'Stream_Velocity', 'Track Score']
    
    # 2. Calculate correlation matrix
    corr_matrix = df[corr_cols].corr()
    
    # 3. Create an interactive heatmap using Plotly
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title="Relationship Between Music Platforms"
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
# 6. The Watchlist (The "Product")
st.header(f"🚀 Top {top_n} Predicted Breakout Artists")
watchlist = df[df['Spotify Popularity'] <= min_popularity]
watchlist = watchlist.sort_values(by='Breakout_Probability', ascending=False).head(top_n)

st.table(watchlist[['Track', 'Artist', 'Spotify Popularity', 'Breakout_Probability']])
