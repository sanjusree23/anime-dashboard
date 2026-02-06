import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# PAGE CONFIG (MUST BE FIRST)
# ===============================
st.set_page_config(page_title="Anime Analysis Dashboard", layout="wide")

# ===============================
# DATA LOADING & CLEANING (NO UI)
# ===============================
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("C:\\Users\\sanju sree\\Downloads\\anime_dashboard\\anime.csv")

    # Keep relevant columns
    cols = ['title','type','episodes','members','score',
            'popularity','rank','synopsis','startdate']
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols]

    # Clean numeric columns
    numeric_cols = ['score','episodes','members','popularity','rank']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Rename score -> rating
    if 'score' in df.columns:
        df = df.rename(columns={'score': 'rating'})
        df = df.dropna(subset=['rating'])

    # Episode length categories
    def episode_category(ep):
        if pd.isna(ep): 
            return 'Unknown'
        if ep <= 13: 
            return 'Short'
        elif ep <= 50: 
            return 'Medium'
        else: 
            return 'Long'

    if 'episodes' in df.columns:
        df['episodelength'] = df['episodes'].apply(episode_category)

    # Year extraction
    if 'startdate' in df.columns:
        df['year'] = pd.to_datetime(df['startdate'], errors='coerce').dt.year

    # Popularity class (safe qcut)
    if 'members' in df.columns:
        df['popclass'] = pd.qcut(
            df['members'], 
            3, 
            labels=['Low','Medium','High'], 
            duplicates='drop'
        )

    return df


# ===============================
# LOAD DATA
# ===============================
df = load_and_clean_data()

# ===============================
# HEADER
# ===============================
st.title("🎌 Complete Anime Analysis Dashboard")
st.markdown("**All your notebook analysis in one interactive dashboard**")

# ===============================
# DATASET OVERVIEW METRICS
# ===============================
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Columns", len(df.columns))
col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ===============================
# SIDEBAR FILTERS
# ===============================
st.sidebar.header("🔍 Filters")

type_options = sorted(df['type'].dropna().unique().tolist())
type_filter = st.sidebar.multiselect(
    "Type", 
    options=['All'] + type_options, 
    default=['All']
)

# Safe year slider
if 'year' in df.columns and df['year'].notna().any():
    year_min = int(df['year'].dropna().min())
    year_max = int(df['year'].dropna().max())

    year_filter = st.sidebar.slider(
        "Year", 
        min_value=year_min, 
        max_value=year_max, 
        value=(year_min, year_max)
    )
else:
    year_filter = None

# ===============================
# APPLY FILTERS
# ===============================
filtered_df = df.copy()

if year_filter is not None:
    filtered_df = filtered_df[
        filtered_df['year'].between(*year_filter)
    ]

if 'All' not in type_filter:
    filtered_df = filtered_df[
        filtered_df['type'].isin(type_filter)
    ]

# Empty data protection
if filtered_df.empty:
    st.warning("⚠️ No data available for selected filters.")
    st.stop()

st.markdown(f"**📈 Showing {len(filtered_df):,} anime records**")

# ===============================
# KPI METRICS
# ===============================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Rating", f"{filtered_df['rating'].mean():.2f}")
col2.metric("Top Rating", f"{filtered_df['rating'].max():.2f}")
col3.metric("Most Members", f"{filtered_df['members'].max():,}")
col4.metric("Avg Episodes", f"{filtered_df['episodes'].mean():.0f}")

# ===============================
# Q1: RATING DISTRIBUTION
# ===============================
st.subheader("📊 Q1: Rating Distribution")

col1, col2 = st.columns([3,1])
with col1:
    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.histplot(filtered_df['rating'], bins=30, kde=True, ax=ax1)
    ax1.axvline(
        filtered_df['rating'].mean(), 
        color='red', 
        linestyle='--', 
        label=f"Mean: {filtered_df['rating'].mean():.2f}"
    )
    ax1.legend()
    st.pyplot(fig1)

with col2:
    st.metric("Mean Rating", f"{filtered_df['rating'].mean():.2f}")
    st.metric("Median Rating", f"{filtered_df['rating'].median():.2f}")

# ===============================
# Q2: AVERAGE RATING BY TYPE
# ===============================
st.subheader("🏆 Q2: Average Rating by Type")

avg_type = filtered_df.groupby('type')['rating'].mean().sort_values(ascending=False)

col1, col2 = st.columns(2)
with col1:
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.barplot(x=avg_type.values, y=avg_type.index, ax=ax2)
    ax2.set_title("Average Rating by Anime Type")
    ax2.set_xlabel("Rating")
    st.pyplot(fig2)

with col2:
    st.dataframe(avg_type.round(3))

# ===============================
# Q3: POPULARITY VS RATING
# ===============================
st.subheader("⭐ Q3: Popularity vs Rating")

fig3, ax3 = plt.subplots(figsize=(12,6))
sns.scatterplot(
    data=filtered_df, 
    x='members', 
    y='rating', 
    hue='type', 
    size='episodes', 
    sizes=(50,300), 
    alpha=0.6, 
    ax=ax3
)
ax3.set_xscale('log')
ax3.set_xlabel("Members (Log Scale)")
ax3.set_title("Popularity (Members) vs Rating")
st.pyplot(fig3)

# ===============================
# Q4: EPISODES VS RATING
# ===============================
st.subheader("📺 Q4: Episodes vs Rating")

col1, col2 = st.columns(2)
with col1:
    fig4, ax4 = plt.subplots(figsize=(8,6))
    sns.scatterplot(
        data=filtered_df, 
        x='episodes', 
        y='rating', 
        hue='episodelength', 
        ax=ax4
    )

    # Safe correlation
    corr_df = filtered_df[['episodes','rating']].dropna()
    corr = corr_df.corr().iloc[0,1] if len(corr_df) > 1 else np.nan

    ax4.set_title(f"Episodes vs Rating (Corr: {corr:.3f})")
    st.pyplot(fig4)

with col2:
    episode_stats = (
        filtered_df
        .groupby('episodelength')['rating']
        .agg(['mean','count'])
        .round(2)
    )
    st.dataframe(episode_stats)

# ===============================
# TOP 10 ANIME
# ===============================
st.subheader("🥇 Top 10 Highest Rated Anime")

st.dataframe(
    filtered_df
    .nlargest(10, 'rating')[['title','rating','members','type']]
    .style.format({'rating': '{:.2f}'})
)

# ===============================
# FULL DATA PREVIEW
# ===============================
with st.expander("📋 Full Dataset Preview"):
    st.dataframe(filtered_df.head(1000), use_container_width=True)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown("*Complete EDA analysis from your Jupyter notebook, now interactive! 💫*")
