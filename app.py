"""
Amazon Reviews Sentiment Analyzer
----------------------------------
Production-ready Streamlit app for scraping Amazon reviews and performing sentiment analysis.
Author: Senior Python Engineer
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
import re
from collections import Counter
import time

# Sentiment Analysis Libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    st.warning("⚠️ VADER non installato. Installa con: `pip install vaderSentiment`")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    st.warning("⚠️ TextBlob non installato. Installa con: `pip install textblob`")

# Wordcloud (optional)
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


# ==================== CONFIGURATION ====================

st.set_page_config(
    page_title="Amazon Reviews Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF9900;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================== APIFY INTEGRATION ====================

def scrape_reviews_apify(product_url: str, max_reviews: int = 100) -> Optional[List[Dict]]:
    """
    Scrape Amazon reviews using Apify Actor (web_wanderer/amazon-reviews-extractor).
    
    Args:
        product_url: Amazon product URL
        max_reviews: Maximum number of reviews to scrape
        
    Returns:
        List of review dictionaries or None if error
    """
    try:
        from apify_client import ApifyClient
        
        # Initialize Apify client (richiede API token)
        api_token = st.secrets.get("APIFY_API_TOKEN", None)
        
        if not api_token:
            st.error("❌ APIFY_API_TOKEN non configurato in secrets. Usa la modalità fallback.")
            return None
        
        client = ApifyClient(api_token)
        
        # Prepare Actor input
        run_input = {
            "productUrls": [product_url],
            "maxReviews": max_reviews,
            "scrapeReviewerInfo": True,
        }
        
        # Run Actor
        with st.spinner("🔄 Scraping recensioni da Amazon via Apify..."):
            run = client.actor("web_wanderer/amazon-reviews-extractor").call(run_input=run_input)
        
        # Fetch results
        reviews = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            reviews.append({
                'title': item.get('reviewTitle', ''),
                'text': item.get('reviewText', ''),
                'rating': item.get('stars', 0),
                'date': item.get('reviewDate', ''),
                'verified': item.get('verifiedPurchase', False),
                'reviewer': item.get('reviewerName', 'Anonymous')
            })
        
        return reviews
    
    except Exception as e:
        st.error(f"❌ Errore Apify: {str(e)}")
        return None


# ==================== FALLBACK SCRAPING ====================

def scrape_reviews_fallback(product_url: str, max_reviews: int = 50) -> Optional[List[Dict]]:
    """
    Fallback: simulated scraping for demo purposes.
    In production, implement with BeautifulSoup/Selenium or use Apify.
    
    NOTE: Real scraping requires handling Amazon's anti-bot measures.
    """
    st.warning("⚠️ Modalità DEMO: utilizzando recensioni simulate. Configura Apify per dati reali.")
    
    # Simulated reviews for demo
    import random
    
    sample_reviews = [
        {"title": "Ottimo prodotto!", "text": "Sono molto soddisfatto dell'acquisto. Qualità eccellente.", "rating": 5},
        {"title": "Buon rapporto qualità/prezzo", "text": "Funziona bene, consegna veloce.", "rating": 4},
        {"title": "Deludente", "text": "Non corrisponde alla descrizione. Materiale scadente.", "rating": 2},
        {"title": "Fantastico!", "text": "Supera le aspettative. Lo consiglio vivamente.", "rating": 5},
        {"title": "Nella media", "text": "Va bene ma niente di speciale.", "rating": 3},
        {"title": "Pessimo", "text": "Si è rotto dopo una settimana. Sconsigliato.", "rating": 1},
        {"title": "Soddisfatto", "text": "Buon prodotto, arrivato nei tempi previsti.", "rating": 4},
        {"title": "Non male", "text": "Discreto ma potrebbe essere migliore.", "rating": 3},
        {"title": "Eccellente!", "text": "Migliore del previsto. Cinque stelle meritate.", "rating": 5},
        {"title": "Problematico", "text": "Difficile da usare, istruzioni poco chiare.", "rating": 2},
    ]
    
    # Generate more reviews
    reviews = []
    for i in range(min(max_reviews, 50)):
        base = random.choice(sample_reviews)
        reviews.append({
            'title': base['title'],
            'text': base['text'],
            'rating': base['rating'],
            'date': f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            'verified': random.choice([True, False]),
            'reviewer': f"User_{i+1}"
        })
    
    time.sleep(1)  # Simulate scraping delay
    return reviews


# ==================== SENTIMENT ANALYSIS ====================

@st.cache_data(show_spinner=False)
def analyze_sentiment_vader(text: str) -> Dict[str, float]:
    """Analyze sentiment using VADER."""
    if not VADER_AVAILABLE:
        return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
    
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores


@st.cache_data(show_spinner=False)
def analyze_sentiment_textblob(text: str) -> Dict[str, float]:
    """Analyze sentiment using TextBlob."""
    if not TEXTBLOB_AVAILABLE:
        return {'polarity': 0, 'subjectivity': 0}
    
    blob = TextBlob(text)
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }


def classify_sentiment(compound_score: float) -> str:
    """Classify sentiment based on VADER compound score."""
    if compound_score >= 0.05:
        return "Positivo"
    elif compound_score <= -0.05:
        return "Negativo"
    else:
        return "Neutro"


def process_reviews_sentiment(reviews: List[Dict]) -> pd.DataFrame:
    """Process reviews and add sentiment analysis."""
    df = pd.DataFrame(reviews)
    
    # Combine title and text for analysis
    df['full_text'] = df['title'] + ". " + df['text']
    
    # VADER sentiment
    if VADER_AVAILABLE:
        vader_scores = df['full_text'].apply(analyze_sentiment_vader)
        df['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
        df['vader_pos'] = vader_scores.apply(lambda x: x['pos'])
        df['vader_neg'] = vader_scores.apply(lambda x: x['neg'])
        df['vader_neu'] = vader_scores.apply(lambda x: x['neu'])
        df['sentiment'] = df['vader_compound'].apply(classify_sentiment)
    
    # TextBlob sentiment
    if TEXTBLOB_AVAILABLE:
        tb_scores = df['full_text'].apply(analyze_sentiment_textblob)
        df['textblob_polarity'] = tb_scores.apply(lambda x: x['polarity'])
        df['textblob_subjectivity'] = tb_scores.apply(lambda x: x['subjectivity'])
    
    return df


# ==================== VISUALIZATIONS ====================

def create_sentiment_distribution(df: pd.DataFrame):
    """Create sentiment distribution pie chart."""
    sentiment_counts = df['sentiment'].value_counts()
    
    colors = {
        'Positivo': '#28a745',
        'Neutro': '#ffc107',
        'Negativo': '#dc3545'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        marker=dict(colors=[colors.get(label, '#6c757d') for label in sentiment_counts.index]),
        hole=0.4,
        textposition='inside',
        textinfo='percent+label'
    )])
    
    fig.update_layout(
        title="Distribuzione Sentiment",
        height=400,
        showlegend=True
    )
    
    return fig


def create_rating_distribution(df: pd.DataFrame):
    """Create rating distribution bar chart."""
    rating_counts = df['rating'].value_counts().sort_index()
    
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={'x': 'Rating (stelle)', 'y': 'Numero recensioni'},
        title="Distribuzione Rating",
        color=rating_counts.values,
        color_continuous_scale='YlOrRd'
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_sentiment_over_time(df: pd.DataFrame):
    """Create sentiment trend over time."""
    if 'date' not in df.columns:
        return None
    
    df_time = df.copy()
    df_time['date'] = pd.to_datetime(df_time['date'], errors='coerce')
    df_time = df_time.dropna(subset=['date'])
    df_time = df_time.sort_values('date')
    
    # Calculate rolling average
    df_time['vader_compound_ma'] = df_time['vader_compound'].rolling(window=5, min_periods=1).mean()
    
    fig = px.line(
        df_time,
        x='date',
        y='vader_compound_ma',
        title="Trend Sentiment nel Tempo (Media Mobile)",
        labels={'vader_compound_ma': 'Sentiment Score', 'date': 'Data'}
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutro")
    fig.update_layout(height=400)
    
    return fig


def create_wordcloud(df: pd.DataFrame, sentiment_filter: Optional[str] = None):
    """Generate word cloud from reviews."""
    if not WORDCLOUD_AVAILABLE:
        st.info("📊 Installa wordcloud per visualizzare la nuvola di parole: `pip install wordcloud`")
        return None
    
    # Filter by sentiment if specified
    if sentiment_filter:
        df_filtered = df[df['sentiment'] == sentiment_filter]
    else:
        df_filtered = df
    
    # Combine all text
    text = ' '.join(df_filtered['full_text'].astype(str))
    
    # Generate wordcloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(text)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud - {sentiment_filter if sentiment_filter else "Tutte le recensioni"}')
    
    return fig


# ==================== MAIN APP ====================

def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">📊 Amazon Reviews Sentiment Analyzer</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150)
        st.markdown("## ⚙️ Configurazione")
        
        use_apify = st.checkbox("🚀 Usa Apify (raccomandato)", value=False, 
                                help="Richiede APIFY_API_TOKEN configurato in secrets")
        
        max_reviews = st.slider("📝 Numero massimo recensioni", 10, 200, 50, step=10)
        
        st.markdown("---")
        st.markdown("### 📚 Info")
        st.info("""
        **Librerie richieste:**
```bash
        pip install streamlit pandas plotly
        pip install vaderSentiment textblob
        pip install wordcloud matplotlib
        pip install apify-client  # per Apify
```
        """)
        
        if use_apify:
            st.markdown("""
            **Setup Apify:**
            1. Crea account su [apify.com](https://apify.com)
            2. Ottieni API token
            3. Aggiungi in `.streamlit/secrets.toml`:
```toml
            APIFY_API_TOKEN = "your_token_here"
```
            """)
    
    # Main content
    st.markdown("### 🔗 Inserisci URL prodotto Amazon")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        product_url = st.text_input(
            "URL prodotto",
            placeholder="https://www.amazon.it/dp/XXXXXXXXXX",
            label_visibility="collapsed"
        )
    
    with col2:
        analyze_button = st.button("🚀 Analizza", type="primary", use_container_width=True)
    
    # URL validation
    if analyze_button:
        if not product_url:
            st.error("❌ Inserisci un URL valido!")
            return
        
        if not re.search(r'amazon\.(com|it|co\.uk|de|fr|es)', product_url):
            st.warning("⚠️ L'URL non sembra essere un link Amazon valido.")
        
        # Scrape reviews
        if use_apify:
            reviews = scrape_reviews_apify(product_url, max_reviews)
        else:
            reviews = scrape_reviews_fallback(product_url, max_reviews)
        
        if not reviews:
            st.error("❌ Impossibile recuperare le recensioni. Riprova.")
            return
        
        # Store in session state
        st.session_state['reviews'] = reviews
        st.session_state['product_url'] = product_url
    
    # Process and display results
    if 'reviews' in st.session_state:
        reviews = st.session_state['reviews']
        
        st.success(f"✅ Scaricate {len(reviews)} recensioni!")
        
        # Perform sentiment analysis
        with st.spinner("🧠 Analisi sentiment in corso..."):
            df_results = process_reviews_sentiment(reviews)
        
        # Store processed dataframe
        st.session_state['df_results'] = df_results
        
        # === METRICS ===
        st.markdown("---")
        st.markdown("## 📈 Metriche Principali")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            avg_rating = df_results['rating'].mean()
            st.metric("⭐ Rating Medio", f"{avg_rating:.2f}/5")
        
        with col2:
            if VADER_AVAILABLE:
                avg_sentiment = df_results['vader_compound'].mean()
                st.metric("😊 Sentiment Medio", f"{avg_sentiment:.3f}")
        
        with col3:
            positive_pct = (df_results['sentiment'] == 'Positivo').sum() / len(df_results) * 100
            st.metric("✅ Recensioni Positive", f"{positive_pct:.1f}%")
        
        with col4:
            negative_pct = (df_results['sentiment'] == 'Negativo').sum() / len(df_results) * 100
            st.metric("❌ Recensioni Negative", f"{negative_pct:.1f}%")
        
        with col5:
            verified_pct = df_results['verified'].sum() / len(df_results) * 100
            st.metric("✔️ Acquisti Verificati", f"{verified_pct:.1f}%")
        
        # === VISUALIZATIONS ===
        st.markdown("---")
        st.markdown("## 📊 Visualizzazioni")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "⏰ Trend", "☁️ Word Cloud", "📋 Dati"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_sentiment = create_sentiment_distribution(df_results)
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            with col2:
                fig_rating = create_rating_distribution(df_results)
                st.plotly_chart(fig_rating, use_container_width=True)
        
        with tab2:
            if VADER_AVAILABLE:
                fig_trend = create_sentiment_over_time(df_results)
                if fig_trend:
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("📅 Dati temporali non disponibili")
        
        with tab3:
            if WORDCLOUD_AVAILABLE:
                sentiment_filter = st.selectbox(
                    "Filtra per sentiment",
                    ['Tutte', 'Positivo', 'Neutro', 'Negativo']
                )
                
                filter_val = None if sentiment_filter == 'Tutte' else sentiment_filter
                fig_wc = create_wordcloud(df_results, filter_val)
                
                if fig_wc:
                    st.pyplot(fig_wc)
        
        with tab4:
            st.markdown("### 📋 Tabella Completa")
            
            # Display options
            col1, col2 = st.columns([1, 3])
            
            with col1:
                show_sentiment = st.selectbox(
                    "Filtra per sentiment",
                    ['Tutti', 'Positivo', 'Neutro', 'Negativo']
                )
            
            # Filter dataframe
            if show_sentiment != 'Tutti':
                df_display = df_results[df_results['sentiment'] == show_sentiment]
            else:
                df_display = df_results
            
            # Select columns to display
            display_cols = ['title', 'text', 'rating', 'sentiment', 'vader_compound', 'date', 'verified']
            display_cols = [col for col in display_cols if col in df_display.columns]
            
            st.dataframe(
                df_display[display_cols].style.background_gradient(
                    subset=['vader_compound'] if 'vader_compound' in display_cols else [],
                    cmap='RdYlGn',
                    vmin=-1,
                    vmax=1
                ),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Scarica CSV",
                data=csv,
                file_name="amazon_reviews_sentiment.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>💡 <strong>Tip:</strong> Per migliori risultati, usa Apify con un account premium per evitare rate limits.</p>
        <p>Developed with ❤️ using Streamlit | <a href='https://apify.com'>Powered by Apify</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
