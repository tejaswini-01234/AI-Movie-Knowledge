import os
import glob
import re
import html
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Movie Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .movie-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
    }
    .movie-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .badge-genre {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 9999px;
        margin-right: 6px;
        margin-bottom: 4px;
        border: 1px solid #DBEAFE;
    }
    .badge-rating {
        display: inline-block;
        background-color: #FEF3C7;
        color: #B45309;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
        margin-right: 8px;
    }
    .badge-year {
        display: inline-block;
        background-color: #F1F5F9;
        color: #475569;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        margin-right: 8px;
    }
    .badge-source {
        display: inline-block;
        background-color: #F3F4F6;
        color: #6B7280;
        font-family: monospace;
        font-size: 0.72rem;
        padding: 2px 7px;
        border-radius: 4px;
    }
    .movie-overview {
        font-size: 0.93rem;
        line-height: 1.5;
        color: #334155;
        margin-top: 10px;
    }
    .pill-btn {
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Knowledge Base Loading & Indexing
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KNOWLEDGE_DIR = os.path.join(BASE_DIR, "movies")
if not os.path.exists(DEFAULT_KNOWLEDGE_DIR):
    # Fallback to current working directory
    DEFAULT_KNOWLEDGE_DIR = os.path.join(os.getcwd(), "movies")

@st.cache_data(show_spinner="Loading movie knowledge files from disk...")
def load_movie_knowledge(folder_path):
    """
    Reads and parses all movie markdown files from the specified knowledge folder.
    Extracts title, movie ID, release date, year, genres, rating, vote count, and overview.
    """
    if not os.path.exists(folder_path):
        return pd.DataFrame(), f"Knowledge folder not found: {folder_path}"
    
    file_paths = glob.glob(os.path.join(folder_path, "*.md"))
    if not file_paths:
        return pd.DataFrame(), f"No markdown (.md) files found in {folder_path}"

    records = []
    for fp in file_paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                content = f.read()

            # 1. Title: Heading 1 (# Title) or frontmatter
            m_h1 = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            if m_h1:
                title = m_h1.group(1).strip()
            else:
                m_fm = re.search(r'title:\s*\"?(.*?)\"?$', content, re.MULTILINE)
                title = m_fm.group(1).strip() if m_fm else os.path.splitext(os.path.basename(fp))[0]

            # 2. Movie ID
            m_id = re.search(r"## Movie ID\s*\n(\d+)", content)
            movie_id = m_id.group(1).strip() if m_id else ""

            # 3. Release Date & Year
            m_date = re.search(r"## Release Date\s*\n([^\n#]+)", content)
            release_date = m_date.group(1).strip() if m_date else "Unknown"
            m_yr = re.search(r"(\d{4})", release_date)
            year = int(m_yr.group(1)) if m_yr else None

            # 4. Language
            m_lang = re.search(r"## Language\s*\n([^\n#]+)", content)
            language = m_lang.group(1).strip() if m_lang else "en"

            # 5. Genres
            m_gen = re.search(r"## Genres\s*\n([^\n#]*)", content)
            genres = []
            if m_gen and m_gen.group(1).strip():
                genres = [g.strip() for g in m_gen.group(1).split(",") if g.strip()]

            # 6. Rating (0.0 to 10.0)
            m_rate = re.search(r"## Rating\s*\n([0-9.]+)", content)
            rating = float(m_rate.group(1).strip()) if m_rate else 0.0

            # 7. Vote Count
            m_vc = re.search(r"## Vote Count\s*\n(\d+)", content)
            vote_count = int(m_vc.group(1).strip()) if m_vc else 0

            # 8. Overview
            m_ov = re.search(r"## Overview\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL)
            overview = m_ov.group(1).strip() if m_ov else "No overview available."

            records.append({
                "filename": os.path.basename(fp),
                "filepath": fp,
                "rel_path": os.path.relpath(fp, BASE_DIR),
                "title": title,
                "title_lower": title.lower(),
                "movie_id": movie_id,
                "release_date": release_date,
                "year": year,
                "language": language,
                "genres": genres,
                "genres_str": ", ".join(genres) if genres else "Unknown",
                "rating": rating,
                "vote_count": vote_count,
                "overview": overview,
                "raw_content": content,
                "search_text": f"{title} {' '.join(genres)} {overview}".lower()
            })
        except Exception:
            continue

    df = pd.DataFrame(records)
    
    # Calculate Bayesian Weighted Rating to balance score & vote count
    # Formula: WR = (v / (v + m)) * R + (m / (v + m)) * C
    if not df.empty:
        C = df["rating"].mean()
        m = 100  # Vote threshold baseline
        df["weighted_rating"] = df.apply(
            lambda r: (r["vote_count"] / (r["vote_count"] + m) * r["rating"]) + (m / (r["vote_count"] + m) * C),
            axis=1
        )
    return df, None

@st.cache_resource(show_spinner="Building TF-IDF semantic search index...")
def build_search_index(df):
    """Creates a TF-IDF vectorizer and matrix over movie search texts for fast semantic search."""
    if df.empty:
        return None, None
    tfidf = TfidfVectorizer(stop_words="english", max_features=12000)
    matrix = tfidf.fit_transform(df["search_text"])
    return tfidf, matrix

# ---------------------------------------------------------
# Query Understanding & Natural Language Processing
# ---------------------------------------------------------
GENRE_MAP = {
    "action": "Action",
    "action-packed": "Action",
    "adventure": "Adventure",
    "animation": "Animation",
    "animated": "Animation",
    "anime": "Animation",
    "cartoon": "Animation",
    "comedy": "Comedy",
    "comedies": "Comedy",
    "funny": "Comedy",
    "crime": "Crime",
    "documentary": "Documentary",
    "documentaries": "Documentary",
    "drama": "Drama",
    "dramas": "Drama",
    "family": "Family",
    "kids": "Family",
    "fantasy": "Fantasy",
    "foreign": "Foreign",
    "history": "History",
    "historical": "History",
    "horror": "Horror",
    "scary": "Horror",
    "spooky": "Horror",
    "music": "Music",
    "musical": "Music",
    "mystery": "Mystery",
    "mysteries": "Mystery",
    "romance": "Romance",
    "romantic": "Romance",
    "rom-com": "Romance",
    "romcom": "Romance",
    "science fiction": "Science Fiction",
    "sci-fi": "Science Fiction",
    "scifi": "Science Fiction",
    "sci fi": "Science Fiction",
    "thriller": "Thriller",
    "thrillers": "Thriller",
    "suspense": "Thriller",
    "war": "War",
    "western": "Western"
}

def parse_user_query(user_query, df):
    """
    Parses natural language queries to detect intent, movie titles,
    genres, years, ratings, and semantic keywords.
    """
    q_clean = user_query.strip()
    q_lower = q_clean.lower()
    
    # 1. Check for specific movie lookup
    detail_match = re.search(
        r"(?:tell me about|what is|info on|details of|synopsis of|overview of|plot of|about|find movie|search for)\s+[\"']?([^\"'?]+)[\"']?",
        q_lower
    )
    candidate_title = detail_match.group(1).strip() if detail_match else None
    
    # Check if direct query matches a movie title
    direct_match = df[df["title_lower"] == q_lower.strip("? \"'")]
    if not direct_match.empty:
        return {"intent": "movie_detail", "movie": direct_match.iloc[0], "query": q_clean}
        
    if candidate_title:
        # Check exact title match
        m1 = df[df["title_lower"] == candidate_title]
        if not m1.empty:
            return {"intent": "movie_detail", "movie": m1.iloc[0], "query": q_clean}
        # Check title word boundary containment
        m2 = df[df["title_lower"].str.contains(r"\b" + re.escape(candidate_title) + r"\b", regex=True)]
        if not m2.empty:
            # Select highest vote count match
            best_match = m2.sort_values(by="vote_count", ascending=False).iloc[0]
            return {"intent": "movie_detail", "movie": best_match, "query": q_clean}

    # 2. Extract Genres
    detected_genres = []
    for term, g_name in GENRE_MAP.items():
        if re.search(r"\b" + re.escape(term) + r"\b", q_lower):
            if g_name not in detected_genres:
                detected_genres.append(g_name)

    # 3. Extract Year Filters
    year_filter = None
    # after 2015, post 2015, since 2015, > 2015
    m_after = re.search(r"(?:after|post|since|later than|>|>=)\s*(\d{4})", q_lower)
    if m_after:
        year_filter = ("after", int(m_after.group(1)))
        
    # before 2000, prior to 2000, < 2000
    m_before = re.search(r"(?:before|prior to|earlier than|<|<=)\s*(\d{4})", q_lower)
    if m_before:
        year_filter = ("before", int(m_before.group(1)))
        
    # in 2010, released in 2010
    m_exact_yr = re.search(r"(?:in|released in|from year)\s*(\d{4})", q_lower)
    if m_exact_yr and not year_filter:
        year_filter = ("exact", int(m_exact_yr.group(1)))
        
    # Decades: 90s, 80s, 70s, 2000s
    m_decade = re.search(r"\b(19\d0|\d0)s\b", q_lower)
    if m_decade and not year_filter:
        dec = m_decade.group(1)
        start_yr = int(dec) if len(dec) == 4 else int("19" + dec)
        year_filter = ("range", (start_yr, start_yr + 9))

    # 4. Extract Rating Filters
    high_rating_intent = bool(
        re.search(r"\b(high|highest|top|best|greatest|acclaimed)\b", q_lower) and
        re.search(r"\b(rating|ratings|rated|movies|films|recommend)\b", q_lower)
    )
    m_rate_thresh = re.search(r"(?:rating|rated)\s*(?:above|over|higher than|>|>=)\s*([0-9.]+)", q_lower)
    min_rating = float(m_rate_thresh.group(1)) if m_rate_thresh else 0.0

    return {
        "intent": "search_filter",
        "genres": detected_genres,
        "year_filter": year_filter,
        "high_rating": high_rating_intent,
        "min_rating": min_rating,
        "query": q_clean
    }

def execute_movie_query(parsed_intent, df, tfidf, tfidf_matrix, max_results=5, ranking_mode="Weighted Rating (Quality & Popularity)"):
    """
    Executes search and filtering over the movie dataset based on parsed intent.
    Strictly uses available dataset fields: Title, Genres, Release Date, Rating, Vote Count, Overview.
    """
    if parsed_intent["intent"] == "movie_detail":
        return "detail", [parsed_intent["movie"].to_dict()], parsed_intent

    sub_df = df.copy()

    # 1. Filter by Genres
    if parsed_intent["genres"]:
        for g in parsed_intent["genres"]:
            sub_df = sub_df[sub_df["genres"].apply(lambda gl: g in gl)]

    # 2. Filter by Year
    yf = parsed_intent.get("year_filter")
    if yf:
        op, val = yf
        if op == "after":
            sub_df = sub_df[sub_df["year"] > val]
        elif op == "before":
            sub_df = sub_df[sub_df["year"] < val]
        elif op == "exact":
            sub_df = sub_df[sub_df["year"] == val]
        elif op == "range":
            sub_df = sub_df[(sub_df["year"] >= val[0]) & (sub_df["year"] <= val[1])]

    # 3. Filter by Rating Threshold
    if parsed_intent["min_rating"] > 0:
        sub_df = sub_df[sub_df["rating"] >= parsed_intent["min_rating"]]

    if sub_df.empty:
        return "no_results", [], parsed_intent

    # 4. Extract search keywords outside common query boilerplate
    stop_words = {
        "recommend", "find", "show", "movies", "movie", "films", "film", "which",
        "what", "have", "high", "rating", "ratings", "rated", "released", "after",
        "before", "top", "best", "the", "a", "an", "in", "of", "from", "with",
        "about", "tell", "me", "give", "list", "some", "good", "great"
    }
    for g in GENRE_MAP.keys():
        for token in g.split():
            stop_words.add(token)

    query_tokens = [
        w for w in re.findall(r"\b\w+\b", parsed_intent["query"].lower())
        if w not in stop_words and len(w) > 2
    ]

    # 5. Semantic Search vs Categorical Ranking
    if query_tokens and tfidf is not None and tfidf_matrix is not None:
        keyword_str = " ".join(query_tokens)
        q_vec = tfidf.transform([keyword_str])
        
        sub_indices = sub_df.index.tolist()
        sub_matrix = tfidf_matrix[sub_indices]
        sims = cosine_similarity(q_vec, sub_matrix).flatten()
        
        sub_df = sub_df.copy()
        sub_df["sim_score"] = sims
        # Combine semantic match with quality score
        sub_df["composite_score"] = (sub_df["sim_score"] * 0.65) + ((sub_df["weighted_rating"] / 10.0) * 0.35)
        ranked = sub_df.sort_values(by=["composite_score", "weighted_rating"], ascending=[False, False])
    else:
        # Sort based on user selected ranking mode
        if ranking_mode == "Absolute Rating (Raw 0-10)":
            ranked = sub_df.sort_values(by=["rating", "vote_count"], ascending=[False, False])
        elif ranking_mode == "Release Date (Newest First)":
            ranked = sub_df.sort_values(by=["release_date", "weighted_rating"], ascending=[False, False])
        elif ranking_mode == "Vote Count (Most Popular)":
            ranked = sub_df.sort_values(by=["vote_count", "rating"], ascending=[False, False])
        else: # Default: Weighted Rating
            ranked = sub_df.sort_values(by=["weighted_rating", "rating"], ascending=[False, False])

    return "list", ranked.head(max_results).to_dict("records"), parsed_intent

# ---------------------------------------------------------
# Response Generation (Grounded & Conversational)
# ---------------------------------------------------------
def generate_conversational_response(res_type, results, intent, total_found, df_len):
    """
    Constructs a grounded, factual response message.
    Strictly adheres to data contained in the markdown knowledge files.
    """
    if res_type == "no_results":
        genres_text = f" in genre(s) **{', '.join(intent['genres'])}**" if intent.get("genres") else ""
        year_text = ""
        if intent.get("year_filter"):
            op, val = intent["year_filter"]
            year_text = f" released {op} {val}"
        return (
            f"🔍 **No movies found** matching your specific criteria{genres_text}{year_text} "
            f"in the knowledge base ({df_len:,} movies indexed).\n\n"
            f"> *Note: All responses are strictly grounded in your local TMDB markdown files.* "
            f"Try adjusting your filters or asking about another genre or movie!"
        )

    if res_type == "detail":
        movie = results[0]
        return (
            f"Here is the verified dossier for **{movie['title']}** from your knowledge files:\n"
        )

    # List response
    intro_parts = []
    if intent.get("high_rating"):
        intro_parts.append("highest-rated")
    if intent.get("genres"):
        intro_parts.append(f"**{', '.join(intent['genres'])}**")
    
    intro_desc = " ".join(intro_parts) if intro_parts else "relevant"
    
    if intent.get("year_filter"):
        op, val = intent["year_filter"]
        if op == "after":
            filter_desc = f"released after **{val}**"
        elif op == "before":
            filter_desc = f"released before **{val}**"
        elif op == "exact":
            filter_desc = f"released in **{val}**"
        elif op == "range":
            filter_desc = f"from the **{val[0]}s**"
        else:
            filter_desc = ""
    else:
        filter_desc = ""

    header_msg = f"I searched your movie knowledge base ({df_len:,} files) and found "
    if filter_desc and intro_desc:
        header_msg += f"top {intro_desc} movies {filter_desc}:"
    elif filter_desc:
        header_msg += f"movies {filter_desc}:"
    elif intro_desc != "relevant":
        header_msg += f"the top {intro_desc} movies:"
    else:
        header_msg += "the most relevant movies for your query:"

    return header_msg

def render_movie_card(movie):
    """Renders an individual movie as a clean HTML/Markdown card with full metadata."""
    title = html.escape(movie["title"])
    year = f"({int(movie['year'])})" if pd.notna(movie.get("year")) else ""
    rel_date = movie.get("release_date", "Unknown")
    rating = movie.get("rating", 0.0)
    vote_count = movie.get("vote_count", 0)
    genres = movie.get("genres", [])
    overview = html.escape(movie.get("overview", "No overview provided."))
    rel_path = movie.get("rel_path", movie.get("filename", ""))

    genre_badges = "".join([f"<span class='badge-genre'>{html.escape(g)}</span>" for g in genres])
    if not genre_badges:
        genre_badges = "<span class='badge-genre'>General</span>"

    card_html = f"""
    <div class="movie-card">
        <div class="movie-title">🎬 {title} <span class="badge-year">{year}</span></div>
        <div style="margin-bottom: 8px;">
            <span class="badge-rating">⭐ {rating:.1f} / 10 ({vote_count:,} votes)</span>
            <span class="badge-year">📅 {rel_date}</span>
            <span class="badge-source">📄 {rel_path}</span>
        </div>
        <div style="margin-bottom: 8px;">{genre_badges}</div>
        <div class="movie-overview"><strong>Overview:</strong> {overview}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    
    with st.expander(f"🔎 View Raw Markdown File Content ({movie.get('filename')})", expanded=False):
        raw_code = movie.get("raw_content", "")
        if raw_code:
            st.code(raw_code, language="markdown")
        else:
            st.info("Raw content not loaded.")

# ---------------------------------------------------------
# Sidebar: Knowledge Base Explorer & Configuration
# ---------------------------------------------------------
st.sidebar.title("🎬 Movie Knowledge Base")

# Verify Knowledge Folder Path
knowledge_folder = st.sidebar.text_input("Knowledge Directory Path", value=DEFAULT_KNOWLEDGE_DIR)
folder_exists = os.path.exists(knowledge_folder)

if folder_exists:
    st.sidebar.success(f"✓ Connected to `{os.path.basename(knowledge_folder)}/`")
else:
    st.sidebar.error(f"✗ Folder not found: {knowledge_folder}")

# Load Data
df, error_msg = load_movie_knowledge(knowledge_folder)

if error_msg:
    st.sidebar.warning(error_msg)
    st.stop()

# Build TF-IDF index
tfidf, tfidf_matrix = build_search_index(df)

# Display Knowledge Base Statistics
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Statistics")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Total Movies", f"{len(df):,}")
with col2:
    all_genres = set()
    for g_list in df["genres"]:
        all_genres.update(g_list)
    st.metric("Unique Genres", f"{len(all_genres)}")

valid_years = df["year"].dropna()
if not valid_years.empty:
    st.sidebar.caption(f"🗓️ **Release Period:** {int(valid_years.min())} – {int(valid_years.max())}")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Chatbot Search Settings")
max_results = st.sidebar.slider("Number of recommendations to show", min_value=1, max_value=15, value=5)
ranking_mode = st.sidebar.selectbox(
    "Ranking Priority",
    options=[
        "Weighted Rating (Quality & Popularity)",
        "Absolute Rating (Raw 0-10)",
        "Release Date (Newest First)",
        "Vote Count (Most Popular)"
    ],
    index=0
)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

st.sidebar.caption("Built with Streamlit & OKF Markdown TMDB Knowledge Base")

# ---------------------------------------------------------
# Main UI: Chat Interface
# ---------------------------------------------------------
st.markdown('<div class="main-header">🎬 AI Movie Knowledge Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask me anything about movies from your TMDB Markdown knowledge base! '
    'I provide grounded, accurate movie details, recommendations, genre searches, and release year queries.</div>',
    unsafe_allow_html=True
)

# Suggested Query Quick-Buttons
st.markdown("**💡 Quick Questions to Try:**")
col_p1, col_p2, col_p3, col_p4 = st.columns(4)

suggested_prompt = None
with col_p1:
    if st.button("💥 Recommend action movies", use_container_width=True):
        suggested_prompt = "Recommend action movies."
with col_p2:
    if st.button("📅 Movies released after 2015", use_container_width=True):
        suggested_prompt = "Find movies released after 2015."
with col_p3:
    if st.button("⭐ Which movies have high ratings?", use_container_width=True):
        suggested_prompt = "Which movies have high ratings?"
with col_p4:
    if st.button("🦇 Tell me about The Dark Knight", use_container_width=True):
        suggested_prompt = "Tell me about The Dark Knight"

# Initialize Session State Messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am your **AI Movie Knowledge Assistant**.\n\n"
                f"I have loaded **{len(df):,} movie markdown notes** from your knowledge folder (`movies/`).\n\n"
                "You can ask me questions like:\n"
                "- **Recommend action movies** (or comedy, sci-fi, horror, etc.)\n"
                "- **Find movies released after 2015** (or in the 90s, before 2000)\n"
                "- **Which movies have high ratings?**\n"
                "- **Tell me about Inception** (or Star Wars, Pulp Fiction, etc.)\n"
                "- **Sci-fi movies about space and time travel**\n\n"
                "All information is directly grounded in your local markdown files with zero fabricated details!"
            ),
            "cards": []
        }
    ]

# Display Existing Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "cards" in msg and msg["cards"]:
            for m in msg["cards"]:
                render_movie_card(m)

# Capture User Input
user_input = st.chat_input("Ask a question about movies (e.g. Recommend action movies, movies after 2015...)...")
if suggested_prompt:
    user_input = suggested_prompt

if user_input:
    # 1. Append User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Process Query with Natural Language Engine
    with st.chat_message("assistant"):
        with st.spinner("Searching local movie knowledge files..."):
            parsed_intent = parse_user_query(user_input, df)
            res_type, results, intent = execute_movie_query(
                parsed_intent, df, tfidf, tfidf_matrix,
                max_results=max_results,
                ranking_mode=ranking_mode
            )
            response_text = generate_conversational_response(
                res_type, results, intent, len(results), len(df)
            )
            
            st.markdown(response_text)
            if results:
                for movie_data in results:
                    render_movie_card(movie_data)

    # 3. Store Assistant Response in Session State
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "cards": results
    })
