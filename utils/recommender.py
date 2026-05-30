import math

def score_title(row, user_profile=None, query_vector_score=0):

    """
    Hybrid ranking score for Netflix-style recommendations
    """

    user_profile = user_profile or {}

    # ------------------------------------------------------------
    # 1. Popularity signal
    # ------------------------------------------------------------
    watch = float(row.get("total_watchtime", 0) or 0)
    watch_z = float(row.get("watchtime_zscore", 0) or 0)

    popularity_score = math.tanh(watch_z) + math.log1p(watch)

    # ------------------------------------------------------------
    # 2. Quality signal (IMDb)
    # ------------------------------------------------------------
    imdb = float(row.get("imdb_rating", 0) or 0)
    votes = float(row.get("imdb_votes", 0) or 1)

    quality_score = (imdb / 10) * math.log10(votes + 1)

    # ------------------------------------------------------------
    # 3. Recency / evergreen
    # ------------------------------------------------------------
    evergreen = float(row.get("evergreen_score", 0) or 0)

    # ------------------------------------------------------------
    # 4. Preference match (simple)
    # ------------------------------------------------------------
    genre = row.get("genre", "")

    pref_score = 0

    if genre in user_profile.get("genres", []):
        pref_score += 1.5

    # ------------------------------------------------------------
    # 5. Semantic similarity
    # ------------------------------------------------------------
    semantic = query_vector_score

    # ------------------------------------------------------------
    # FINAL SCORE
    # ------------------------------------------------------------
    score = (
        0.30 * semantic +
        0.25 * popularity_score +
        0.20 * quality_score +
        0.15 * evergreen +
        0.10 * pref_score
    )

    return round(score, 4)