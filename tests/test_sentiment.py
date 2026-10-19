from sentiment import score_headline, label_for_score, summarize_articles


def test_finance_vocabulary_scores_correctly():
    # VADER's general-purpose lexicon scores these as neutral (0.0) without
    # the finance-specific lexicon extension -- this is the whole point of it.
    assert score_headline("Stock soars to record high on blowout guidance") > 0.5
    assert score_headline("Company plunges after disappointing earnings miss") < -0.5
    assert score_headline("Regulators launch investigation into accounting practices") < 0


def test_score_headline_empty_is_neutral():
    assert score_headline("") == 0.0
    assert score_headline(None) == 0.0


def test_label_for_score_thresholds():
    assert label_for_score(0.5) == "positive"
    assert label_for_score(-0.5) == "negative"
    assert label_for_score(0.0) == "neutral"
    assert label_for_score(0.19) == "neutral"
    assert label_for_score(0.2) == "positive"


def test_summarize_articles_empty_list():
    result = summarize_articles([])
    assert result["articles"] == []
    assert result["mean_score"] is None
    assert result["overall_label"] == "no recent headlines"


def test_summarize_articles_computes_mean_and_labels():
    articles = [
        {"title": "Stock soars to record high", "publisher": "A"},
        {"title": "Shares plunge on miss", "publisher": "B"},
    ]
    result = summarize_articles(articles)
    assert len(result["articles"]) == 2
    assert result["articles"][0]["sentiment_label"] == "positive"
    assert result["articles"][1]["sentiment_label"] == "negative"
    # mean should land between the two individual scores
    scores = [a["sentiment_score"] for a in result["articles"]]
    assert min(scores) <= result["mean_score"] <= max(scores)
