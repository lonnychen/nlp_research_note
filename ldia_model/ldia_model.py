import pandas as pd
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import CoherenceModel
import gensim.corpora as corpora
from data.data_preprocessing import custom_preprocessor_transcripts, custom_tokenizer_spacy_lemma


def fit_ldia_model(
    bow_csr: csr_matrix,
    k_topics: int,
    random_state: int = None,
) -> tuple[NDArray, NDArray]:
    '''

    Fits scikit-learn LatentDirichletAllocation on BoW document-term matrix.

    Args:
        bow_csr (csr_matrix): Input BoW matrix.
        k_topics (int): Target number of topics for modeling.
        random_state (int): Random number to set for reproducible results

    Returns:
        tuple[NDArray, NDArray]: Returns document-topics and topic-words arrays from LDiA fitting.

    '''
    lda_model = LatentDirichletAllocation(n_components=k_topics, random_state=random_state)
    lda_doc_topics = lda_model.fit_transform(bow_csr) #ndarray (docs, topics)
    lda_topic_words = lda_model.components_ #ndarray (topics, words)
        
    return (lda_doc_topics, lda_topic_words)


def create_df_topic_words_list(
    n_top_words: int,
    topic_word_scores: NDArray,
    word_names: list[str]
) -> list[pd.DataFrame]:
    '''

    Create list of DataFrames for each topic containing top words and scores.
    Code referenced from Raschka (YEAR).

    Args:
        n_top_words (int): Number of top word scores to keep.
        topic_word_scores (NDArray): 2d array of per-topic word scores.
        word_names (list[str]): List of word/token names.

    Returns:
        list[pd.DataFrame]: Returns list of DataFrames for each topic containing top words and scores.

    '''
    df_topic_words_list = []
    for idx, topic in enumerate(topic_word_scores):
        topic_words = [{'word': word_names[i], 'score': topic[i]} for i in topic.argsort()[:-n_top_words - 1:-1]]
        df_topic_words_list.append(pd.DataFrame(topic_words))

    return df_topic_words_list


def get_gensim_coherence_score(
    df_topic_words_list: list[pd.DataFrame],
    transcripts: pd.Series,
    coherence_measure: str = 'c_v'
) -> float:
    '''

    Adapt scikit-learn outputs to inputs for gensim CoherenceModel and get "Cv" coherence score.
    Code referenced from https://stackoverflow.com/questions/60613532/how-do-i-calculate-the-coherence-score-of-an-sklearn-lda-model.

    Args:
        df_topic_words_list (list[pd.DataFrame]): Input list of per-topic DataFrames storing word and score.
        transcripts (pd.Series): Original text data.
        coherence_measure (str): Choice of coherence measure among {'u_mass', 'c_v', 'c_uci', 'c_npmi'}.

    Returns:
        list[pd.DataFrame]: Returns list of DataFrames for each topic containing top words and scores.

    '''
    # Get topics argument
    topic_words_list = []
    for df_topic_words in df_topic_words_list:
        topic_words_list.append(df_topic_words['word'].tolist())
    
    # Get texts and dictionary arguments
    # Put transcripts through Data Preprocessing pipeline (preprocess -> tokenize/lemmatize) but not stop word removal (which is OK)
    texts = [custom_tokenizer_spacy_lemma(custom_preprocessor_transcripts(transcript)) for transcript in transcripts]
    dictionary = corpora.Dictionary(texts)

    # Instantiate and run gensim CoherenceModel
    coherence_model = CoherenceModel(topics=topic_words_list, texts=texts, dictionary=dictionary, coherence=coherence_measure)
    coherence = coherence_model.get_coherence()

    return coherence
