import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from typing import Callable
import re


features_dict_default = {
    'max_features': None, #default: no maximum
    'max_df': 1.0, #default: no maximum frequency (100%)
    'min_df': 1 #default: no minimum absolute frequency (1)
}


def find_annotations(transcripts: pd.Series) -> set[str]:
    '''

    Find set of transcript annotations inside square brackets.

    Args:
        transcripts (pd.Series): List of transcript texts.

    Returns:
        set[str]: Returns a sequence of unique annotations.

    '''
    # Gather set (unique elements) of annotations in [...]
    annotations = set()
    for transcript in transcripts:
        matches = re.findall(r'\[.*?\]', transcript)
        annotations.update(matches)
        
    return annotations


def custom_preprocessor_transcripts(text: str) -> str:
    '''

    Preprocess input text BEFORE tokenization. Callable input into scikit-learn CountVectorizer.
    1. Lowercase
    2. TODO: strip accents
    3. Remove transcript annotations

    Args:
        text (str): Input text.

    Returns:
        str: Returns text with preprocessing steps executed.

    '''

    # 1. Default preprocessor: lowercase
    text1 = text.lower()

    # 2. Default preprocessor: strip_accents (TODO)
    text2 = text1

    # 3. Remove transcript annotations
    text3 = re.sub(r'\[.*?\]', '', text2)
    
    return text3


def custom_tokenizer_spacy0(text: str) -> list[str]:
    '''

    Tokenize input text using SpaCy tokenizer. Callable input into scikit-learn CountVectorizer.

    Args:
        text (str): Input string.

    Returns:
        list[str]: Returns a sequence of tokens.

    '''
    # Instantiate tokenizer and get spacy.tokens.doc.Doc list of spacy.tokens.token.Token
    nlp = spacy.load('en_core_web_sm', disable=['tok2vec', 'tagger', 'parser', 'ner', 'lemmatizer', 'attibute_ruler'])
    tokens_doc = nlp(text)

    # Return 'text' str portion of each Token
    tokens_sequence = [token.text for token in tokens_doc]

    return tokens_sequence


def custom_tokenizer_spacy1(text: str) -> list[str]:
    '''

    Tokenize input text using SpaCy tokenizer. Drops "non-word" tokens. Callable input into scikit-learn CountVectorizer.

    Args:
        text (str): Input string.

    Returns:
        list[str]: Returns a sequence of tokens with whitespace, punctuation, and digit-only tokens dropped.

    '''
    # Instantiate tokenizer and get spacy.tokens.doc.Doc list of spacy.tokens.token.Token
    nlp = spacy.load('en_core_web_sm', disable=['tok2vec', 'tagger', 'parser', 'ner', 'lemmatizer', 'attibute_ruler'])
    tokens_doc = nlp(text)

    # Return 'text' str portion of each Token
    # 1. Drop '\n' whitespace
    # 2. Drop punctuation (.,'"!?)
    # 3. Drop digits (pure)
    tokens_sequence = [token.text for token in tokens_doc if not token.is_space and not token.is_punct and not token.is_digit and len(token.text) > 1]

    return tokens_sequence


def custom_tokenizer_spacy_lemma(text: str) -> list[str]:
    '''

    Tokenize and lemmatize input text using SpaCy tokenizer. Drops "non-word" tokens. Callable input into scikit-learn CountVectorizer.

    Args:
        text (str): Input string.

    Returns:
        list[str]: Returns a sequence of lemmatized tokens with whitespace, punctuation, and digit-only tokens dropped.

    '''
    # Instantiate tokenizer and get spacy.tokens.doc.Doc list of spacy.tokens.token.Token
    nlp = spacy.load('en_core_web_sm')
    tokens_doc = nlp(text)

    # Return 'lemma_' str portion of each Token
    # 1. Drop '\n' whitespace
    # 2. Drop punctuation (.,'"!?)
    # 3. Drop digits (pure)
    tokens_sequence = [token.lemma_ for token in tokens_doc if not token.is_space and not token.is_punct and not token.is_digit and len(token.text) > 1]

    return tokens_sequence


def vectorize_transcripts_to_bow(
    transcripts: pd.Series,
    custom_preprocessor: Callable[[str], str],
    custom_tokenizer: Callable[[str], list[str]],
    stop_words: list[str] = None,
    features_dict: dict = features_dict_default
) -> tuple[csr_matrix, list[str]]:
    '''

    Call scikit-learn CountVectorizer which combines preprocessor, tokenizer, and stopword removal.

    Args:
        transcripts (pd.Series): ...
        custom_preprocessor (Callable[[str], str]): ...
        custom_tokenizer (Callable[[str], list[str]]): ...
        stop_words (list[str]):
        features_dict (dict):

    Returns:
        tuple[csr_matrix, list[str]]: Returns a tuple of
                                      0. "Bag of Words" document-term matrix in CSR (Compressed Sparse Row) format.
                                      1. List of tokens

    '''
    transcripts_vectorizer = CountVectorizer(preprocessor=custom_preprocessor,
                                             tokenizer=custom_tokenizer,
                                             #analyzer='word', #default
                                             #lowercase=True, #default (N/A if custom preprocessor)
                                             stop_words=stop_words,
                                             max_features=features_dict['max_features'],
                                             max_df=features_dict['max_df'], # Ignore terms appearing in >n% of documents
                                             min_df=features_dict['min_df'], # Ignore terms appearing in <n% of documents
                                           )
    transcripts_bow_csr = transcripts_vectorizer.fit_transform(transcripts)
    transcripts_bow_tokens = transcripts_vectorizer.get_feature_names_out()
    transcripts_bow_tuple = (transcripts_bow_csr, transcripts_bow_tokens)
    
    return transcripts_bow_tuple
