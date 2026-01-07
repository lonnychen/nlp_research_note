import pandas as pd
import numpy as np
from numpy.typing import NDArray
import seaborn as sns
sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt


def print_topic_words(df_topic_words_list: list[pd.DataFrame]) -> None:
    '''

    Print per-topic top words in order in one continuous string.

    Args:
        df_topic_words_list (list[pd.DataFrame]): Input list of per-topic DataFrames storing word and score.

    Returns:
        None: Just print.

    '''
    k_topics = len(df_topic_words_list)
    for idx in range(k_topics):
        print(f'Topic {idx+1}:')
        print(' '.join(df_topic_words_list[idx]['word'].astype(str)))
    return None


def plot_topic_words(df_topic_words_list: list[pd.DataFrame]) -> None:
    '''

    Plots per-topic top word scrores vertically (topics organized side-by-side)

    Args:
        df_topic_words_list (list[pd.DataFrame]): Input list of per-topic DataFrames storing word and score.

    Returns:
        None: Just plots.

    '''
    k_topics = len(df_topic_words_list)
    n_top_words = len(df_topic_words_list[0])
    fig, axes = plt.subplots(nrows=1, ncols=k_topics, figsize=(16,n_top_words/5))
    
    for idx, ax in enumerate(axes):
        sns.barplot(data=df_topic_words_list[idx], y='word', x='score', ax=ax)
        if idx!=0:
            ax.set_ylabel(None)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_title(f'Topic {idx+1}', fontsize=14)
    plt.tight_layout()

    return None


def print_dominant_topics(doc_topics: NDArray) -> None:
    '''

    Prints the distribution of dominant topics in the dataset.

    Args:
        doc_topics (NDArray): Input document-topics array (output of LDiA).

    Returns:
        None: Just prints.

    '''
    dominant_topic_idx = np.argmax(doc_topics, axis=1)
    dominant_topics = [f'topic_{idx}' for idx in dominant_topic_idx]
    df_dominant_topics = pd.DataFrame(dominant_topics, columns=['dominant_topic'])
    print(df_dominant_topics.value_counts())

    return None
