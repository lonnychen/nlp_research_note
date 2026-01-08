import pandas as pd
import numpy as np
from numpy.typing import NDArray
import math
import seaborn as sns
sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt


def print_topic_words(df_topic_words_list: list[pd.DataFrame], txtfile: bool = False) -> None:
    '''

    Print per-topic top words in order in one continuous string.

    Args:
        df_topic_words_list (list[pd.DataFrame]): Input list of per-topic DataFrames storing word and score.
        txtfile (bool): Also output print statements to a text file.

    Returns:
        None: Just print.

    '''
    k_topics = len(df_topic_words_list)
    for idx in range(k_topics):
        print(f'Topic {idx+1}:')
        print(' '.join(df_topic_words_list[idx]['word'].astype(str)), '\n')

    if txtfile:
        with open('results/topic_words.txt', 'w', encoding="utf-8") as f:
            for idx in range(k_topics):
                print(f'Topic {idx+1}:', file=f)
                print(' '.join(df_topic_words_list[idx]['word'].astype(str)), '\n', file=f)
 
    return None


def plot_topic_words(df_topic_words_list: list[pd.DataFrame]) -> None:
    '''

    Plot per-topic top word scroes vertically (topics organized side-by-side).

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


def plot_topic_words_4cols(df_topic_words_list: list[pd.DataFrame]) -> None:
    '''

    Plot per-topic top word scrores vertically (topics organized as N rows x 4 cols).

    Args:
        df_topic_words_list (list[pd.DataFrame]): Input list of per-topic DataFrames storing word and score.

    Returns:
        None: Just plots.

    '''
    k_topics = len(df_topic_words_list)
    n_top_words = len(df_topic_words_list[0])

    # Special subplots setup for "4cols"
    nrows = math.ceil(k_topics/4)
    ncols = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16,nrows*n_top_words/5))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        sns.barplot(data=df_topic_words_list[idx], y='word', x='score', ax=ax)
        if idx!=0:
            ax.set_ylabel(None)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_title(f'Topic {idx+1}', fontsize=14)

    plt.tight_layout()
    plt.savefig('plots/topic_top_words.png')
    
    return None


def print_dominant_topics(doc_topics: NDArray, txtfile: bool = False) -> pd.DataFrame:
    '''

    Print the distribution of dominant topics in the dataset.

    Args:
        doc_topics (NDArray): Input document-topics array (output of LDiA).
        txtfile (bool): Also output print statements to a text file.

    Returns:
        pd.DataFrame: Returns DataFrame of numbered dominant topics.

    '''
    dominant_topic_idx = np.argmax(doc_topics, axis=1)
    dominant_topics = [f'topic_{idx+1}' for idx in dominant_topic_idx]
    df_dominant_topics = pd.DataFrame(dominant_topics, columns=['dominant_topic'])
    print(df_dominant_topics.value_counts())
    if txtfile:
        with open('results/dominant_topics.txt', 'w', encoding="utf-8") as f:
            print(df_dominant_topics.value_counts(), file=f)
 
    return df_dominant_topics


def plot_counts_by_group(
    df_plot: pd.DataFrame,
    counts: str = 'dominant_topic',
    groups: str = 'channel_id',
) -> None:
    '''

    Plot a "countplot" on the counts column separated by hue of the groups column.

    Args:
        df_plot (pd.DataFrame): DataFrame with expected columns for counts and groups.
        counts (str): Name of the counts column to plot on the y-axis.
        groups (str): Name of the group column to separate counts by hue.

    Returns:
        None: Just plots.

    '''
    # Setup plot
    plt.subplots(figsize=(8, 4))

    # Use first channel's order for x-axis
    first_channel = df_plot[groups].unique()[0]
    x_order = df_plot[df_plot[groups]==first_channel][counts].value_counts().index

    # PLOT
    sns.countplot(data=df_plot, x=counts, hue=groups, order=x_order)

    plt.title('YouTube Stand-Up Comedy Shorts:\nDominant Topics by Comedy Central Channel (US vs. UK)')
    plt.tight_layout()
    plt.savefig('plots/doc_dominant_topic_by_channel.png')

    return None
