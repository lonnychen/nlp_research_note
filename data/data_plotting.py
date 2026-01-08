import pandas as pd
import numpy as np
import math
from scipy.sparse import csr_matrix
import seaborn as sns
sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt


def plot_transcript_len(
    transcript_lengths: pd.Series,
    plot_mean: bool = True,
    plot_median: bool = True
) -> None:
    '''

    Plot the transcript length distribution as a histogram.

    Args:
        transcript_lengths (pd.Series): Input data.
        plot_mean (bool): Plot vertical mean line.
        plot_median (bool): Plot vertical median line.

    Returns:
        None: Just plots.

    '''
    # Setup plot
    plt.figure(figsize=(8, 3))
    
    # PLOT
    bins = int(math.sqrt(len(transcript_lengths)))
    sns.histplot(transcript_lengths, bins=bins)
    plt.xlabel('Transcript length in tokens')

    # Lines
    if plot_mean:
        plt.axvline(x=transcript_lengths.mean(), color='red', linestyle='--', linewidth=2)
    if plot_median:
        plt.axvline(x=transcript_lengths.median(), color='black', linestyle='-', linewidth=2)

    plt.tight_layout()
    plt.savefig('plots/transcript_lengths.png')
    
    return None


def create_token_frequency(
    transcripts_bow_tuple: tuple[csr_matrix, list[str]],
) -> pd.DataFrame:
    '''

    Extract tokens and calculate ranks and frequencies for each token.

    Args:
        transcripts_bow_tuple (tuple[csr_matrix, list[str]]): A tuple of
                                                                0. "Bag of Words" document-term matrix in CSR (Compressed Sparse Row) format
                                                                1. List of tokens

    Returns:
        pd.DataFrame: A DataFrame with rank, frequency, and token columns.

    '''
    df_token_frequency = pd.DataFrame({
        'token': transcripts_bow_tuple[1], #list of feature names
        'total_frequency': np.asarray(transcripts_bow_tuple[0].sum(axis=0)).ravel(), #sum total occurances per-token column
        'doc_frequency': np.asarray((transcripts_bow_tuple[0] > 0).sum(axis=0)).ravel() #sum doc occurances per-token column
    })

    # Total frequency rank of tokens
    df_token_frequency = df_token_frequency.sort_values(by='total_frequency', ascending=False).reset_index(drop=True)
    df_token_frequency['total_rank'] = df_token_frequency.index + 1

    # Document frequency rank of tokens
    df_token_frequency = df_token_frequency.sort_values(by='doc_frequency', ascending=False).reset_index(drop=True)
    df_token_frequency['doc_rank'] = df_token_frequency.index + 1

    return df_token_frequency


def plot_zipf(
    df_token_frequency: pd.DataFrame,
    rank_by: str = 'doc',
    top_n = 3,
    bottom_n = 1,
    mid_ranks = [30, 100, 300],
    min_df_line: int = None,
    max_df_line: int = None
) -> None:
    '''

    Plot "Zipf" word rank vs. frequency for linear-linear and log-log as side-by-side subplots.

    Args:
        df_token_frequency (pd.DataFrame): Input data with expected rank, frequency, and token columns.
        rank_by (str): Use the ranking of either 'doc' or 'total' frequency.
        top_n (int); Number of top-ranked tokens to token-annotate on plots.
        bottom_n (int); Number of bottom-ranked tokens to token-annotate on plots.
        mid_ranks (list[int]): List of middle ranks to token-annotate on plots.
        min_df_line (int): x-location of the min_df cut-off
        max_df_line (int): x-location of the max_df limit

    Returns:
        None: Just plots.

    '''
    # Setup subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))

    for idx, ax in enumerate(axes):
        
        # PLOT
        sns.scatterplot(data=df_token_frequency, x=f'{rank_by}_rank', y=f'{rank_by}_frequency', ax=ax)
        
        if idx==0:
            ax.set_title("Linear-linear plot") 
        if idx==1:
            ax.set_title("Log-log plot") 
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        # Top N annotations
        for i, row in df_token_frequency.head(top_n).iterrows():
            ax.text(row[f'{rank_by}_rank'], row[f'{rank_by}_frequency'], row['token'],
                    fontsize=9, ha='left', va='bottom')

        # Bottom N annotations
        for i, row in df_token_frequency.tail(bottom_n).iterrows():
            ax.text(row[f'{rank_by}_rank'], row[f'{rank_by}_frequency'], row['token'],
                    fontsize=9, ha='left', va='bottom')

        # Mid N annotations
        for i, row in df_token_frequency.iloc[mid_ranks].iterrows():
            ax.text(row[f'{rank_by}_rank'], row[f'{rank_by}_frequency'], row['token'],
                    fontsize=9, ha='left', va='bottom')

        # Plot FINAL token selection lines
        if min_df_line is not None:
            ax.axvline(x=min_df_line, color='red', linestyle='--', linewidth=1)
        if max_df_line is not None:
            ax.axvline(x=max_df_line, color='red', linestyle='--', linewidth=1)

    plt.suptitle("Zipf's law: token rank vs. frequency plots")
    plt.tight_layout()
    plt.savefig('plots/vocabulary_zipf_plots.png')
    
    return None
