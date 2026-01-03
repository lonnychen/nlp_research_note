import pandas as pd
import re
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

def _get_shorts_video_ids(
    channel_search_url: str,
    list_search_str: list[str] = [],
    duration_max: int = 60,
) -> pd.DataFrame:
    '''

    Uses the YT-DLP library to extract information found from the list of videos returned at one YouTube (search) URL. Reference:  https://github.com/Standup4AI/dataset/blob/main/URL_videos.py `get_standup_video_urls`.

    Args:
        channel_search_url (str): Entire YouTube URL search string for video info extraction.
        list_search_str (list[str]): List of strings to OR filter-in (else excluded) in found video titles.
        duration_max (int): Maximum duration of YouTube shorts to filter for in seconds if videos are not from YouTube 'Shorts'.

    Returns:
        pd.DataFrame: Returns a DataFrame of per-video IDs and other metadata (when available). 

    '''    
    ydl_opts = {
        'extract_flat': True,  # Do not extract complete metadata for each video
        'skip_download': True, # We are not downloading video files
        'quiet': True,         # Suppress extra logging output
    }    
    data_list = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract the info dictionary from the provided URL
        info = ydl.extract_info(channel_search_url, download=False)
    
        # The entries key holds the list of videos found
        entries = info.get('entries', [])
        for entry in entries:
            title = entry.get('title', '')
            if not list_search_str or sum([k.lower() in title.lower() for k in list_search_str]):
    
                # Create dict of entry metadata
                video_data = {
                    'video_id': entry.get('id', None),
                    'url': entry.get('url', None),
                    'title': entry.get('title', title),
                    'duration': entry.get('duration', None),        # Duration in seconds
                    'view_count': entry.get('view_count', None)
                    #'description': entry.get('description', None),
                    #'channel_id': entry.get('uploader_id', None),  # Typically the channel's ID
                    #'channel': entry.get('uploader', None),        # Channel name
                }
                data_list.append(video_data)
    
    # Create and check for missing data.
    df = pd.DataFrame(data_list)
    if df[['video_id', 'url']].isna().any().any():
        raise ValueError('Missing value found in video_id and/or URL. Please check.')

    # Return based on type of version of video (short/normal)
    if re.search(r'\/shorts$', channel_search_url):
        return df.copy()
    else:
        return df[df['duration'] <= duration_max].copy()


def get_channels_shorts_video_ids(channels: list[dict]) -> pd.DataFrame:
    '''

    Implement `_get_shorts_video_ids` for multiple channels and combines data into one DataFrame.

    Args:
        channels (list[dict]): A list of per-channel dictionaries that define video ID and metadata extraction. Expected keys:
                                 - 'channel' (str): YouTube channel name that appears as page title.
                                 - 'search_url' (str): URL where videos are listed.
                                 - 'list_search_str' (str): List of strings to OR filter-in (else excluded) in found video titles.
                                 - 'language_code' (str): Language code to look for in available transcripts (for later use).

    Returns:
        pd.DataFrame: Returns a DataFrame of all channels' combined video IDs and metadata. 

    '''   

    df_channels_list = []
    for channel in channels:

        # Get channel_id from search_url else raise error
        m = re.search(r'youtube\.com\/(@\w+)\/', channel['search_url'])
        if m:
            channel_id = m.group(1)
        else:
            raise ValueError('Missing channel_id in search_url. Please check.')

        # Extract all videos' info
        df_channel = _get_shorts_video_ids(channel_search_url=channel['search_url'],
                                     list_search_str=channel['list_search_str'])

        # Assign channel info, add to list, print,
        df_channel['channel'] = channel['channel']
        df_channel['channel_id'] = channel_id
        df_channel['language_code'] = channel['language_code']
        df_channels_list.append(df_channel)
        print(f'Channel {channel_id} extracted {len(df_channel)} videos from {channel['search_url']}')

    return pd.concat(df_channels_list)


def get_transcript_from_video_id(video_id: str, language_code: str) -> str:
    '''

    Apply per-row to get a video's transcript in the specified language. Asserts `ValueError` if not available.

    Args:
        video_id (str): The YouTube video ID.
        language_code (str): The target language code.

    Returns:
        str: Returns the correct language transcript from YouTube ASR.

    '''
    # Instantiate API methods
    ytt_api = YouTubeTranscriptApi()
    formatter = TextFormatter()

    # Check for expected language transcript
    transcript_list = ytt_api.list(video_id)
    language_code_found = False
    for t in transcript_list:
        if t.language_code == language_code:
            language_code_found = True
            break
    if not language_code_found:
        # Else NoTranscriptFound below
        raise ValueError(f"Language code '{language_code}' not found for video '{video_id}'!")
    
    # Pipeline: find > fetch > convert to text
    # By default this module always chooses manually created transcripts over automatically created ones
    fetched_transcript = transcript_list.find_transcript([language_code]).fetch()
    
    # TODO: check for missing content
    
    text_formatted_transcript = formatter.format_transcript(fetched_transcript)
    return text_formatted_transcript
