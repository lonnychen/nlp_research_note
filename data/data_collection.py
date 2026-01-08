import pandas as pd
import re
import time
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api.proxies import WebshareProxyConfig
from youtube_transcript_api._errors import TranscriptsDisabled


def _get_shorts_video_ids(
    channel_search_url: str,
    list_search_str: list[str] = [],
    duration_max: int = 60,
) -> pd.DataFrame:
    '''

    Use the YT-DLP library to extract information found from the list of videos returned at one YouTube (search) URL. Reference:  https://github.com/Standup4AI/dataset/blob/main/URL_videos.py `get_standup_video_urls`.

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


def get_transcript_from_video_id(
    video_id: str,
    language_code: str,
    use_proxy: bool = True,
    retries: int = 10
) -> str:
    '''

    Apply per row to obtain a video's transcript in the specified language. Handles connection or unavailable transcript errors, so it should not crash due to an exception.

    Args:
        video_id (str): The YouTube video ID.
        language_code (str): The target language code.
        use_proxy (bool): Use IP proxies to avoid IpBlocked from repeated transcript fetches.
        retries (int): Number of retries for API requests.

    Returns:
        str: Returns the correct language transcript from YouTube ASR.

    '''
    # Instantiate API method
    if use_proxy:
        # Workaround for IpBlocked: Webshare "Residential Proxy"
        ytt_api = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username='username', #FILL-IN LOCALLY
                proxy_password='password', #FILL-IN LOCALLY
                filter_ip_locations=['de']
            )
        )      
    else:
        ytt_api = YouTubeTranscriptApi()

    # Get list of available languages
    # Wrap around try/except in case of IncompleteRead, ProtocolError -> ChunkedEncodingError
    for retry in range(retries):
        try:
            transcript_list = ytt_api.list(video_id)
            break #success
        except TranscriptsDisabled:
            print(f"TranscriptsDisabled exception for video '{video_id}', skipping ...")
            return 'TranscriptsDisabled'
        except:
            if retry == retries-1:
                print(f"List retrieval failed for video '{video_id}', skipping ...")  
                return 'ReachedListRetryLimit'
            else:
                print(f"List retrieval failed for video '{video_id}', retrying ...")    
                time.sleep(10)

    # Check for expected language transcript
    language_code_found = False
    for t in transcript_list:
        if t.language_code == language_code:
            language_code_found = True
            break
    if not language_code_found:
        # Else NoTranscriptFound below
        print(f"Language code '{language_code}' not found for video '{video_id}', skipping ...")
        return 'UnexpectedLanguage'
    
    # Pipeline: find > fetch
    # By default, this module always chooses manually created transcripts over automatically created ones
    # Wrap around try/except in case of IncompleteRead, ProtocolError -> ChunkedEncodingError
    for retry in range(retries):
        try:
            fetched_transcript = transcript_list.find_transcript([language_code]).fetch()
            print(f"Fetch completed for video '{video_id}'")
            break #success
        except:
            if retry == retries-1:
                print(f"Fetch failed for video '{video_id}', skipping ...")  
                return 'ReachedFetchRetryLimit'
            else:
                print(f"Fetch failed for video '{video_id}', retrying ...")
                time.sleep(10)
                
    # Pipeline: convert to text
    formatter = TextFormatter()
    text_formatted_transcript = formatter.format_transcript(fetched_transcript)
    
    return text_formatted_transcript
