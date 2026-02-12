"""
YouTube transcript fetcher module.
Extracts transcripts from YouTube videos for concept extraction.
"""
import re
from typing import Optional, Tuple, List
from dataclasses import dataclass
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


@dataclass
class TranscriptSegment:
    """A segment of transcript with timing info."""
    text: str
    start: float  # seconds
    duration: float


@dataclass
class Transcript:
    """Full transcript for a video."""
    video_id: str
    language: str
    segments: List[TranscriptSegment]
    full_text: str
    duration_seconds: float
    word_count: int
    
    @property
    def estimated_tokens(self) -> int:
        """Rough estimate of token count (words * 1.3)."""
        return int(self.word_count * 1.3)


def extract_video_id(youtube_url: str) -> Optional[str]:
    """
    Extract video ID from various YouTube URL formats.
    
    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    """
    if not youtube_url:
        return None
        
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',  # Just the ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    
    return None


def get_transcript(
    youtube_url: str,
    languages: List[str] = ['en', 'hi', 'en-US', 'en-GB']
) -> Tuple[Optional[Transcript], Optional[str]]:
    """
    Fetch transcript for a YouTube video.
    
    Args:
        youtube_url: YouTube video URL or ID
        languages: Preferred languages in order of preference
        
    Returns:
        Tuple of (Transcript object, error message if any)
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return None, f"Could not extract video ID from: {youtube_url}"
    
    try:
        api = YouTubeTranscriptApi()
        
        # Try each language in order
        fetched = None
        language_used = None
        
        for lang in languages:
            try:
                fetched = api.fetch(video_id, languages=[lang])
                language_used = f"{fetched.language} ({fetched.language_code})"
                break
            except NoTranscriptFound:
                continue
            except Exception:
                continue
        
        # If no preferred language found, try to get any available
        if fetched is None:
            try:
                # List available transcripts
                transcript_list = api.list(video_id)
                if transcript_list:
                    # Get first available
                    first_available = transcript_list[0]
                    fetched = api.fetch(video_id, languages=[first_available.language_code])
                    language_used = f"{fetched.language} ({fetched.language_code})"
            except Exception as e:
                return None, f"No transcript available for video {video_id}: {str(e)}"
        
        if fetched is None:
            return None, f"No transcript available for video: {video_id}"
        
        # Build transcript object from snippets
        segments = []
        full_text_parts = []
        total_duration = 0
        
        for snippet in fetched.snippets:
            segment = TranscriptSegment(
                text=snippet.text.replace('\n', ' ').strip(),
                start=snippet.start,
                duration=snippet.duration
            )
            segments.append(segment)
            if segment.text and segment.text not in ['[संगीत]', '[Music]', '[Applause]']:
                full_text_parts.append(segment.text)
            total_duration = max(total_duration, snippet.start + snippet.duration)
        
        full_text = ' '.join(full_text_parts)
        word_count = len(full_text.split())
        
        result = Transcript(
            video_id=video_id,
            language=language_used or "unknown",
            segments=segments,
            full_text=full_text,
            duration_seconds=total_duration,
            word_count=word_count
        )
        
        return result, None
        
    except TranscriptsDisabled:
        return None, f"Transcripts are disabled for video: {video_id}"
    except NoTranscriptFound:
        return None, f"No transcript found for video: {video_id}"
    except Exception as e:
        return None, f"Error fetching transcript for {video_id}: {str(e)}"


def chunk_transcript(
    transcript: Transcript,
    max_tokens: int = 100000,
    overlap_tokens: int = 500
) -> List[str]:
    """
    Split transcript into chunks for LLM processing.
    
    Args:
        transcript: Transcript object
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks for context continuity
        
    Returns:
        List of text chunks
    """
    if transcript.estimated_tokens <= max_tokens:
        return [transcript.full_text]
    
    # Split by sentences (rough approximation)
    sentences = re.split(r'(?<=[.!?])\s+', transcript.full_text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = int(len(sentence.split()) * 1.3)
        
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_text = ' '.join(current_chunk[-3:])  # Last 3 sentences
            current_chunk = [overlap_text, sentence]
            current_tokens = int(len(overlap_text.split()) * 1.3) + sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


if __name__ == '__main__':
    # Test with a sample video
    test_url = "https://www.youtube.com/watch?v=plXIz2AzHcw"
    
    print(f"Fetching transcript for: {test_url}")
    transcript, error = get_transcript(test_url)
    
    if error:
        print(f"Error: {error}")
    else:
        print(f"\nVideo ID: {transcript.video_id}")
        print(f"Language: {transcript.language}")
        print(f"Duration: {transcript.duration_seconds:.0f} seconds")
        print(f"Word count: {transcript.word_count}")
        print(f"Estimated tokens: {transcript.estimated_tokens}")
        print(f"\nFirst 500 chars of transcript:")
        print(transcript.full_text[:500])
