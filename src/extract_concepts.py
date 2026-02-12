"""
Concept extraction pipeline.
Processes lecture transcripts and adds concept nodes to the graph.
"""
import asyncio
import json
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.transcript_fetcher import get_transcript, chunk_transcript
from src.concept_extractor import ConceptExtractor, Concept
from src.graph_builder import GraphBuilder, Node, Edge


@dataclass
class ProcessingResult:
    """Result of processing a single lecture."""
    lecture_id: str
    lecture_name: str
    youtube_url: str
    transcript_words: int
    concepts_extracted: int
    error: Optional[str] = None


def load_graph(output_dir: Path) -> GraphBuilder:
    """Load existing graph from JSONL files."""
    builder = GraphBuilder()
    builder.load_from_jsonl(output_dir)
    return builder


def get_lectures_with_youtube(builder: GraphBuilder) -> List[Dict]:
    """Get all lecture nodes that have YouTube URLs."""
    lectures = []
    
    for node_id, node in builder.nodes.items():
        if node.type == 'V_lecture':
            youtube_url = node.properties.get('youtube_url')
            if youtube_url:
                # Find parent course
                course_id = node.properties.get('course_id')
                course = builder.nodes.get(course_id)
                
                # Find parent stream
                stream_name = "Unknown"
                if course:
                    stream_id = course.properties.get('stream_id')
                    stream = builder.nodes.get(stream_id)
                    if stream:
                        stream_name = stream.name
                
                lectures.append({
                    'lecture_id': node_id,
                    'lecture_name': node.name,
                    'youtube_url': youtube_url,
                    'course_id': course_id,
                    'course_name': course.name if course else "Unknown",
                    'discipline': stream_name,
                    'week': node.properties.get('week', 1),
                    'lecture_num': node.properties.get('lecture_num', 1)
                })
    
    # Sort by course and lecture number
    lectures.sort(key=lambda x: (x['course_id'], x['lecture_num']))
    return lectures


def process_lecture(
    lecture: Dict,
    extractor: ConceptExtractor,
    builder: GraphBuilder,
    max_transcript_tokens: int = 100000
) -> ProcessingResult:
    """
    Process a single lecture: fetch transcript, extract concepts, add to graph.
    
    Args:
        lecture: Lecture info dict
        extractor: ConceptExtractor instance
        builder: GraphBuilder to add concepts to
        max_transcript_tokens: Max tokens before chunking
        
    Returns:
        ProcessingResult with status
    """
    lecture_id = lecture['lecture_id']
    lecture_name = lecture['lecture_name']
    youtube_url = lecture['youtube_url']
    
    result = ProcessingResult(
        lecture_id=lecture_id,
        lecture_name=lecture_name,
        youtube_url=youtube_url,
        transcript_words=0,
        concepts_extracted=0
    )
    
    # Fetch transcript
    transcript, error = get_transcript(youtube_url)
    if error:
        result.error = error
        return result
    
    result.transcript_words = transcript.word_count
    
    # Chunk if needed
    if transcript.estimated_tokens > max_transcript_tokens:
        chunks = chunk_transcript(transcript, max_tokens=max_transcript_tokens)
        print(f"    Split into {len(chunks)} chunks")
        
        concepts = extractor.extract_from_chunks(
            chunks=chunks,
            lecture_name=lecture_name,
            course_name=lecture['course_name'],
            discipline=lecture['discipline'],
            lecture_id=lecture_id
        )
    else:
        concepts = extractor.extract_concepts(
            transcript=transcript.full_text,
            lecture_name=lecture_name,
            course_name=lecture['course_name'],
            discipline=lecture['discipline'],
            lecture_id=lecture_id
        )
    
    result.concepts_extracted = len(concepts)
    
    # Add concepts to graph
    for concept in concepts:
        builder.add_concept(
            lecture_id=lecture_id,
            name=concept.name,
            description=concept.description,
            keywords=concept.keywords,
            confidence=concept.confidence
        )
    
    return result


def process_lectures(
    output_dir: Path,
    api_key: str,
    model: str = "llama",
    limit: Optional[int] = None,
    delay: float = 2.0
) -> List[ProcessingResult]:
    """
    Process all lectures with YouTube URLs.
    
    Args:
        output_dir: Directory with graph JSONL files
        api_key: OpenRouter API key
        model: Model to use ("deepseek", "llama", or "qwen")
        limit: Maximum number of lectures to process
        delay: Delay between API calls (seconds)
        
    Returns:
        List of ProcessingResults
    """
    print("=" * 60)
    print("CONCEPT EXTRACTION PIPELINE")
    print("=" * 60)
    
    # Load existing graph
    print("\n[1/4] Loading existing graph...")
    builder = load_graph(output_dir)
    stats = builder.get_stats()
    print(f"  Loaded {stats['total_nodes']} nodes, {stats['total_edges']} edges")
    
    # Get lectures with YouTube URLs
    print("\n[2/4] Finding lectures with YouTube URLs...")
    lectures = get_lectures_with_youtube(builder)
    print(f"  Found {len(lectures)} lectures with YouTube URLs")
    
    if limit:
        lectures = lectures[:limit]
        print(f"  Processing first {limit} lectures")
    
    # Initialize extractor
    print(f"\n[3/4] Extracting concepts using {model} model...")
    extractor = ConceptExtractor(api_key=api_key, model=model)
    
    results = []
    
    for i, lecture in enumerate(tqdm(lectures, desc="Processing lectures")):
        print(f"\n  [{i+1}/{len(lectures)}] {lecture['lecture_name']}")
        
        try:
            result = process_lecture(lecture, extractor, builder)
            results.append(result)
            
            if result.error:
                print(f"    Error: {result.error}")
            else:
                print(f"    Transcript: {result.transcript_words} words")
                print(f"    Concepts: {result.concepts_extracted}")
                
        except Exception as e:
            print(f"    Exception: {e}")
            results.append(ProcessingResult(
                lecture_id=lecture['lecture_id'],
                lecture_name=lecture['lecture_name'],
                youtube_url=lecture['youtube_url'],
                transcript_words=0,
                concepts_extracted=0,
                error=str(e)
            ))
        
        # Rate limiting
        if i < len(lectures) - 1:
            time.sleep(delay)
    
    # Save updated graph
    print("\n[4/4] Saving updated graph...")
    builder.save_to_jsonl(output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    
    successful = [r for r in results if not r.error]
    failed = [r for r in results if r.error]
    total_concepts = sum(r.concepts_extracted for r in results)
    
    print(f"\nSummary:")
    print(f"  Lectures processed: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total concepts extracted: {total_concepts}")
    
    if failed:
        print(f"\nFailed lectures:")
        for r in failed[:5]:
            print(f"  - {r.lecture_name}: {r.error}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    
    # Final graph stats
    final_stats = builder.get_stats()
    print(f"\nFinal graph:")
    for node_type, count in final_stats['nodes_by_type'].items():
        print(f"  {node_type}: {count}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Extract concepts from lecture transcripts'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/output',
        help='Directory with graph JSONL files'
    )
    parser.add_argument(
        '--api-key', '-k',
        type=str,
        required=True,
        help='OpenRouter API key'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='llama',
        choices=['deepseek', 'llama', 'qwen'],
        help='Model to use for extraction'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Maximum number of lectures to process'
    )
    parser.add_argument(
        '--delay', '-d',
        type=float,
        default=2.0,
        help='Delay between API calls (seconds)'
    )
    
    args = parser.parse_args()
    
    # Resolve path
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / args.output
    
    process_lectures(
        output_dir=output_dir,
        api_key=args.api_key,
        model=args.model,
        limit=args.limit,
        delay=args.delay
    )


if __name__ == '__main__':
    main()
