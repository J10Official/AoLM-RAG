#!/usr/bin/env python3
"""
Unified NPTEL Knowledge Graph Pipeline.

Single command to run the full pipeline:
1. Parse Excel → Create stream/course nodes
2. Scrape NPTEL → Get lectures with YouTube URLs
3. Extract concepts from transcripts → Create concept nodes

Features:
- Stream filtering (--stream)
- Limit control (--limit, divided equally among streams)
- Resume functionality (skips already processed items)
- Multi-stream course handling
- Model rotation to avoid API limits
"""
import asyncio
import argparse
import json
import sys
import time
import os
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.excel_parser import get_courses, get_unique_disciplines, slugify
from src.nptel_scraper import NPTELScraper
from src.graph_builder import GraphBuilder
from src.transcript_fetcher import get_transcript, chunk_transcript
from src.concept_extractor import ConceptExtractor
from src.async_concept_extractor import AsyncConceptExtractor, extract_concepts_parallel


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_OUTPUT_DIR = "data/output"
EXCEL_FILE = "Final Course List (Jan - Apr 2026).xlsx"

# API Models (rotate to avoid rate limits)
MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1-0528:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
]

# Rate limit: ~1000 requests per model
REQUESTS_PER_MODEL = 900  # Leave some buffer

# Concurrency for parallel API calls
CONCURRENCY_LIMIT = 20  # Start with 20, can bump to 50 or 100 if no 429s


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    excel_path: Path
    output_dir: Path
    api_key: str
    streams: Optional[List[str]] = None  # None = all streams
    limit: Optional[int] = None  # None = no limit
    scrape_delay: float = 1.5
    api_delay: float = 2.0
    skip_scraping: bool = False
    skip_concepts: bool = False
    concurrency: int = CONCURRENCY_LIMIT  # Parallel API calls


@dataclass
class PipelineStats:
    """Statistics for pipeline run."""
    courses_total: int = 0
    courses_scraped: int = 0
    courses_skipped: int = 0
    lectures_total: int = 0
    lectures_with_concepts: int = 0
    lectures_processed: int = 0
    lectures_skipped: int = 0
    concepts_extracted: int = 0
    api_calls: int = 0
    errors: List[str] = field(default_factory=list)


class ModelRotator:
    """Rotates between models to avoid rate limits."""
    
    def __init__(self, models: List[str], requests_per_model: int = 900):
        self.models = models
        self.requests_per_model = requests_per_model
        self.current_index = 0
        self.request_counts = [0] * len(models)
    
    def get_model(self) -> str:
        """Get current model, rotating if needed."""
        # Check if current model is exhausted
        if self.request_counts[self.current_index] >= self.requests_per_model:
            self.current_index = (self.current_index + 1) % len(self.models)
            if all(c >= self.requests_per_model for c in self.request_counts):
                raise RuntimeError("All models have reached their request limits!")
        
        return self.models[self.current_index]
    
    def record_request(self):
        """Record a request to the current model."""
        self.request_counts[self.current_index] += 1
    
    def get_remaining(self) -> int:
        """Get total remaining requests across all models."""
        return sum(self.requests_per_model - c for c in self.request_counts)
    
    def get_status(self) -> str:
        """Get status string."""
        parts = []
        for i, (model, count) in enumerate(zip(self.models, self.request_counts)):
            name = model.split("/")[1].split(":")[0][:15]
            marker = "→" if i == self.current_index else " "
            parts.append(f"{marker}{name}: {count}/{self.requests_per_model}")
        return " | ".join(parts)


def filter_courses_by_streams(
    courses: List[Dict],
    stream_filters: Optional[List[str]]
) -> List[Dict]:
    """
    Filter courses by stream names.
    
    For courses with multiple streams, only include if ALL listed streams
    are in the filter (strict matching).
    
    Args:
        courses: List of course dicts with 'discipline' field
        stream_filters: List of stream names to filter by (case-insensitive)
        
    Returns:
        Filtered list of courses
    """
    if not stream_filters:
        return courses
    
    # Normalize filter names
    filter_set = {s.lower().strip() for s in stream_filters}
    
    filtered = []
    for course in courses:
        discipline = course.get('discipline', '')
        
        # Handle multi-stream courses (separated by newline in Excel)
        course_streams = [s.strip().lower() for s in discipline.split('\n') if s.strip()]
        
        # Check if ANY of the course's streams match ANY filter
        # (Changed from ALL to ANY for practical filtering)
        if any(cs in filter_set or any(f in cs for f in filter_set) for cs in course_streams):
            filtered.append(course)
    
    return filtered


def get_courses_with_limit(
    courses: List[Dict],
    limit: Optional[int],
    stream_filters: Optional[List[str]]
) -> List[Dict]:
    """
    Get courses up to the limit. We return MORE courses than the limit
    because some may have no lectures and will be skipped during scraping.
    The actual limit is enforced during the scraping phase.
    
    Args:
        courses: List of course dicts
        limit: Target number of courses WITH lectures (None = no limit)
        stream_filters: Streams being filtered
        
    Returns:
        List of courses (more than limit to account for skipped ones)
    """
    if not limit:
        return courses
    
    # Return 2x the limit to account for courses that may have no lectures
    # The actual limit is enforced during scraping
    buffer_limit = limit * 2
    
    if not stream_filters or len(stream_filters) == 1:
        return courses[:buffer_limit]
    
    # Group courses by primary stream
    by_stream = defaultdict(list)
    for course in courses:
        primary_stream = course.get('discipline', '').split('\n')[0].strip().lower()
        by_stream[primary_stream].append(course)
    
    # Distribute buffered limit among streams
    per_stream = buffer_limit // len(stream_filters)
    remainder = buffer_limit % len(stream_filters)
    
    result = []
    for i, stream in enumerate(stream_filters):
        stream_lower = stream.lower()
        # Find matching stream key
        for key in by_stream:
            if stream_lower in key or key in stream_lower:
                stream_limit = per_stream + (1 if i < remainder else 0)
                result.extend(by_stream[key][:stream_limit])
                break
    
    return result


def get_processed_lectures(builder: GraphBuilder) -> Set[str]:
    """Get set of lecture IDs that already have concepts."""
    processed = set()
    
    for edge in builder.edges:
        if edge.type == 'COVERS_CONCEPT':
            processed.add(edge.source)
    
    return processed


async def run_scraping_phase(
    config: PipelineConfig,
    builder: GraphBuilder,
    courses: List[Dict],
    stats: PipelineStats
) -> None:
    """
    Phase 1 & 2: Create stream/course nodes and scrape lectures.
    """
    print("\n" + "=" * 70)
    print("PHASE 1: Creating stream and course nodes")
    print("=" * 70)
    
    # Create stream nodes
    stream_ids = {}
    for course in courses:
        discipline = course['discipline']
        slug = course['discipline_slug']
        if slug not in stream_ids:
            stream_id = builder.add_stream(discipline, slug)
            stream_ids[slug] = stream_id
            print(f"  Created stream: {discipline}")
    
    print(f"\n  Total streams: {len(stream_ids)}")
    
    print("\n" + "=" * 70)
    print("PHASE 2: Scraping course details and lectures")
    print("=" * 70)
    
    stats.courses_total = len(courses)
    target_courses = config.limit or len(courses)  # Target number of courses WITH lectures
    valid_courses_count = 0  # Counter for courses that have lectures
    
    print(f"\n  Starting scraping (target: {target_courses} courses with lectures)...", flush=True)
    
    async with NPTELScraper(headless=True, slow_mo=50) as scraper:
        for idx, course in enumerate(courses):
            # Stop if we've reached our target
            if valid_courses_count >= target_courses:
                tqdm.write(f"\n  Reached target of {target_courses} courses with lectures. Stopping.")
                break
                
            course_id = course['nptel_id']
            existing_course_id = f"course_{course_id}"
            course_name = course['course_name'][:40]
            
            tqdm.write(f"\n[{valid_courses_count + 1}/{target_courses}] Processing: {course_name}...")
            
            # Check if course already exists with lectures
            if existing_course_id in builder.nodes:
                # Count existing lectures
                existing_lectures = sum(
                    1 for e in builder.edges 
                    if e.source == existing_course_id and e.type == 'HAS_LECTURE'
                )
                if existing_lectures > 0:
                    tqdm.write(f"  → Already has {existing_lectures} lectures (counting towards target)")
                    stats.courses_skipped += 1
                    stats.lectures_total += existing_lectures
                    valid_courses_count += 1
                    continue
            
            # Scrape course with timeout
            try:
                tqdm.write(f"  → Scraping {course['nptel_url']}")
                details = await asyncio.wait_for(
                    scraper.get_course_details(course['nptel_url']),
                    timeout=60.0  # 60 second timeout per course
                )
                
                # Skip courses with no lectures - don't count towards limit
                if not details.lectures:
                    tqdm.write(f"  → NO LECTURES found, skipping (not counting towards target)")
                    if details.error:
                        stats.errors.append(f"Course {course['course_name']}: No lectures - {details.error}")
                    continue
                
                # Add course node (only if it has lectures)
                stream_id = stream_ids[course['discipline_slug']]
                course_node_id = builder.add_course(
                    stream_id=stream_id,
                    course_id=course['course_id'],
                    name=course['course_name'],
                    nptel_url=course['nptel_url'],
                    nptel_id=course['nptel_id'],
                    abstract=details.abstract,
                    professor=course['professor'],
                    institute=course['institute'],
                    duration=course['duration']
                )
                
                # Add lecture nodes
                for lecture in details.lectures:
                    builder.add_lecture(
                        course_id=course_node_id,
                        name=lecture.name,
                        week=lecture.week,
                        lecture_num=lecture.lecture_num,
                        youtube_url=lecture.youtube_url
                    )
                
                stats.courses_scraped += 1
                stats.lectures_total += len(details.lectures)
                valid_courses_count += 1
                tqdm.write(f"  → Found {len(details.lectures)} lectures ✓")
                
                if details.error:
                    tqdm.write(f"  → Warning: {details.error}")
                    stats.errors.append(f"Course {course['course_name']}: {details.error}")
            
            except asyncio.TimeoutError:
                tqdm.write(f"  → TIMEOUT: Skipping course (took >60s)")
                stats.errors.append(f"Course {course['course_name']}: Timeout after 60s")
                    
            except Exception as e:
                tqdm.write(f"  → ERROR: {str(e)[:50]}")
                stats.errors.append(f"Course {course['course_name']}: {str(e)}")
            
            await asyncio.sleep(config.scrape_delay)
    
    # Save after scraping
    builder.save_to_jsonl(config.output_dir)
    print(f"\n  Courses scraped: {stats.courses_scraped}")
    print(f"  Courses skipped (already exist): {stats.courses_skipped}")
    print(f"  Total lectures: {stats.lectures_total}")


async def run_concept_extraction_phase_async(
    config: PipelineConfig,
    builder: GraphBuilder,
    stats: PipelineStats,
    concurrency: int = CONCURRENCY_LIMIT
) -> None:
    """
    Phase 3: Extract concepts from lecture transcripts using parallel processing.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: Extracting concepts from transcripts (PARALLEL)")
    print("=" * 70)
    
    # Get lectures with YouTube URLs
    lectures = []
    for node_id, node in builder.nodes.items():
        if node.type == 'V_lecture':
            youtube_url = node.properties.get('youtube_url')
            if youtube_url:
                course_id = node.properties.get('course_id')
                course = builder.nodes.get(course_id)
                stream_id = course.properties.get('stream_id') if course else None
                stream = builder.nodes.get(stream_id) if stream_id else None
                
                lectures.append({
                    'lecture_id': node_id,
                    'lecture_name': node.name,
                    'youtube_url': youtube_url,
                    'course_name': course.name if course else "Unknown",
                    'discipline': stream.name if stream else "Unknown"
                })
    
    # Get already processed lectures
    processed = get_processed_lectures(builder)
    stats.lectures_with_concepts = len(processed)
    
    # Filter out already processed
    to_process = [l for l in lectures if l['lecture_id'] not in processed]
    stats.lectures_skipped = len(lectures) - len(to_process)
    
    print(f"\n  Lectures with YouTube: {len(lectures)}")
    print(f"  Already processed: {stats.lectures_skipped}")
    print(f"  To process: {len(to_process)}")
    print(f"  Concurrency limit: {concurrency}")
    
    if not to_process:
        print("\n  All lectures already have concepts. Nothing to do!")
        return
    
    # Estimate capacity
    total_capacity = REQUESTS_PER_MODEL * len(MODELS)
    if len(to_process) > total_capacity:
        print(f"\n  WARNING: Need {len(to_process)} API calls but only ~{total_capacity} available")
        print(f"  Will process as many as possible...")
    
    # Run parallel extraction
    results, extraction_stats = await extract_concepts_parallel(
        api_key=config.api_key,
        lectures=to_process,
        get_transcript_fn=get_transcript,
        concurrency_limit=concurrency,
        save_interval=50
    )
    
    # Process results and add to graph
    print("\n  Adding concepts to graph...")
    save_interval = 100
    
    for i, result in enumerate(results):
        if result.error:
            stats.errors.append(f"{result.lecture_name}: {result.error}")
            continue
        
        # Add concepts to graph
        for concept in result.concepts:
            builder.add_concept(
                lecture_id=result.lecture_id,
                name=concept.name,
                description=concept.description,
                keywords=concept.keywords,
                confidence=concept.confidence
            )
        
        stats.lectures_processed += 1
        stats.concepts_extracted += len(result.concepts)
        stats.api_calls += 1
        
        # Periodic save
        if (i + 1) % save_interval == 0:
            builder.save_to_jsonl(config.output_dir)
            print(f"    Saved progress: {i + 1}/{len(results)} results processed")
    
    # Final save
    builder.save_to_jsonl(config.output_dir)
    print(f"\n  Concepts added to graph: {stats.concepts_extracted}")


def run_concept_extraction_phase(
    config: PipelineConfig,
    builder: GraphBuilder,
    stats: PipelineStats,
    concurrency: int = CONCURRENCY_LIMIT
) -> None:
    """
    Wrapper to run async concept extraction.
    """
    asyncio.run(run_concept_extraction_phase_async(config, builder, stats, concurrency))
    
    # Final save
    builder.save_to_jsonl(config.output_dir)


def run_pipeline(config: PipelineConfig) -> PipelineStats:
    """
    Run the full pipeline.
    """
    stats = PipelineStats()
    
    print("╔" + "═" * 68 + "╗")
    print("║" + " NPTEL KNOWLEDGE GRAPH PIPELINE ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    print(f"\nConfiguration:")
    print(f"  Excel: {config.excel_path}")
    print(f"  Output: {config.output_dir}")
    print(f"  Streams: {config.streams or 'ALL'}")
    print(f"  Limit: {config.limit or 'No limit'}")
    
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create graph
    builder = GraphBuilder()
    nodes_file = config.output_dir / 'nodes.jsonl'
    if nodes_file.exists():
        print(f"\n  Loading existing graph from {config.output_dir}")
        builder.load_from_jsonl(config.output_dir)
        existing_stats = builder.get_stats()
        print(f"    Existing nodes: {existing_stats['total_nodes']}")
        print(f"    Existing edges: {existing_stats['total_edges']}")
    
    # Read and filter courses
    print(f"\n  Reading courses from Excel...")
    all_courses = get_courses(config.excel_path)
    print(f"    Total courses in Excel: {len(all_courses)}")
    
    # Filter by streams
    courses = filter_courses_by_streams(all_courses, config.streams)
    print(f"    After stream filter: {len(courses)}")
    
    # Apply limit (returns more courses than limit to account for skipped ones)
    courses = get_courses_with_limit(courses, config.limit, config.streams)
    print(f"    Candidates to scrape: {len(courses)} (target: {config.limit or 'all'})")
    
    if not courses:
        print("\n  ERROR: No courses match the filter criteria!")
        return stats
    
    # Phase 1 & 2: Scraping
    if not config.skip_scraping:
        asyncio.run(run_scraping_phase(config, builder, courses, stats))
    else:
        print("\n  Skipping scraping phase (--skip-scraping)")
    
    # Phase 3: Concept extraction
    if not config.skip_concepts:
        run_concept_extraction_phase(config, builder, stats, config.concurrency)
    else:
        print("\n  Skipping concept extraction phase (--skip-concepts)")
    
    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    final_stats = builder.get_stats()
    print(f"\nFinal Graph:")
    for node_type, count in final_stats['nodes_by_type'].items():
        print(f"  {node_type}: {count}")
    
    print(f"\nProcessing Stats:")
    print(f"  Courses: {stats.courses_scraped} scraped, {stats.courses_skipped} skipped")
    print(f"  Lectures: {stats.lectures_processed} processed, {stats.lectures_skipped} skipped")
    print(f"  Concepts: {stats.concepts_extracted} extracted")
    print(f"  API calls: {stats.api_calls}")
    
    if stats.errors:
        print(f"\nErrors ({len(stats.errors)}):")
        for err in stats.errors[:10]:
            print(f"  - {err[:80]}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='NPTEL Knowledge Graph Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all streams, no limit
  python pipeline.py --api-key YOUR_KEY
  
  # Run on specific streams with limit
  python pipeline.py --api-key YOUR_KEY --stream "Mathematics" --stream "Computer Science" --limit 50
  
  # Resume previous run (will skip already processed items)
  python pipeline.py --api-key YOUR_KEY --stream "Mathematics" --limit 100
  
  # Only scrape, skip concept extraction
  python pipeline.py --api-key YOUR_KEY --skip-concepts
        """
    )
    
    parser.add_argument(
        '--api-key', '-k',
        type=str,
        required=True,
        help='OpenRouter API key'
    )
    parser.add_argument(
        '--stream', '-s',
        type=str,
        action='append',
        dest='streams',
        help='Filter by stream name (can be used multiple times)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit total courses (divided equally among streams)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for JSONL files'
    )
    parser.add_argument(
        '--scrape-delay',
        type=float,
        default=1.5,
        help='Delay between scraping requests (seconds)'
    )
    parser.add_argument(
        '--api-delay',
        type=float,
        default=2.0,
        help='Delay between API calls (seconds)'
    )
    parser.add_argument(
        '--skip-scraping',
        action='store_true',
        help='Skip scraping phase (use existing data)'
    )
    parser.add_argument(
        '--skip-concepts',
        action='store_true',
        help='Skip concept extraction phase'
    )
    parser.add_argument(
        '--concurrency', '-c',
        type=int,
        default=CONCURRENCY_LIMIT,
        help=f'Number of parallel API calls (default: {CONCURRENCY_LIMIT})'
    )
    
    args = parser.parse_args()
    
    # Build config
    base_dir = Path(__file__).parent.parent
    config = PipelineConfig(
        excel_path=base_dir / EXCEL_FILE,
        output_dir=base_dir / args.output,
        api_key=args.api_key,
        streams=args.streams,
        limit=args.limit,
        scrape_delay=args.scrape_delay,
        api_delay=args.api_delay,
        skip_scraping=args.skip_scraping,
        skip_concepts=args.skip_concepts,
        concurrency=args.concurrency
    )
    
    run_pipeline(config)


if __name__ == '__main__':
    main()
