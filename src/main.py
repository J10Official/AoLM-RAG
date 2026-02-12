"""
Main orchestrator for building the NPTEL knowledge graph.
Coordinates Excel parsing, web scraping, and graph construction.
"""
import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.excel_parser import get_courses, get_unique_disciplines
from src.nptel_scraper import NPTELScraper
from src.graph_builder import GraphBuilder


async def build_graph(
    excel_path: Path,
    output_dir: Path,
    limit: Optional[int] = None,
    headless: bool = True,
    delay: float = 1.5
) -> GraphBuilder:
    """
    Build the NPTEL knowledge graph.
    
    Args:
        excel_path: Path to the Excel file with course list
        output_dir: Directory to save output JSONL files
        limit: Maximum number of courses to process (None for all)
        headless: Whether to run browser in headless mode
        delay: Delay between requests in seconds
        
    Returns:
        GraphBuilder with the constructed graph
    """
    print(f"Building NPTEL Knowledge Graph")
    print(f"=" * 50)
    
    # Step 1: Read courses from Excel
    print(f"\n[1/4] Reading courses from Excel...")
    courses = get_courses(excel_path, limit=limit)
    print(f"  Found {len(courses)} courses to process")
    
    # Step 2: Create stream nodes
    print(f"\n[2/4] Creating stream nodes...")
    builder = GraphBuilder()
    disciplines = get_unique_disciplines(courses)
    
    for disc in disciplines:
        builder.add_stream(name=disc['name'], slug=disc['slug'])
    print(f"  Created {len(disciplines)} stream nodes")
    
    # Step 3: Scrape course details and create nodes
    print(f"\n[3/4] Scraping course details and lectures...")
    
    async with NPTELScraper(headless=headless) as scraper:
        for course in tqdm(courses, desc="Processing courses"):
            try:
                # Get course details from NPTEL
                details = await scraper.get_course_details(course['nptel_url'])
                
                if details.error:
                    print(f"\n  Warning: {course['course_name']}: {details.error}")
                
                # Create course node
                stream_id = f"stream_{course['discipline_slug']}"
                course_node_id = builder.add_course(
                    stream_id=stream_id,
                    course_id=course['course_id'],
                    name=course['course_name'],
                    nptel_url=course['nptel_url'],
                    nptel_id=course['nptel_id'],
                    abstract=details.abstract or "",
                    professor=course['professor'],
                    institute=course['institute'],
                    duration=course['duration']
                )
                
                # Create lecture nodes
                for lecture in details.lectures:
                    builder.add_lecture(
                        course_id=course_node_id,
                        name=lecture.name,
                        week=lecture.week,
                        lecture_num=lecture.lecture_num,
                        youtube_url=lecture.youtube_url
                    )
                
                # Rate limiting
                await asyncio.sleep(delay)
                
            except Exception as e:
                print(f"\n  Error processing {course['course_name']}: {e}")
                continue
    
    # Step 4: Save graph
    print(f"\n[4/4] Saving graph to JSONL files...")
    builder.save_to_jsonl(output_dir)
    
    # Print summary
    stats = builder.get_stats()
    print(f"\n" + "=" * 50)
    print("Graph Construction Complete!")
    print(f"=" * 50)
    print(f"\nSummary:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"\n  Nodes by type:")
    for node_type, count in stats['nodes_by_type'].items():
        print(f"    {node_type}: {count}")
    print(f"\n  Edges by type:")
    for edge_type, count in stats['edges_by_type'].items():
        print(f"    {edge_type}: {count}")
    print(f"\nOutput files:")
    print(f"  {output_dir / 'nodes.jsonl'}")
    print(f"  {output_dir / 'edges.jsonl'}")
    
    return builder


def main():
    parser = argparse.ArgumentParser(
        description='Build NPTEL knowledge graph from course list'
    )
    parser.add_argument(
        '--excel', '-e',
        type=str,
        default='Final Course List (Jan - Apr 2026).xlsx',
        help='Path to Excel file with course list'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/output',
        help='Output directory for JSONL files'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Maximum number of courses to process (default: all)'
    )
    parser.add_argument(
        '--visible',
        action='store_true',
        help='Run browser in visible mode (not headless)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.5,
        help='Delay between requests in seconds (default: 1.5)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = Path(__file__).parent.parent
    excel_path = base_dir / args.excel
    output_dir = base_dir / args.output
    
    if not excel_path.exists():
        print(f"Error: Excel file not found: {excel_path}")
        return 1
    
    # Run the async builder
    asyncio.run(build_graph(
        excel_path=excel_path,
        output_dir=output_dir,
        limit=args.limit,
        headless=not args.visible,
        delay=args.delay
    ))
    
    return 0


if __name__ == '__main__':
    exit(main())
