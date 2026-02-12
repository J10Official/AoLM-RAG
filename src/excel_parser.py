"""
Excel parser module for reading NPTEL course list.
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import re


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text


def get_courses(
    excel_path: str | Path,
    limit: Optional[int] = None
) -> List[Dict]:
    """
    Read courses from the NPTEL Excel file.
    
    Args:
        excel_path: Path to the Excel file
        limit: Maximum number of courses to return (None for all)
    
    Returns:
        List of course dictionaries with keys:
        - course_id: Unique course identifier (e.g., noc26-ae01)
        - course_name: Name of the course
        - discipline: Stream/discipline name
        - professor: SME/Professor name
        - institute: Institute name
        - duration: Course duration
        - nptel_url: URL to NPTEL course page
        - nptel_id: Numeric NPTEL course ID
    """
    df = pd.read_excel(excel_path, skiprows=10)
    
    # Clean up column names
    df.columns = df.columns.str.strip()
    
    # Filter out rows where essential fields are NaN
    df = df.dropna(subset=['Course Name', 'NPTEL URL'])
    
    # Select and rename columns
    courses = []
    for idx, row in df.iterrows():
        if limit and len(courses) >= limit:
            break
            
        # Skip if NPTEL URL is not valid
        nptel_url = str(row.get('NPTEL URL', '')).strip()
        if not nptel_url.startswith('http'):
            continue
        
        # Extract NPTEL ID from URL
        nptel_id = nptel_url.rstrip('/').split('/')[-1]
        
        # Handle multi-line disciplines (take first one)
        discipline = str(row.get('Discipline', 'Unknown'))
        if '\n' in discipline:
            discipline = discipline.split('\n')[0].strip()
        
        course = {
            'course_id': str(row.get('Course ID', '')).strip(),
            'course_name': str(row.get('Course Name', '')).strip(),
            'discipline': discipline.strip(),
            'professor': str(row.get('SME Name', '')).strip(),
            'institute': str(row.get('Institute', '')).strip(),
            'duration': str(row.get('Duration', '')).strip(),
            'nptel_url': nptel_url,
            'nptel_id': nptel_id,
            'discipline_slug': slugify(discipline),
        }
        courses.append(course)
    
    return courses


def get_unique_disciplines(courses: List[Dict]) -> List[Dict]:
    """
    Extract unique disciplines from course list.
    
    Returns:
        List of discipline dictionaries with keys:
        - name: Discipline name
        - slug: URL-friendly slug
    """
    seen = set()
    disciplines = []
    
    for course in courses:
        slug = course['discipline_slug']
        if slug not in seen:
            seen.add(slug)
            disciplines.append({
                'name': course['discipline'],
                'slug': slug,
            })
    
    return disciplines


if __name__ == '__main__':
    # Test the parser
    import sys
    
    excel_file = Path(__file__).parent.parent / 'Final Course List (Jan - Apr 2026).xlsx'
    
    print("Reading courses...")
    courses = get_courses(excel_file, limit=5)
    
    print(f"\nFound {len(courses)} courses:")
    for c in courses:
        print(f"  - {c['course_name']} ({c['discipline']})")
        print(f"    URL: {c['nptel_url']}")
    
    print("\nUnique disciplines:")
    for d in get_unique_disciplines(courses):
        print(f"  - {d['name']} (slug: {d['slug']})")
