"""
Graph builder module for creating JSONL graph representation.
Schema designed for easy Neo4j conversion.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import re


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text


@dataclass
class Node:
    """Base node representation."""
    id: str
    type: str  # V_stream, V_course, V_lecture, V_concept
    name: str
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type,
            'name': self.name,
            'properties': self.properties
        }


@dataclass  
class Edge:
    """Edge/relationship representation."""
    source: str
    target: str
    type: str  # HAS_COURSE, HAS_LECTURE, COVERS_CONCEPT
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'target': self.target,
            'type': self.type,
            'properties': self.properties
        }


class GraphBuilder:
    """
    Builder for the NPTEL knowledge graph.
    
    Graph Structure:
    - V_stream (discipline) -> HAS_COURSE -> V_course
    - V_course -> HAS_LECTURE -> V_lecture
    - V_lecture -> COVERS_CONCEPT -> V_concept (future)
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}  # id -> Node
        self.edges: List[Edge] = []
        
    def add_stream(self, name: str, slug: Optional[str] = None) -> str:
        """
        Add a stream/discipline node.
        
        Args:
            name: Display name of the discipline
            slug: Optional URL-friendly identifier (auto-generated if None)
            
        Returns:
            Node ID
        """
        if slug is None:
            slug = slugify(name)
            
        node_id = f"stream_{slug}"
        
        if node_id not in self.nodes:
            node = Node(
                id=node_id,
                type='V_stream',
                name=name,
                properties={
                    'slug': slug
                }
            )
            self.nodes[node_id] = node
            
        return node_id
    
    def add_course(
        self,
        stream_id: str,
        course_id: str,
        name: str,
        nptel_url: str,
        nptel_id: str,
        abstract: str = "",
        professor: str = "",
        institute: str = "",
        duration: str = "",
    ) -> str:
        """
        Add a course node and link it to its stream.
        
        Args:
            stream_id: ID of the parent stream node
            course_id: Unique course identifier (e.g., noc26-ae01)
            name: Course name
            nptel_url: URL to NPTEL course page
            nptel_id: Numeric NPTEL ID
            abstract: Course description
            professor: Professor/SME name
            institute: Institute name
            duration: Course duration
            
        Returns:
            Node ID
        """
        node_id = f"course_{nptel_id}"
        
        if node_id not in self.nodes:
            node = Node(
                id=node_id,
                type='V_course',
                name=name,
                properties={
                    'course_id': course_id,
                    'nptel_url': nptel_url,
                    'nptel_id': nptel_id,
                    'abstract': abstract,
                    'professor': professor,
                    'institute': institute,
                    'duration': duration,
                    'stream_id': stream_id
                }
            )
            self.nodes[node_id] = node
            
            # Create edge from stream to course
            edge = Edge(
                source=stream_id,
                target=node_id,
                type='HAS_COURSE',
                properties={}
            )
            self.edges.append(edge)
            
        return node_id
    
    def add_lecture(
        self,
        course_id: str,
        name: str,
        week: int,
        lecture_num: int,
        youtube_url: Optional[str] = None,
    ) -> str:
        """
        Add a lecture node and link it to its course.
        
        Args:
            course_id: ID of the parent course node
            name: Lecture name
            week: Week number
            lecture_num: Lecture number within course
            youtube_url: YouTube video URL
            
        Returns:
            Node ID
        """
        # Extract nptel_id from course_id (format: course_NNNNNN)
        nptel_id = course_id.replace('course_', '')
        node_id = f"lecture_{nptel_id}_w{week}_l{lecture_num}"
        
        if node_id not in self.nodes:
            node = Node(
                id=node_id,
                type='V_lecture',
                name=name,
                properties={
                    'week': week,
                    'lecture_num': lecture_num,
                    'youtube_url': youtube_url,
                    'course_id': course_id
                }
            )
            self.nodes[node_id] = node
            
            # Create edge from course to lecture
            edge = Edge(
                source=course_id,
                target=node_id,
                type='HAS_LECTURE',
                properties={
                    'week': week,
                    'lecture_num': lecture_num
                }
            )
            self.edges.append(edge)
            
        return node_id
    
    def add_concept(
        self,
        lecture_id: str,
        name: str,
        description: str = "",
        keywords: Optional[List[str]] = None,
        confidence: float = 1.0
    ) -> str:
        """
        Add a concept node and link it to its lecture.
        If concept already exists, adds a new edge from the lecture.
        
        Args:
            lecture_id: ID of the parent lecture node
            name: Concept name
            description: Concept description
            keywords: Related keywords
            confidence: Confidence score for the concept extraction
            
        Returns:
            Node ID
        """
        slug = slugify(name)
        node_id = f"concept_{slug}"
        
        if node_id not in self.nodes:
            node = Node(
                id=node_id,
                type='V_concept',
                name=name,
                properties={
                    'description': description,
                    'keywords': keywords or [],
                    'source_lectures': [lecture_id]
                }
            )
            self.nodes[node_id] = node
        else:
            # Update existing concept - add to source lectures if not already there
            existing = self.nodes[node_id]
            source_lectures = existing.properties.get('source_lectures', [])
            if lecture_id not in source_lectures:
                source_lectures.append(lecture_id)
                existing.properties['source_lectures'] = source_lectures
            # Merge keywords
            existing_keywords = set(existing.properties.get('keywords', []))
            existing_keywords.update(keywords or [])
            existing.properties['keywords'] = list(existing_keywords)
        
        # Check if edge already exists (avoid duplicates)
        edge_exists = any(
            e.source == lecture_id and e.target == node_id and e.type == 'COVERS_CONCEPT'
            for e in self.edges
        )
        
        if not edge_exists:
            edge = Edge(
                source=lecture_id,
                target=node_id,
                type='COVERS_CONCEPT',
                properties={
                    'confidence': confidence
                }
            )
            self.edges.append(edge)
            
        return node_id
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the graph."""
        node_counts = {}
        for node in self.nodes.values():
            node_type = node.type
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
            
        edge_counts = {}
        for edge in self.edges:
            edge_type = edge.type
            edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1
            
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'nodes_by_type': node_counts,
            'edges_by_type': edge_counts
        }
    
    def save_to_jsonl(self, output_dir: str | Path):
        """
        Save graph to JSONL files.
        
        Creates:
        - nodes.jsonl: One node per line
        - edges.jsonl: One edge per line
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        nodes_file = output_dir / 'nodes.jsonl'
        edges_file = output_dir / 'edges.jsonl'
        
        # Write nodes
        with open(nodes_file, 'w') as f:
            for node in self.nodes.values():
                f.write(json.dumps(node.to_dict()) + '\n')
                
        # Write edges
        with open(edges_file, 'w') as f:
            for edge in self.edges:
                f.write(json.dumps(edge.to_dict()) + '\n')
                
        print(f"Saved {len(self.nodes)} nodes to {nodes_file}")
        print(f"Saved {len(self.edges)} edges to {edges_file}")
        
    def load_from_jsonl(self, input_dir: str | Path):
        """
        Load graph from JSONL files.
        """
        input_dir = Path(input_dir)
        nodes_file = input_dir / 'nodes.jsonl'
        edges_file = input_dir / 'edges.jsonl'
        
        # Load nodes
        if nodes_file.exists():
            with open(nodes_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    node = Node(
                        id=data['id'],
                        type=data['type'],
                        name=data['name'],
                        properties=data['properties']
                    )
                    self.nodes[node.id] = node
                    
        # Load edges
        if edges_file.exists():
            with open(edges_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    edge = Edge(
                        source=data['source'],
                        target=data['target'],
                        type=data['type'],
                        properties=data['properties']
                    )
                    self.edges.append(edge)


if __name__ == '__main__':
    # Test the graph builder
    builder = GraphBuilder()
    
    # Add sample data
    stream_id = builder.add_stream("Computer Science and Engineering")
    course_id = builder.add_course(
        stream_id=stream_id,
        course_id="noc26-cs01",
        name="Introduction to Programming",
        nptel_url="https://nptel.ac.in/courses/106101001",
        nptel_id="106101001",
        abstract="Learn the basics of programming",
        professor="Prof. John Doe",
        institute="IIT Example",
        duration="12 Weeks"
    )
    
    lecture_id = builder.add_lecture(
        course_id=course_id,
        name="Lecture 1: Getting Started",
        week=1,
        lecture_num=1,
        youtube_url="https://www.youtube.com/watch?v=example"
    )
    
    print("Graph Stats:")
    stats = builder.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Save to JSONL
    from pathlib import Path
    output_dir = Path(__file__).parent.parent / 'data' / 'output'
    builder.save_to_jsonl(output_dir)
    
    print("\nSample node:")
    print(json.dumps(list(builder.nodes.values())[0].to_dict(), indent=2))
