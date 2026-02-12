"""
Graph Visualizer for NPTEL Knowledge Graph.
Creates interactive HTML visualization using pyvis.

Supports mastery visualization:
- Concept nodes: Green (mastered) -> Gray (not studied)
- Lecture nodes: Completion-based coloring (red=low -> green=high)
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime

try:
    from pyvis.network import Network
except ImportError:
    print("pyvis not installed. Install with: pip install pyvis")
    exit(1)


# Base node type colors (used when no mastery data)
NODE_COLORS = {
    'V_stream': '#e74c3c',      # Red
    'V_course': '#3498db',      # Blue
    'V_lecture': '#2ecc71',     # Green
    'V_concept': '#9b59b6',     # Purple
}

# Neutral colors for nodes without mastery data
NEUTRAL_COLORS = {
    'V_lecture': '#7f8c8d',     # Gray for untracked lectures
    'V_concept': '#95a5a6',     # Lighter gray for untracked concepts
}

# Node type sizes
NODE_SIZES = {
    'V_stream': 50,
    'V_course': 35,
    'V_lecture': 20,
    'V_concept': 15,
}

# Edge colors
EDGE_COLORS = {
    'HAS_COURSE': '#e74c3c',
    'HAS_LECTURE': '#3498db',
    'COVERS_CONCEPT': '#9b59b6',
}


def retention_to_color(retention: float) -> str:
    """
    Convert retention (0-1) to a color gradient.
    
    0.0 = Gray (#95a5a6) - not studied
    0.01-0.4 = Red shades (struggling/forgotten)
    0.4-0.7 = Yellow/Orange (needs review)
    0.7-1.0 = Green shades (mastered)
    """
    if retention <= 0:
        return NEUTRAL_COLORS['V_concept']
    
    if retention < 0.4:
        # Gray to Red (0 -> 0.4)
        t = retention / 0.4
        # Gray RGB(149, 165, 166) -> Red RGB(231, 76, 60)
        r = int(149 + (231 - 149) * t)
        g = int(165 + (76 - 165) * t)
        b = int(166 + (60 - 166) * t)
    elif retention < 0.7:
        # Red to Yellow (0.4 -> 0.7)
        t = (retention - 0.4) / 0.3
        # Red RGB(231, 76, 60) -> Yellow RGB(241, 196, 15)
        r = int(231 + (241 - 231) * t)
        g = int(76 + (196 - 76) * t)
        b = int(60 + (15 - 60) * t)
    else:
        # Yellow to Green (0.7 -> 1.0)
        t = (retention - 0.7) / 0.3
        # Yellow RGB(241, 196, 15) -> Green RGB(39, 174, 96)
        r = int(241 + (39 - 241) * t)
        g = int(196 + (174 - 196) * t)
        b = int(15 + (96 - 15) * t)
    
    return f'#{r:02x}{g:02x}{b:02x}'


def completion_to_color(completion: float) -> str:
    """
    Convert lecture completion (0-1) to a color gradient.
    
    0.0 = Gray (#7f8c8d) - not started
    0.01-0.4 = Red shades (low completion)
    0.4-0.7 = Yellow/Orange (partial)
    0.7-1.0 = Green shades (mostly complete)
    """
    if completion <= 0:
        return NEUTRAL_COLORS['V_lecture']
    
    if completion < 0.4:
        # Gray to Red (0 -> 0.4)
        t = completion / 0.4
        # Gray RGB(127, 140, 141) -> Red RGB(231, 76, 60)
        r = int(127 + (231 - 127) * t)
        g = int(140 + (76 - 140) * t)
        b = int(141 + (60 - 141) * t)
    elif completion < 0.7:
        # Red to Yellow (0.4 -> 0.7)
        t = (completion - 0.4) / 0.3
        # Red RGB(231, 76, 60) -> Yellow RGB(241, 196, 15)
        r = int(231 + (241 - 231) * t)
        g = int(76 + (196 - 76) * t)
        b = int(60 + (15 - 60) * t)
    else:
        # Yellow to Green (0.7 -> 1.0)
        t = (completion - 0.7) / 0.3
        # Yellow RGB(241, 196, 15) -> Green RGB(39, 174, 96)
        r = int(241 + (39 - 241) * t)
        g = int(196 + (174 - 196) * t)
        b = int(15 + (96 - 15) * t)
    
    return f'#{r:02x}{g:02x}{b:02x}'


def load_mastery_data(
    data_dir: Path,
    student_id: str
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    Load student mastery data and build lecture->concepts mapping.
    
    Returns:
        (concept_retentions, lecture_concepts)
        - concept_retentions: Dict[concept_id, retention 0-1]
        - lecture_concepts: Dict[lecture_id, List[concept_id]]
    """
    from src.student_mastery import MasteryTracker
    
    tracker = MasteryTracker(data_dir)
    
    # Get current retentions for all studied concepts
    concept_retentions = tracker.get_all_masteries(student_id, datetime.now())
    
    # Build lecture -> concepts mapping from edges
    lecture_concepts = defaultdict(list)
    edges_file = data_dir / 'edges.jsonl'
    
    if edges_file.exists():
        with open(edges_file, 'r') as f:
            for line in f:
                if line.strip():
                    edge = json.loads(line)
                    if edge['type'] == 'COVERS_CONCEPT':
                        lecture_concepts[edge['source']].append(edge['target'])
    
    return concept_retentions, dict(lecture_concepts)


def calculate_lecture_completions(
    concept_retentions: Dict[str, float],
    lecture_concepts: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Calculate lecture completion percentages from concept mastery.
    
    Completion = average retention of concepts covered by the lecture.
    """
    completions = {}
    
    for lecture_id, concept_ids in lecture_concepts.items():
        if not concept_ids:
            completions[lecture_id] = 0.0
            continue
        
        retentions = [concept_retentions.get(cid, 0.0) for cid in concept_ids]
        completions[lecture_id] = sum(retentions) / len(retentions)
    
    return completions


def load_graph(data_dir: Path) -> tuple[Dict[str, dict], List[dict]]:
    """Load nodes and edges from JSONL files."""
    nodes = {}
    edges = []
    
    nodes_file = data_dir / 'nodes.jsonl'
    edges_file = data_dir / 'edges.jsonl'
    
    if not nodes_file.exists():
        raise FileNotFoundError(f"Nodes file not found: {nodes_file}")
    
    # Load nodes
    with open(nodes_file, 'r') as f:
        for line in f:
            if line.strip():
                node = json.loads(line)
                nodes[node['id']] = node
    
    # Load edges
    if edges_file.exists():
        with open(edges_file, 'r') as f:
            for line in f:
                if line.strip():
                    edges.append(json.loads(line))
    
    return nodes, edges


def get_graph_stats(nodes: Dict[str, dict], edges: List[dict]) -> Dict:
    """Get statistics about the graph."""
    type_counts = defaultdict(int)
    edge_type_counts = defaultdict(int)
    
    for node in nodes.values():
        type_counts[node['type']] += 1
    
    for edge in edges:
        edge_type_counts[edge['type']] += 1
    
    return {
        'total_nodes': len(nodes),
        'total_edges': len(edges),
        'node_types': dict(type_counts),
        'edge_types': dict(edge_type_counts),
    }


def filter_graph(
    nodes: Dict[str, dict],
    edges: List[dict],
    node_types: Optional[Set[str]] = None,
    max_concepts: int = 0,
    course_filter: Optional[str] = None,
) -> tuple[Dict[str, dict], List[dict]]:
    """
    Filter graph to manageable size.
    
    Args:
        nodes: All nodes
        edges: All edges
        node_types: Set of node types to include (None = all)
        max_concepts: Max concepts to include (0 = all)
        course_filter: Only show this course and its descendants
    """
    filtered_nodes = {}
    filtered_edges = []
    
    # If course filter specified, find all related nodes
    if course_filter:
        related_ids = set()
        course_id = None
        
        # Find the course
        for nid, node in nodes.items():
            if node['type'] == 'V_course' and course_filter.lower() in node['name'].lower():
                course_id = nid
                related_ids.add(nid)
                break
        
        if course_id:
            # Find parent stream
            for edge in edges:
                if edge['target'] == course_id and edge['type'] == 'HAS_COURSE':
                    related_ids.add(edge['source'])
            
            # Find lectures
            lecture_ids = set()
            for edge in edges:
                if edge['source'] == course_id and edge['type'] == 'HAS_LECTURE':
                    related_ids.add(edge['target'])
                    lecture_ids.add(edge['target'])
            
            # Find concepts (limited)
            concept_count = 0
            for edge in edges:
                if edge['source'] in lecture_ids and edge['type'] == 'COVERS_CONCEPT':
                    if max_concepts == 0 or concept_count < max_concepts:
                        related_ids.add(edge['target'])
                        concept_count += 1
            
            # Filter nodes
            filtered_nodes = {nid: node for nid, node in nodes.items() if nid in related_ids}
            
            # Filter edges
            filtered_edges = [
                e for e in edges
                if e['source'] in related_ids and e['target'] in related_ids
            ]
            
            return filtered_nodes, filtered_edges
    
    # Filter by node types
    if node_types:
        filtered_nodes = {
            nid: node for nid, node in nodes.items()
            if node['type'] in node_types
        }
    else:
        filtered_nodes = dict(nodes)
    
    # Limit concepts if requested
    if max_concepts > 0 and 'V_concept' in (node_types or {'V_concept'}):
        concept_ids = [
            nid for nid, node in filtered_nodes.items()
            if node['type'] == 'V_concept'
        ]
        if len(concept_ids) > max_concepts:
            # Keep only first N concepts
            to_remove = set(concept_ids[max_concepts:])
            filtered_nodes = {
                nid: node for nid, node in filtered_nodes.items()
                if nid not in to_remove
            }
    
    # Filter edges to only include nodes in filtered set
    filtered_ids = set(filtered_nodes.keys())
    filtered_edges = [
        e for e in edges
        if e['source'] in filtered_ids and e['target'] in filtered_ids
    ]
    
    return filtered_nodes, filtered_edges


def filter_to_studied_subgraph(
    nodes: Dict[str, dict],
    edges: List[dict],
    concept_retentions: Dict[str, float],
    lecture_completions: Dict[str, float],
) -> tuple[Dict[str, dict], List[dict]]:
    """
    Filter graph to only show nodes with mastery data.
    
    Includes:
    - Concepts with retention > 0
    - Lectures with completion > 0
    - Parent courses and streams for context
    """
    included_ids = set()
    
    # Add studied concepts
    for concept_id, retention in concept_retentions.items():
        if retention > 0:
            included_ids.add(concept_id)
    
    # Add tracked lectures
    for lecture_id, completion in lecture_completions.items():
        if completion > 0:
            included_ids.add(lecture_id)
    
    # Find parent courses and streams
    course_ids = set()
    for edge in edges:
        if edge['type'] == 'HAS_LECTURE' and edge['target'] in included_ids:
            course_ids.add(edge['source'])
            included_ids.add(edge['source'])
    
    for edge in edges:
        if edge['type'] == 'HAS_COURSE' and edge['target'] in course_ids:
            included_ids.add(edge['source'])
    
    # Filter nodes and edges
    filtered_nodes = {nid: node for nid, node in nodes.items() if nid in included_ids}
    filtered_edges = [
        e for e in edges
        if e['source'] in included_ids and e['target'] in included_ids
    ]
    
    return filtered_nodes, filtered_edges


def create_visualization(
    nodes: Dict[str, dict],
    edges: List[dict],
    output_path: Path,
    height: str = "900px",
    width: str = "100%",
    physics: bool = True,
    hierarchical: bool = False,
    radial: bool = False,
    concept_retentions: Dict[str, float] = None,
    lecture_completions: Dict[str, float] = None,
    auto_freeze: bool = False,
) -> None:
    """
    Create interactive HTML visualization.
    
    Args:
        nodes: Graph nodes
        edges: Graph edges
        output_path: Output HTML file path
        height/width: Canvas dimensions
        physics: Enable physics simulation
        hierarchical: Use hierarchical layout
        radial: Use radial layout
        concept_retentions: Optional dict of concept_id -> retention (0-1) for mastery coloring
        lecture_completions: Optional dict of lecture_id -> completion (0-1) for progress coloring
        auto_freeze: If True, freeze physics after 5 seconds to save CPU
    """
    import math
    
    concept_retentions = concept_retentions or {}
    lecture_completions = lecture_completions or {}
    
    # Create network
    net = Network(
        height=height,
        width=width,
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
        select_menu=True,
        filter_menu=True,
    )
    
    # Pre-compute positions for radial layout
    positions = {}
    if radial:
        # Build course -> lectures mapping
        course_lectures = defaultdict(list)
        stream_courses = defaultdict(list)
        
        for edge in edges:
            if edge['type'] == 'HAS_LECTURE':
                course_lectures[edge['source']].append(edge['target'])
            elif edge['type'] == 'HAS_COURSE':
                stream_courses[edge['source']].append(edge['target'])
        
        # Find all courses and streams
        courses = [nid for nid, n in nodes.items() if n['type'] == 'V_course']
        streams = [nid for nid, n in nodes.items() if n['type'] == 'V_stream']
        
        # Position streams at center
        for i, stream_id in enumerate(streams):
            angle = (2 * math.pi * i / max(len(streams), 1))
            positions[stream_id] = (math.cos(angle) * 100, math.sin(angle) * 100)
        
        # Position courses in a ring around streams
        num_courses = len(courses)
        course_radius = 400
        for i, course_id in enumerate(courses):
            angle = 2 * math.pi * i / max(num_courses, 1)
            positions[course_id] = (
                math.cos(angle) * course_radius,
                math.sin(angle) * course_radius
            )
        
        # Position lectures in clusters around their course
        for course_id, lectures in course_lectures.items():
            if course_id not in positions:
                continue
            cx, cy = positions[course_id]
            num_lec = len(lectures)
            
            # Arrange in a small arc/cluster around the course
            lecture_radius = 150 + num_lec * 2  # Scale with number of lectures
            # Get the angle of the course from center
            course_angle = math.atan2(cy, cx)
            # Spread lectures in an arc centered on the outward direction
            arc_span = min(math.pi * 0.8, num_lec * 0.05)  # Max 144 degrees
            
            for j, lec_id in enumerate(lectures):
                if num_lec == 1:
                    lec_angle = course_angle
                else:
                    lec_angle = course_angle - arc_span/2 + (arc_span * j / (num_lec - 1))
                positions[lec_id] = (
                    cx + math.cos(lec_angle) * lecture_radius,
                    cy + math.sin(lec_angle) * lecture_radius
                )
    
    # Configure physics/layout
    if radial:
        # Radial uses pre-computed positions with gentle physics for fine-tuning
        net.set_options("""
        {
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -2000,
                    "centralGravity": 0.1,
                    "springLength": 100,
                    "springConstant": 0.02,
                    "damping": 0.5
                },
                "minVelocity": 0.75,
                "solver": "barnesHut",
                "stabilization": {
                    "enabled": true,
                    "iterations": 100
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "navigationButtons": true,
                "keyboard": true,
                "zoomView": true,
                "dragView": true
            }
        }
        """)
    elif hierarchical:
        # Horizontal hierarchical - much better for bottom-heavy graphs
        net.set_options("""
        {
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "LR",
                    "sortMethod": "directed",
                    "levelSeparation": 300,
                    "nodeSpacing": 50,
                    "treeSpacing": 100,
                    "blockShifting": true,
                    "edgeMinimization": true,
                    "parentCentralization": true
                }
            },
            "physics": {
                "enabled": false
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "navigationButtons": true,
                "keyboard": true,
                "zoomView": true
            }
        }
        """)
    elif physics:
        # Force-directed with strong clustering
        net.set_options("""
        {
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -8000,
                    "centralGravity": 0.5,
                    "springLength": 150,
                    "springConstant": 0.04,
                    "damping": 0.3,
                    "avoidOverlap": 0.5
                },
                "minVelocity": 0.75,
                "solver": "barnesHut",
                "stabilization": {
                    "enabled": true,
                    "iterations": 500,
                    "updateInterval": 50
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "navigationButtons": true,
                "keyboard": true
            }
        }
        """)
    else:
        net.set_options("""
        {
            "physics": {
                "enabled": false
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "navigationButtons": true,
                "keyboard": true
            }
        }
        """)
    
    # Add nodes
    for node_id, node in nodes.items():
        node_type = node['type']
        size = NODE_SIZES.get(node_type, 20)
        
        # Determine color based on mastery data
        if node_type == 'V_concept' and concept_retentions:
            retention = concept_retentions.get(node_id, 0.0)
            color = retention_to_color(retention)
        elif node_type == 'V_lecture' and lecture_completions:
            completion = lecture_completions.get(node_id, 0.0)
            color = completion_to_color(completion)
        else:
            color = NODE_COLORS.get(node_type, '#ffffff')
        
        # Build title (tooltip)
        title_parts = [f"<b>{node['name']}</b>", f"Type: {node_type}"]
        
        # Add mastery info to tooltip
        if node_type == 'V_concept' and concept_retentions:
            retention = concept_retentions.get(node_id, 0.0)
            title_parts.append(f"Retention: {retention*100:.1f}%")
        elif node_type == 'V_lecture' and lecture_completions:
            completion = lecture_completions.get(node_id, 0.0)
            title_parts.append(f"Completion: {completion*100:.1f}%")
        
        if 'properties' in node and node['properties']:
            for key, value in node['properties'].items():
                if key not in ('name', 'id', 'type') and value:
                    title_parts.append(f"{key}: {value}")
        title = "<br>".join(title_parts)
        
        # Truncate label for display
        label = node['name'][:30] + "..." if len(node['name']) > 30 else node['name']
        
        # Use pre-computed position if available (radial layout)
        # Note: Don't use 'group' parameter when using custom colors - it overrides color
        if node_id in positions:
            x, y = positions[node_id]
            net.add_node(
                node_id,
                label=label,
                title=title,
                color=color,
                size=size,
                x=x,
                y=y,
            )
        else:
            net.add_node(
                node_id,
                label=label,
                title=title,
                color=color,
                size=size,
            )
    
    # Add edges
    for edge in edges:
        edge_type = edge['type']
        color = EDGE_COLORS.get(edge_type, '#ffffff')
        
        net.add_edge(
            edge['source'],
            edge['target'],
            title=edge_type,
            color=color,
            arrows='to',
        )
    
    # Save
    net.save_graph(str(output_path))
    
    # Post-process HTML to add hideable toolbar
    has_mastery = bool(concept_retentions) or bool(lecture_completions)
    _inject_toggle_toolbar(output_path, has_mastery, auto_freeze)
    
    print(f"Visualization saved to: {output_path}")


def _inject_toggle_toolbar(html_path: Path, has_mastery: bool = False, auto_freeze: bool = False) -> None:
    """Inject CSS/JS to make the toolbar hideable with a click, and optionally auto-freeze physics."""
    with open(html_path, 'r') as f:
        html = f.read()
    
    # CSS for toggle button and hidden state
    toggle_css = """
        <style>
            #toolbar-toggle {
                position: fixed;
                top: 10px;
                right: 10px;
                z-index: 9999;
                background: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            }
            #toolbar-toggle:hover {
                background: #2980b9;
            }
            .card-header.hidden {
                display: none !important;
            }
            .toolbar-hidden #mynetwork {
                height: 100vh !important;
            }
            .toolbar-hidden .card {
                border: none !important;
            }
            #mastery-legend {
                position: fixed;
                bottom: 20px;
                left: 20px;
                z-index: 9999;
                background: rgba(26, 26, 46, 0.95);
                color: white;
                padding: 15px;
                border-radius: 8px;
                font-size: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.5);
                max-width: 200px;
            }
            #mastery-legend h4 {
                margin: 0 0 10px 0;
                font-size: 14px;
                border-bottom: 1px solid #444;
                padding-bottom: 5px;
            }
            .legend-item {
                display: flex;
                align-items: center;
                margin: 5px 0;
            }
            .legend-color {
                width: 20px;
                height: 12px;
                margin-right: 8px;
                border-radius: 2px;
            }
            .legend-gradient {
                width: 100px;
                height: 12px;
                margin-right: 8px;
                border-radius: 2px;
            }
            .toolbar-hidden #mastery-legend {
                display: none;
            }
        </style>
    """
    
    # Mastery legend HTML
    legend_html = ""
    if has_mastery:
        legend_html = """
        <div id="mastery-legend">
            <h4>Mastery Legend</h4>
            <div><strong>Concepts:</strong></div>
            <div class="legend-item">
                <div class="legend-color" style="background: #95a5a6;"></div>
                <span>Not studied</span>
            </div>
            <div class="legend-item">
                <div class="legend-gradient" style="background: linear-gradient(to right, #95a5a6, #27ae60);"></div>
                <span>Retention</span>
            </div>
            <div style="margin-top: 10px;"><strong>Lectures:</strong></div>
            <div class="legend-item">
                <div class="legend-color" style="background: #7f8c8d;"></div>
                <span>Not started</span>
            </div>
            <div class="legend-item">
                <div class="legend-gradient" style="background: linear-gradient(to right, #e74c3c, #27ae60);"></div>
                <span>Completion</span>
            </div>
        </div>
        """
    
    # JS for toggle functionality and auto-freeze
    auto_freeze_js = ""
    if auto_freeze:
        auto_freeze_js = """
                // Auto-freeze physics after 5 seconds to save CPU
                setTimeout(function() {
                    if (typeof network !== 'undefined') {
                        network.setOptions({ physics: { enabled: false } });
                        console.log('Physics frozen after 5 seconds');
                        
                        // Add indicator
                        var indicator = document.createElement('div');
                        indicator.id = 'freeze-indicator';
                        indicator.style.cssText = 'position:fixed;bottom:20px;right:20px;background:rgba(39,174,96,0.9);color:white;padding:8px 12px;border-radius:5px;font-size:12px;z-index:9999;';
                        indicator.innerHTML = '⏸ Physics frozen (drag nodes to move)';
                        document.body.appendChild(indicator);
                        setTimeout(function() { indicator.style.opacity = '0.6'; }, 2000);
                    }
                }, 5000);
        """
    
    toggle_js = f"""
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                // Create toggle button
                var btn = document.createElement('button');
                btn.id = 'toolbar-toggle';
                btn.innerHTML = '☰ Hide Controls';
                btn.onclick = function() {{
                    var headers = document.querySelectorAll('.card-header');
                    var body = document.body;
                    var isHidden = body.classList.toggle('toolbar-hidden');
                    headers.forEach(function(h) {{ h.classList.toggle('hidden'); }});
                    btn.innerHTML = isHidden ? '☰ Show Controls' : '☰ Hide Controls';
                }};
                document.body.appendChild(btn);
                {auto_freeze_js}
            }});
        </script>
    """
    
    # Inject before </head>
    html = html.replace('</head>', toggle_css + '</head>')
    
    # Inject legend and toggle JS before </body>
    html = html.replace('</body>', legend_html + toggle_js + '</body>')
    
    with open(html_path, 'w') as f:
        f.write(html)


def print_stats(stats: Dict) -> None:
    """Print graph statistics."""
    print("\n" + "=" * 50)
    print("GRAPH STATISTICS")
    print("=" * 50)
    print(f"Total Nodes: {stats['total_nodes']}")
    print(f"Total Edges: {stats['total_edges']}")
    print("\nNode Types:")
    for ntype, count in sorted(stats['node_types'].items()):
        print(f"  {ntype}: {count}")
    print("\nEdge Types:")
    for etype, count in sorted(stats['edge_types'].items()):
        print(f"  {etype}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize NPTEL Knowledge Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Force-directed (default, good for smaller graphs)
  python -m src.visualizer --no-concepts
  
  # Single course with concepts
  python -m src.visualizer --course "Machine Learning" --max-concepts 50
  
  # Hierarchical layout (left-to-right tree)
  python -m src.visualizer --no-concepts --hierarchical
  
  # Radial layout (circular with clusters)
  python -m src.visualizer --no-concepts --radial
  
  # With student mastery visualization (generate demo data first)
  python -m src.visualizer --generate-sample --course "Machine Learning" --max-concepts 100
  
  # Show mastery for existing student
  python -m src.visualizer --student demo_student --course "Machine Learning"
"""
    )
    
    parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        default=Path(__file__).parent.parent / 'data' / 'output',
        help='Directory containing nodes.jsonl and edges.jsonl'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path(__file__).parent.parent / 'graph_visualization.html',
        help='Output HTML file path'
    )
    parser.add_argument(
        '--types', '-t',
        nargs='+',
        choices=['V_stream', 'V_course', 'V_lecture', 'V_concept'],
        help='Node types to include'
    )
    parser.add_argument(
        '--no-concepts',
        action='store_true',
        help='Exclude concept nodes (useful for large graphs)'
    )
    parser.add_argument(
        '--max-concepts',
        type=int,
        default=0,
        help='Maximum number of concepts to show (0 = all)'
    )
    parser.add_argument(
        '--course', '-c',
        type=str,
        help='Filter to single course by name (partial match)'
    )
    parser.add_argument(
        '--hierarchical',
        action='store_true',
        help='Use horizontal hierarchical layout (left-to-right)'
    )
    parser.add_argument(
        '--radial',
        action='store_true',
        help='Use radial/circular layout (courses around center, lectures clustered)'
    )
    parser.add_argument(
        '--no-physics',
        action='store_true',
        help='Disable physics simulation (static layout)'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only print statistics, do not create visualization'
    )
    parser.add_argument(
        '--student', '-s',
        type=str,
        help='Student ID to show mastery data (colors nodes by retention/completion)'
    )
    parser.add_argument(
        '--generate-sample',
        action='store_true',
        help='Generate sample student mastery data for demo visualization'
    )
    parser.add_argument(
        '--mastery-only',
        action='store_true',
        help='Only show nodes with mastery data (concepts/lectures where student has progress)'
    )
    
    args = parser.parse_args()
    
    # Load graph
    print(f"Loading graph from {args.data_dir}...")
    nodes, edges = load_graph(args.data_dir)
    
    # Print stats
    stats = get_graph_stats(nodes, edges)
    print_stats(stats)
    
    if args.stats_only:
        return
    
    # Generate sample student data if requested
    if args.generate_sample:
        from src.student_mastery import generate_sample_student_data
        print("\nGenerating sample student mastery data...")
        generate_sample_student_data(
            args.data_dir,
            args.data_dir / 'nodes.jsonl',
            args.data_dir / 'edges.jsonl',
            student_id='demo_student',
            student_name='Demo Student',
            num_lectures=2,
            days_back=30
        )
        # Default to using the demo student
        if not args.student:
            args.student = 'demo_student'
    
    # Load mastery data if student specified
    concept_retentions = {}
    lecture_completions = {}
    
    if args.student:
        print(f"\nLoading mastery data for student: {args.student}")
        try:
            concept_retentions, lecture_concepts = load_mastery_data(args.data_dir, args.student)
            lecture_completions = calculate_lecture_completions(concept_retentions, lecture_concepts)
            
            studied_concepts = sum(1 for r in concept_retentions.values() if r > 0)
            avg_retention = sum(concept_retentions.values()) / max(len(concept_retentions), 1)
            print(f"  Concepts studied: {studied_concepts}")
            print(f"  Average retention: {avg_retention*100:.1f}%")
            
            started_lectures = sum(1 for c in lecture_completions.values() if c > 0)
            print(f"  Lectures with progress: {started_lectures}")
        except Exception as e:
            print(f"  Warning: Could not load mastery data: {e}")
            print("  Use --generate-sample to create demo data")
    
    # Determine node types to include
    node_types = None
    if args.types:
        node_types = set(args.types)
    elif args.no_concepts:
        node_types = {'V_stream', 'V_course', 'V_lecture'}
    
    # Filter graph
    filtered_nodes, filtered_edges = filter_graph(
        nodes, edges,
        node_types=node_types,
        max_concepts=args.max_concepts,
        course_filter=args.course,
    )
    
    # Apply mastery-only filter if requested
    if args.mastery_only:
        if not args.student:
            print("\nERROR: --mastery-only requires --student or --generate-sample")
            return
        filtered_nodes, filtered_edges = filter_to_studied_subgraph(
            filtered_nodes, filtered_edges,
            concept_retentions, lecture_completions
        )
        print(f"Mastery-only filter: {len(filtered_nodes)} nodes, {len(filtered_edges)} edges")
    
    print(f"\nFiltered graph: {len(filtered_nodes)} nodes, {len(filtered_edges)} edges")
    
    if len(filtered_nodes) > 2000:
        print("\nWARNING: Large graph may be slow to render.")
        print("Consider using --no-concepts, --course, or --max-concepts to reduce size.")
    
    # Create visualization
    # Auto-freeze large graphs without mastery data to save CPU
    should_auto_freeze = len(filtered_nodes) > 500 and not args.mastery_only
    
    print("\nCreating visualization...")
    create_visualization(
        filtered_nodes,
        filtered_edges,
        args.output,
        physics=not args.no_physics,
        hierarchical=args.hierarchical,
        radial=args.radial,
        concept_retentions=concept_retentions,
        lecture_completions=lecture_completions,
        auto_freeze=should_auto_freeze,
    )
    
    print(f"\nOpen {args.output} in a browser to view the graph.")
    print("\nTips:")
    print("  - Use mouse wheel to zoom, drag to pan")
    print("  - Click and drag nodes to rearrange")
    print("  - Hover over nodes for details")
    
    if args.student:
        print("\nMastery Legend:")
        print("  Concepts: Gray (not studied) -> Green (mastered)")
        print("  Lectures: Gray (not started) -> Red (low) -> Green (complete)")
    else:
        print("\nLegend:")
        for ntype, color in NODE_COLORS.items():
            print(f"  {ntype}: {color}")


if __name__ == '__main__':
    main()
