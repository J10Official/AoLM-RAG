"""
Validation utility for inspecting the generated graph.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    items = []
    with open(filepath, 'r') as f:
        for line in f:
            items.append(json.loads(line))
    return items


def validate_graph(output_dir: Path) -> Dict[str, Any]:
    """
    Validate and summarize the graph structure.
    
    Returns:
        Dictionary with validation results
    """
    nodes_file = output_dir / 'nodes.jsonl'
    edges_file = output_dir / 'edges.jsonl'
    
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check files exist
    if not nodes_file.exists():
        results['valid'] = False
        results['errors'].append(f"nodes.jsonl not found at {nodes_file}")
        return results
        
    if not edges_file.exists():
        results['valid'] = False
        results['errors'].append(f"edges.jsonl not found at {edges_file}")
        return results
    
    # Load data
    nodes = load_jsonl(nodes_file)
    edges = load_jsonl(edges_file)
    
    # Build node index
    node_ids = {n['id']: n for n in nodes}
    
    # Count by type
    nodes_by_type = defaultdict(list)
    for node in nodes:
        nodes_by_type[node['type']].append(node)
        
    edges_by_type = defaultdict(list)
    for edge in edges:
        edges_by_type[edge['type']].append(edge)
    
    # Validate edges reference valid nodes
    orphan_edges = []
    for edge in edges:
        if edge['source'] not in node_ids:
            orphan_edges.append(f"Edge source '{edge['source']}' not in nodes")
        if edge['target'] not in node_ids:
            orphan_edges.append(f"Edge target '{edge['target']}' not in nodes")
            
    if orphan_edges:
        results['warnings'].extend(orphan_edges[:10])  # Limit output
        if len(orphan_edges) > 10:
            results['warnings'].append(f"... and {len(orphan_edges) - 10} more")
    
    # Check for nodes without edges
    nodes_with_edges = set()
    for edge in edges:
        nodes_with_edges.add(edge['source'])
        nodes_with_edges.add(edge['target'])
        
    orphan_nodes = [n['id'] for n in nodes if n['id'] not in nodes_with_edges]
    if orphan_nodes:
        results['warnings'].append(f"Found {len(orphan_nodes)} nodes without edges")
    
    # Build stats
    results['stats'] = {
        'total_nodes': len(nodes),
        'total_edges': len(edges),
        'nodes_by_type': {k: len(v) for k, v in nodes_by_type.items()},
        'edges_by_type': {k: len(v) for k, v in edges_by_type.items()},
    }
    
    # Sample data for each type
    results['samples'] = {}
    for node_type, type_nodes in nodes_by_type.items():
        if type_nodes:
            sample = type_nodes[0]
            results['samples'][node_type] = {
                'id': sample['id'],
                'name': sample['name'],
                'properties_keys': list(sample['properties'].keys())
            }
    
    return results


def print_report(results: Dict[str, Any], verbose: bool = False):
    """Print validation report."""
    print("\n" + "=" * 60)
    print("GRAPH VALIDATION REPORT")
    print("=" * 60)
    
    stats = results['stats']
    
    print(f"\n📊 Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    
    print(f"\n📁 Nodes by type:")
    for node_type, count in stats['nodes_by_type'].items():
        print(f"  {node_type}: {count}")
        
    print(f"\n🔗 Edges by type:")
    for edge_type, count in stats['edges_by_type'].items():
        print(f"  {edge_type}: {count}")
    
    if results['errors']:
        print(f"\n❌ Errors ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"  - {error}")
            
    if results['warnings']:
        print(f"\n⚠️ Warnings ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"  - {warning}")
            
    if verbose and 'samples' in results:
        print(f"\n📝 Sample nodes by type:")
        for node_type, sample in results['samples'].items():
            print(f"\n  {node_type}:")
            print(f"    ID: {sample['id']}")
            print(f"    Name: {sample['name'][:60]}..." if len(sample['name']) > 60 else f"    Name: {sample['name']}")
            print(f"    Properties: {', '.join(sample['properties_keys'])}")
    
    print(f"\n{'✅ Validation PASSED' if results['valid'] else '❌ Validation FAILED'}")
    print("=" * 60)


def print_sample_data(output_dir: Path, node_type: str = None, limit: int = 5):
    """Print sample data from the graph."""
    nodes_file = output_dir / 'nodes.jsonl'
    edges_file = output_dir / 'edges.jsonl'
    
    nodes = load_jsonl(nodes_file)
    edges = load_jsonl(edges_file)
    
    print(f"\n📋 Sample Data (limit={limit})")
    print("=" * 60)
    
    # Filter nodes by type if specified
    if node_type:
        nodes = [n for n in nodes if n['type'] == node_type]
        print(f"\nFiltering by type: {node_type}")
    
    print(f"\n🔵 NODES:")
    for node in nodes[:limit]:
        print(f"\n  [{node['type']}] {node['id']}")
        print(f"    Name: {node['name']}")
        for key, value in node['properties'].items():
            if isinstance(value, str) and len(value) > 80:
                value = value[:80] + "..."
            print(f"    {key}: {value}")
    
    print(f"\n🔗 EDGES:")
    for edge in edges[:limit]:
        print(f"  {edge['source']} --[{edge['type']}]--> {edge['target']}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate and inspect the generated graph'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/output',
        help='Directory containing JSONL files'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=0,
        help='Number of sample records to show (default: 0)'
    )
    parser.add_argument(
        '--type', '-t',
        type=str,
        default=None,
        help='Filter samples by node type'
    )
    
    args = parser.parse_args()
    
    # Resolve path
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / args.output
    
    # Run validation
    results = validate_graph(output_dir)
    print_report(results, verbose=args.verbose)
    
    # Show samples if requested
    if args.samples > 0:
        print_sample_data(output_dir, node_type=args.type, limit=args.samples)
    
    return 0 if results['valid'] else 1


if __name__ == '__main__':
    exit(main())
