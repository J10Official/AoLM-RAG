"""
Student Mastery Model with Ebbinghaus Forgetting Curve.

Implements spaced repetition memory model based on psychological research.
Uses the Ebbinghaus forgetting curve: R(t) = e^(-t/S)
where:
  - R = retention/recall probability (0 to 1)
  - t = time elapsed since last review (in days)
  - S = memory stability (increases with successful reviews)

Memory stability follows the spacing effect formula from Pimsleur/SuperMemo research:
  S_new = S_base * (1 + D * S_old^(-decay))
where D and decay are empirically derived constants.

References:
  - Ebbinghaus, H. (1885). Memory: A Contribution to Experimental Psychology
  - Pimsleur, P. (1967). A Memory Schedule
  - Wozniak, P. & Gorzelanczyk, E. (1994). Optimization of repetition spacing
"""
import json
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict


# ============================================================================
# FORGETTING CURVE PARAMETERS (empirically derived from spaced repetition research)
# ============================================================================

# Initial memory stability in days (how long until 50% forgotten without review)
INITIAL_STABILITY = 1.0  # 1 day

# Stability growth factor (how much stability increases per successful review)
# From SuperMemo research: stability roughly doubles with each successful review
STABILITY_GROWTH = 2.0

# Minimum retention threshold to consider "mastered"
MASTERY_THRESHOLD = 0.8

# Decay rate for stability growth (diminishing returns)
STABILITY_DECAY = 0.3


@dataclass
class ConceptMastery:
    """Mastery state for a single concept."""
    concept_id: str
    
    # Number of successful reviews
    review_count: int = 0
    
    # Last review timestamp (ISO format)
    last_review: Optional[str] = None
    
    # Memory stability in days (S in the forgetting curve)
    stability: float = INITIAL_STABILITY
    
    # Quality of last recall (0-5 scale, like SM-2 algorithm)
    last_quality: float = 3.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConceptMastery':
        return cls(**data)


@dataclass
class StudentProfile:
    """Student learning profile with concept mastery tracking."""
    student_id: str
    name: str
    
    # Concept ID -> ConceptMastery
    concept_mastery: Dict[str, ConceptMastery] = field(default_factory=dict)
    
    # Profile creation timestamp
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'student_id': self.student_id,
            'name': self.name,
            'concept_mastery': {k: v.to_dict() for k, v in self.concept_mastery.items()},
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StudentProfile':
        profile = cls(
            student_id=data['student_id'],
            name=data['name'],
            created_at=data.get('created_at', datetime.now().isoformat())
        )
        for cid, cdata in data.get('concept_mastery', {}).items():
            profile.concept_mastery[cid] = ConceptMastery.from_dict(cdata)
        return profile


def ebbinghaus_retention(time_elapsed_days: float, stability: float) -> float:
    """
    Calculate retention using Ebbinghaus forgetting curve.
    
    R(t) = e^(-t/S)
    
    Args:
        time_elapsed_days: Days since last review
        stability: Memory stability in days
        
    Returns:
        Retention probability (0 to 1)
    """
    if time_elapsed_days <= 0:
        return 1.0
    if stability <= 0:
        return 0.0
    
    # Ebbinghaus exponential decay
    retention = math.exp(-time_elapsed_days / stability)
    return max(0.0, min(1.0, retention))


def calculate_new_stability(
    old_stability: float,
    recall_quality: float,
    review_count: int
) -> float:
    """
    Calculate new memory stability after a review.
    
    Based on SuperMemo SM-2 algorithm principles:
    - Successful recall (quality >= 3) increases stability
    - Failed recall (quality < 3) resets or decreases stability
    - Stability growth has diminishing returns
    
    Args:
        old_stability: Previous stability in days
        recall_quality: Quality of recall (0-5 scale)
        review_count: Total number of reviews
        
    Returns:
        New stability in days
    """
    if recall_quality < 3:
        # Failed recall - reset to initial but keep some memory
        return max(INITIAL_STABILITY, old_stability * 0.3)
    
    # Successful recall - increase stability
    # Formula: S_new = S_old * growth_factor * quality_modifier
    # Growth has diminishing returns based on current stability
    
    quality_modifier = 1.0 + (recall_quality - 3) * 0.15  # 0-5 scale, 3 is baseline
    
    # Diminishing returns: harder to increase already-stable memories
    growth = STABILITY_GROWTH * (INITIAL_STABILITY / old_stability) ** STABILITY_DECAY
    growth = max(1.1, min(growth, STABILITY_GROWTH))  # Clamp growth factor
    
    new_stability = old_stability * growth * quality_modifier
    
    # Cap maximum stability at ~6 months
    return min(new_stability, 180.0)


def get_current_retention(mastery: ConceptMastery, current_time: datetime = None) -> float:
    """
    Get current retention level for a concept.
    
    Args:
        mastery: ConceptMastery object
        current_time: Current time (default: now)
        
    Returns:
        Current retention (0 to 1)
    """
    if mastery.last_review is None:
        return 0.0
    
    if current_time is None:
        current_time = datetime.now()
    
    last_review = datetime.fromisoformat(mastery.last_review)
    time_elapsed = (current_time - last_review).total_seconds() / 86400  # Convert to days
    
    return ebbinghaus_retention(time_elapsed, mastery.stability)


class MasteryTracker:
    """
    Track student mastery with forgetting curve simulation.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.profiles_file = self.data_dir / 'student_profiles.json'
        self.profiles: Dict[str, StudentProfile] = {}
        
        self._load_profiles()
    
    def _load_profiles(self):
        """Load existing student profiles."""
        if self.profiles_file.exists():
            with open(self.profiles_file, 'r') as f:
                data = json.load(f)
                for sid, sdata in data.items():
                    self.profiles[sid] = StudentProfile.from_dict(sdata)
    
    def save_profiles(self):
        """Save all student profiles."""
        with open(self.profiles_file, 'w') as f:
            data = {sid: p.to_dict() for sid, p in self.profiles.items()}
            json.dump(data, f, indent=2)
    
    def create_student(self, student_id: str, name: str) -> StudentProfile:
        """Create a new student profile."""
        profile = StudentProfile(student_id=student_id, name=name)
        self.profiles[student_id] = profile
        self.save_profiles()
        return profile
    
    def get_student(self, student_id: str) -> Optional[StudentProfile]:
        """Get a student profile by ID."""
        return self.profiles.get(student_id)
    
    def record_review(
        self,
        student_id: str,
        concept_id: str,
        quality: float = 4.0,
        review_time: datetime = None
    ) -> ConceptMastery:
        """
        Record a concept review/study session.
        
        Args:
            student_id: Student ID
            concept_id: Concept ID
            quality: Quality of recall (0-5 scale, 4 is good)
            review_time: Time of review (default: now)
            
        Returns:
            Updated ConceptMastery
        """
        if review_time is None:
            review_time = datetime.now()
        
        profile = self.profiles.get(student_id)
        if not profile:
            raise ValueError(f"Student {student_id} not found")
        
        # Get or create mastery entry
        if concept_id not in profile.concept_mastery:
            mastery = ConceptMastery(concept_id=concept_id)
            profile.concept_mastery[concept_id] = mastery
        else:
            mastery = profile.concept_mastery[concept_id]
        
        # Calculate new stability
        mastery.stability = calculate_new_stability(
            mastery.stability,
            quality,
            mastery.review_count
        )
        
        # Update mastery state
        mastery.review_count += 1
        mastery.last_review = review_time.isoformat()
        mastery.last_quality = quality
        
        self.save_profiles()
        return mastery
    
    def get_concept_retention(
        self,
        student_id: str,
        concept_id: str,
        current_time: datetime = None
    ) -> float:
        """
        Get current retention level for a concept.
        
        Returns 0.0 if concept has never been studied.
        """
        profile = self.profiles.get(student_id)
        if not profile:
            return 0.0
        
        mastery = profile.concept_mastery.get(concept_id)
        if not mastery:
            return 0.0
        
        return get_current_retention(mastery, current_time)
    
    def get_lecture_completion(
        self,
        student_id: str,
        lecture_id: str,
        concept_ids: List[str],
        current_time: datetime = None
    ) -> float:
        """
        Calculate lecture completion percentage based on concept mastery.
        
        Completion = average retention of all concepts covered by the lecture.
        
        Args:
            student_id: Student ID
            lecture_id: Lecture ID
            concept_ids: List of concept IDs covered by the lecture
            current_time: Current time for retention calculation
            
        Returns:
            Completion percentage (0 to 1)
        """
        if not concept_ids:
            return 0.0
        
        retentions = [
            self.get_concept_retention(student_id, cid, current_time)
            for cid in concept_ids
        ]
        
        return sum(retentions) / len(retentions)
    
    def get_all_masteries(
        self,
        student_id: str,
        current_time: datetime = None
    ) -> Dict[str, float]:
        """
        Get current retention for all concepts a student has studied.
        
        Returns:
            Dict of concept_id -> retention (0 to 1)
        """
        profile = self.profiles.get(student_id)
        if not profile:
            return {}
        
        return {
            cid: get_current_retention(mastery, current_time)
            for cid, mastery in profile.concept_mastery.items()
        }
    
    def simulate_study_session(
        self,
        student_id: str,
        concept_ids: List[str],
        quality: float = 4.0,
        study_time: datetime = None
    ) -> Dict[str, float]:
        """
        Simulate a study session where student reviews multiple concepts.
        
        Returns dict of concept_id -> new retention after study.
        """
        results = {}
        for cid in concept_ids:
            mastery = self.record_review(student_id, cid, quality, study_time)
            results[cid] = get_current_retention(mastery, study_time)
        return results


def generate_sample_student_data(
    data_dir: Path,
    nodes_file: Path,
    edges_file: Path,
    student_id: str = "demo_student",
    student_name: str = "Demo Student",
    num_lectures: int = 2,
    days_back: int = 30
) -> StudentProfile:
    """
    Generate sample student mastery data for visualization demo.
    
    Creates mastery data for exactly num_lectures lectures, with concepts
    deliberately spread across the full retention spectrum for good color variety.
    """
    import random
    
    tracker = MasteryTracker(data_dir)
    
    # Create student - always recreate for fresh data
    profile = tracker.create_student(student_id, student_name)
    
    # Load graph data
    nodes = {}
    with open(nodes_file, 'r') as f:
        for line in f:
            if line.strip():
                node = json.loads(line)
                nodes[node['id']] = node
    
    edges = []
    with open(edges_file, 'r') as f:
        for line in f:
            if line.strip():
                edges.append(json.loads(line))
    
    # Build lecture -> concepts mapping
    lecture_concepts = {}
    for edge in edges:
        if edge['type'] == 'COVERS_CONCEPT':
            lecture_id = edge['source']
            if lecture_id not in lecture_concepts:
                lecture_concepts[lecture_id] = []
            lecture_concepts[lecture_id].append(edge['target'])
    
    # Find lectures with enough concepts for good visualization
    lectures_with_concepts = [
        (lid, concepts) for lid, concepts in lecture_concepts.items()
        if len(concepts) >= 5  # Need at least 5 concepts for variety
    ]
    
    if len(lectures_with_concepts) < num_lectures:
        num_lectures = len(lectures_with_concepts)
    
    random.seed(42)  # Reproducible
    selected = random.sample(lectures_with_concepts, num_lectures)
    
    now = datetime.now()
    
    print(f"Selected {num_lectures} lectures for student:")
    
    for lec_idx, (lecture_id, concept_ids) in enumerate(selected):
        lecture_name = nodes[lecture_id]['name'] if lecture_id in nodes else lecture_id
        print(f"  Lecture {lec_idx+1}: {lecture_name[:60]}...")
        print(f"    Concepts: {len(concept_ids)}")
        
        # Create varied mastery states:
        # - 25% low retention (0.1-0.35) - red colors
        # - 25% medium-low retention (0.4-0.55) - orange/yellow colors  
        # - 25% medium-high retention (0.6-0.75) - yellow/light green
        # - 25% high retention (0.8-0.99) - green colors
        n = len(concept_ids)
        low_count = n // 4
        mid_low_count = n // 4
        mid_high_count = n // 4
        high_count = n - low_count - mid_low_count - mid_high_count
        
        shuffled = list(concept_ids)
        random.shuffle(shuffled)
        
        # Assign target retentions to each concept
        concept_targets = []
        for i, concept_id in enumerate(shuffled):
            if i < low_count:
                # Low retention: studied long ago, few reviews
                target_retention = random.uniform(0.05, 0.35)
            elif i < low_count + mid_low_count:
                # Medium-low
                target_retention = random.uniform(0.40, 0.55)
            elif i < low_count + mid_low_count + mid_high_count:
                # Medium-high  
                target_retention = random.uniform(0.60, 0.75)
            else:
                # High retention
                target_retention = random.uniform(0.80, 0.99)
            
            concept_targets.append((concept_id, target_retention))
        
        # Generate review history to achieve target retentions
        for concept_id, target in concept_targets:
            # Work backwards: given target retention, determine study time
            # R = e^(-t/S), so t = -S * ln(R)
            # For new concepts, S starts at ~1.0
            # After good reviews, S grows to ~5-10
            
            if target >= 0.8:
                # High retention: recent study with good stability
                # S ≈ 5, need t ≈ 0.5-2 days for R ≈ 0.85-0.95
                stability = random.uniform(4, 8)
                days_ago = -stability * math.log(target)
                num_reviews = random.randint(3, 5)
            elif target >= 0.6:
                # Medium-high: moderate study
                stability = random.uniform(3, 5)
                days_ago = -stability * math.log(target)
                num_reviews = random.randint(2, 4)
            elif target >= 0.4:
                # Medium-low: starting to forget
                stability = random.uniform(2, 4)
                days_ago = -stability * math.log(target)
                num_reviews = random.randint(1, 3)
            else:
                # Low: studied long ago, mostly forgotten
                stability = random.uniform(1, 2)
                days_ago = -stability * math.log(max(target, 0.01))
                num_reviews = 1
            
            # Clamp days_ago to reasonable range
            days_ago = max(0.1, min(days_ago, days_back))
            
            # Generate review sequence
            for review_num in range(num_reviews):
                if num_reviews > 1:
                    # Reviews spread out, most recent closest to now
                    review_fraction = review_num / (num_reviews - 1)
                    review_days_ago = days_ago * (1 - review_fraction * 0.8)
                else:
                    review_days_ago = days_ago
                
                review_time = now - timedelta(days=review_days_ago)
                
                # Quality based on target retention
                if target >= 0.7:
                    quality = random.uniform(4.0, 5.0)
                elif target >= 0.4:
                    quality = random.uniform(3.0, 4.0)
                else:
                    quality = random.uniform(2.0, 3.0)
                
                tracker.record_review(student_id, concept_id, quality, review_time)
        
        # Print retention distribution for this lecture
        retentions = tracker.get_all_masteries(student_id, now)
        lecture_retentions = [retentions.get(cid, 0) for cid in concept_ids if cid in retentions]
        if lecture_retentions:
            avg_ret = sum(lecture_retentions) / len(lecture_retentions)
            min_ret = min(lecture_retentions)
            max_ret = max(lecture_retentions)
            print(f"    Retention range: {min_ret:.2f} - {max_ret:.2f} (avg: {avg_ret:.2f})")
    
    # Final stats
    all_retentions = list(tracker.get_all_masteries(student_id, now).values())
    print(f"\nTotal concepts: {len(all_retentions)}")
    print(f"Retention distribution:")
    buckets = {'0.0-0.4 (red)': 0, '0.4-0.7 (yellow)': 0, '0.7-1.0 (green)': 0}
    for r in all_retentions:
        if r < 0.4:
            buckets['0.0-0.4 (red)'] += 1
        elif r < 0.7:
            buckets['0.4-0.7 (yellow)'] += 1
        else:
            buckets['0.7-1.0 (green)'] += 1
    for k, v in buckets.items():
        print(f"  {k}: {v} ({100*v/len(all_retentions):.0f}%)")
    
    return tracker.get_student(student_id)


def get_studied_subgraph(
    data_dir: Path,
    student_id: str
) -> Tuple[set, set]:
    """
    Get the set of node IDs and lecture IDs that have mastery data.
    
    Returns:
        (studied_concept_ids, studied_lecture_ids)
    """
    tracker = MasteryTracker(data_dir)
    profile = tracker.get_student(student_id)
    
    if not profile:
        return set(), set()
    
    studied_concepts = set(profile.concept_mastery.keys())
    
    # Find lectures that cover these concepts
    edges_file = data_dir / 'edges.jsonl'
    studied_lectures = set()
    
    with open(edges_file, 'r') as f:
        for line in f:
            if line.strip():
                edge = json.loads(line)
                if edge['type'] == 'COVERS_CONCEPT':
                    if edge['target'] in studied_concepts:
                        studied_lectures.add(edge['source'])
    
    return studied_concepts, studied_lectures


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample student mastery data")
    parser.add_argument('--data-dir', type=Path, default=Path('data/output'))
    parser.add_argument('--student-id', default='demo_student')
    parser.add_argument('--student-name', default='Demo Student')
    parser.add_argument('--study-probability', type=float, default=0.3)
    parser.add_argument('--days-back', type=int, default=30)
    
    args = parser.parse_args()
    
    nodes_file = args.data_dir / 'nodes.jsonl'
    edges_file = args.data_dir / 'edges.jsonl'
    
    generate_sample_student_data(
        args.data_dir,
        nodes_file,
        edges_file,
        args.student_id,
        args.student_name,
        args.study_probability,
        args.days_back
    )
