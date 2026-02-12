"""
Async concept extractor for parallel processing.
Uses aiohttp for concurrent API calls to OpenRouter.
"""
import asyncio
import aiohttp
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from tqdm import tqdm

from src.concept_extractor import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, Concept


# ============================================================================
# CONFIGURATION
# ============================================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models to rotate through
MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1-0528:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
]


@dataclass
class ExtractionResult:
    """Result of a single lecture extraction."""
    lecture_id: str
    lecture_name: str
    concepts: List[Concept]
    error: Optional[str] = None
    model_used: str = ""


# Backoff delays in seconds: 5s, 10s, 20s, 30s, 60s, 120s, 120s
BACKOFF_DELAYS = [5, 10, 20, 30, 60, 120, 120]


class AsyncModelRotator:
    """
    Thread-safe model rotator with round-robin distribution and cooldown tracking.
    
    - Distributes requests across models from the start (not sequential exhaustion)
    - Tracks rate-limited models with cooldown timestamps
    - Skips models that are in cooldown period
    """
    
    def __init__(self, models: List[str], requests_per_model: int = 900):
        self.models = models
        self.requests_per_model = requests_per_model
        self.request_counts = [0] * len(models)
        self.current_index = 0
        self._lock = asyncio.Lock()
        # Cooldown tracking: model_index -> timestamp when cooldown expires
        self.cooldowns: Dict[int, float] = {}
    
    async def get_model_for_request(self, exclude_models: List[str] = None) -> Tuple[str, int, bool]:
        """
        Get next available model using round-robin, skipping rate-limited ones.
        
        Args:
            exclude_models: List of model names to skip (e.g., already tried this attempt)
            
        Returns:
            (model_name, model_index, success). If no models available, success=False.
        """
        exclude_models = exclude_models or []
        now = asyncio.get_event_loop().time()
        
        async with self._lock:
            # Try each model starting from current index
            for _ in range(len(self.models)):
                idx = self.current_index
                model = self.models[idx]
                
                # Advance index for next call (round-robin distribution)
                self.current_index = (self.current_index + 1) % len(self.models)
                
                # Skip if explicitly excluded
                if model in exclude_models:
                    continue
                
                # Skip if in cooldown (rate limited recently)
                if idx in self.cooldowns and now < self.cooldowns[idx]:
                    continue
                
                # Skip if quota exhausted
                if self.request_counts[idx] >= self.requests_per_model:
                    continue
                
                # Found available model
                self.request_counts[idx] += 1
                return model, idx, True
            
            return "", -1, False
    
    async def mark_rate_limited(self, model_index: int, cooldown_seconds: int):
        """Mark a model as rate-limited with a cooldown period."""
        now = asyncio.get_event_loop().time()
        async with self._lock:
            self.cooldowns[model_index] = now + cooldown_seconds
    
    async def clear_cooldown(self, model_index: int):
        """Clear cooldown for a model (successful request)."""
        async with self._lock:
            if model_index in self.cooldowns:
                del self.cooldowns[model_index]
    
    def get_remaining(self) -> int:
        """Get total remaining requests across all models."""
        return sum(self.requests_per_model - c for c in self.request_counts)
    
    def get_status(self) -> str:
        """Get status string showing model usage and cooldown status."""
        parts = []
        now = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        for i, (model, count) in enumerate(zip(self.models, self.request_counts)):
            name = model.split("/")[1].split(":")[0][:12]
            marker = "→" if i == self.current_index else " "
            cooldown_str = ""
            if i in self.cooldowns:
                remaining = max(0, self.cooldowns[i] - now)
                if remaining > 0:
                    cooldown_str = f" [CD:{int(remaining)}s]"
            parts.append(f"{marker}{name}: {count}/{self.requests_per_model}{cooldown_str}")
        return " | ".join(parts)


class AsyncConceptExtractor:
    """
    Async concept extractor for parallel processing.
    """
    
    def __init__(
        self,
        api_key: str,
        concurrency_limit: int = 20,
        timeout: int = 120,
        max_retries: int = 2
    ):
        """
        Initialize the async extractor.
        
        Args:
            api_key: OpenRouter API key
            concurrency_limit: Max concurrent requests
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
        """
        self.api_key = api_key
        self.concurrency_limit = concurrency_limit
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.rotator = AsyncModelRotator(MODELS)
        
        # Stats
        self.successful = 0
        self.failed = 0
        self.rate_limited = 0
    
    def _build_request_body(
        self,
        transcript: str,
        lecture_name: str,
        course_name: str,
        discipline: str,
        model: str
    ) -> Dict[str, Any]:
        """Build the API request body."""
        prompt = USER_PROMPT_TEMPLATE.format(
            lecture_name=lecture_name,
            course_name=course_name,
            discipline=discipline,
            transcript=transcript[:80000]  # Limit transcript length
        )
        
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4096,
            "temperature": 0.3
        }
        
        # Add provider preferences for llama model
        if "llama" in model.lower():
            body["provider"] = {"order": ["open-inference/int8"]}
        
        return body
    
    def _parse_response(self, response_text: str, lecture_id: str) -> List[Concept]:
        """Parse LLM response into Concept objects."""
        # Clean response - remove markdown code blocks if present
        response = response_text.strip()
        if response.startswith("```"):
            response = re.sub(r'^```(?:json)?\n?', '', response)
            response = re.sub(r'\n?```$', '', response)
        
        # Find JSON array in response
        match = re.search(r'\[[\s\S]*\]', response)
        if not match:
            return []
        
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        
        concepts = []
        for item in data:
            if isinstance(item, dict) and 'name' in item:
                concept = Concept.from_dict(item, source_lecture_id=lecture_id)
                if concept.name and concept.description:
                    concepts.append(concept)
        
        return concepts
    
    async def _extract_single(
        self,
        session: aiohttp.ClientSession,
        lecture: Dict[str, str],
        transcript: str,
        pbar: Any
    ) -> ExtractionResult:
        """
        Extract concepts from a single lecture with smart rate limiting.
        
        - Uses round-robin model distribution
        - On 429: switches to different model immediately
        - Backoff delays: 5s, 10s, 20s, 30s, 60s, 120s, 120s
        - Only fails if all retries exhausted with no available models
        """
        lecture_id = lecture['lecture_id']
        lecture_name = lecture['lecture_name']
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/nptel-graph-rag",
            "X-Title": "NPTEL Graph RAG"
        }
        
        tried_models = []  # Models tried in current backoff cycle
        last_model = ""
        last_model_idx = -1
        
        async with self.semaphore:
            for attempt, delay in enumerate(BACKOFF_DELAYS):
                # Get an available model (skip ones we've tried this cycle)
                model, model_idx, available = await self.rotator.get_model_for_request(
                    exclude_models=tried_models if attempt > 0 else []
                )
                
                if not available:
                    # All models exhausted or in cooldown - wait and reset tried list
                    if attempt < len(BACKOFF_DELAYS) - 1:
                        pbar.write(f"  [{lecture_id[:20]}] All models busy, waiting {delay}s...")
                        await asyncio.sleep(delay)
                        tried_models = []  # Reset for next attempt
                        continue
                    else:
                        self.failed += 1
                        return ExtractionResult(
                            lecture_id=lecture_id,
                            lecture_name=lecture_name,
                            concepts=[],
                            error="All models exhausted after all retries"
                        )
                
                last_model = model
                last_model_idx = model_idx
                
                body = self._build_request_body(
                    transcript=transcript,
                    lecture_name=lecture_name,
                    course_name=lecture.get('course_name', ''),
                    discipline=lecture.get('discipline', ''),
                    model=model
                )
                
                try:
                    async with session.post(
                        OPENROUTER_BASE_URL,
                        headers=headers,
                        json=body,
                        timeout=self.timeout
                    ) as response:
                        if response.status == 429:
                            # Rate limited - mark model and switch
                            self.rate_limited += 1
                            retry_after = int(response.headers.get('Retry-After', delay))
                            model_short = model.split("/")[1].split(":")[0][:12]
                            pbar.write(f"  [{lecture_id[:20]}] {model_short} rate limited, cooldown {retry_after}s, switching model...")
                            
                            # Mark this model as rate-limited
                            await self.rotator.mark_rate_limited(model_idx, retry_after)
                            tried_models.append(model)
                            
                            # Don't sleep full backoff - just try next model immediately
                            # (but if all models tried this cycle, we'll hit the !available case above)
                            continue
                        
                        if response.status != 200:
                            error_text = await response.text()
                            # Non-429 error - add to tried and backoff
                            tried_models.append(model)
                            if attempt < len(BACKOFF_DELAYS) - 1:
                                pbar.write(f"  [{lecture_id[:20]}] HTTP {response.status}, retrying in {delay}s...")
                                await asyncio.sleep(delay)
                                continue
                            self.failed += 1
                            return ExtractionResult(
                                lecture_id=lecture_id,
                                lecture_name=lecture_name,
                                concepts=[],
                                error=f"HTTP {response.status}: {error_text[:100]}",
                                model_used=model
                            )
                        
                        # Success! Clear any cooldown for this model
                        await self.rotator.clear_cooldown(model_idx)
                        
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        concepts = self._parse_response(content, lecture_id)
                        
                        self.successful += 1
                        pbar.update(1)
                        
                        return ExtractionResult(
                            lecture_id=lecture_id,
                            lecture_name=lecture_name,
                            concepts=concepts,
                            model_used=model
                        )
                        
                except asyncio.TimeoutError:
                    tried_models.append(model)
                    if attempt < len(BACKOFF_DELAYS) - 1:
                        pbar.write(f"  [{lecture_id[:20]}] Timeout, retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    self.failed += 1
                    return ExtractionResult(
                        lecture_id=lecture_id,
                        lecture_name=lecture_name,
                        concepts=[],
                        error="Timeout after all retries",
                        model_used=last_model
                    )
                except Exception as e:
                    tried_models.append(model)
                    if attempt < len(BACKOFF_DELAYS) - 1:
                        pbar.write(f"  [{lecture_id[:20]}] Error: {str(e)[:50]}, retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    self.failed += 1
                    return ExtractionResult(
                        lecture_id=lecture_id,
                        lecture_name=lecture_name,
                        concepts=[],
                        error=str(e)[:100],
                        model_used=last_model
                    )
            
            # Should not reach here, but safety fallback
            self.failed += 1
            return ExtractionResult(
                lecture_id=lecture_id,
                lecture_name=lecture_name,
                concepts=[],
                error="Max retries exceeded",
                model_used=last_model
            )
    
    async def extract_batch(
        self,
        lectures: List[Dict[str, str]],
        get_transcript_fn,
        progress_callback=None
    ) -> List[ExtractionResult]:
        """
        Extract concepts from a batch of lectures in parallel.
        
        Args:
            lectures: List of lecture dicts with lecture_id, lecture_name, youtube_url, etc.
            get_transcript_fn: Function to get transcript for a lecture
            progress_callback: Optional callback(results_so_far, total)
            
        Returns:
            List of ExtractionResult objects
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        
        # Phase 1: Fetch all transcripts in parallel using threads
        print(f"    Fetching {len(lectures)} transcripts in parallel...")
        transcripts = {}  # lecture_id -> (transcript_text, error)
        
        def fetch_one(lecture):
            lecture_id = lecture['lecture_id']
            try:
                transcript, error = get_transcript_fn(lecture['youtube_url'])
                if error:
                    return lecture_id, None, error
                return lecture_id, transcript.full_text, None
            except Exception as e:
                return lecture_id, None, str(e)
        
        # Use ThreadPoolExecutor for parallel transcript fetching
        with ThreadPoolExecutor(max_workers=min(50, len(lectures))) as executor:
            futures = {executor.submit(fetch_one, lec): lec for lec in lectures}
            
            transcript_pbar = tqdm(total=len(lectures), desc="Fetching transcripts", unit="video")
            for future in as_completed(futures):
                lecture_id, text, error = future.result()
                transcripts[lecture_id] = (text, error)
                transcript_pbar.update(1)
            transcript_pbar.close()
        
        # Count successful transcripts
        successful_transcripts = sum(1 for t, e in transcripts.values() if t is not None)
        print(f"    Transcripts fetched: {successful_transcripts} successful, {len(lectures) - successful_transcripts} failed")
        
        # Phase 2: Extract concepts via API in parallel
        print(f"    Extracting concepts with {self.concurrency_limit} parallel API calls...")
        
        # Create progress bar for extraction
        pbar = tqdm(total=successful_transcripts, desc="Extracting concepts", unit="lecture")
        
        # Create connector with connection limit
        connector = aiohttp.TCPConnector(limit=self.concurrency_limit + 10)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            
            for lecture in lectures:
                lecture_id = lecture['lecture_id']
                transcript_text, error = transcripts.get(lecture_id, (None, "Not fetched"))
                
                if error or transcript_text is None:
                    results.append(ExtractionResult(
                        lecture_id=lecture_id,
                        lecture_name=lecture['lecture_name'],
                        concepts=[],
                        error=f"Transcript: {error}"
                    ))
                    continue
                
                # Create async task for extraction
                task = asyncio.create_task(
                    self._extract_single(session, lecture, transcript_text, pbar)
                )
                tasks.append(task)
            
            # Wait for all tasks
            if tasks:
                completed_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in completed_results:
                    if isinstance(result, Exception):
                        results.append(ExtractionResult(
                            lecture_id="unknown",
                            lecture_name="unknown",
                            concepts=[],
                            error=str(result)
                        ))
                    else:
                        results.append(result)
        
        pbar.close()
        
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """Get extraction statistics."""
        return {
            "successful": self.successful,
            "failed": self.failed,
            "rate_limited": self.rate_limited,
            "remaining_capacity": self.rotator.get_remaining()
        }


async def extract_concepts_parallel(
    api_key: str,
    lectures: List[Dict[str, str]],
    get_transcript_fn,
    concurrency_limit: int = 20,
    save_callback=None,
    save_interval: int = 50
) -> Tuple[List[ExtractionResult], Dict[str, int]]:
    """
    High-level function to extract concepts in parallel.
    
    Args:
        api_key: OpenRouter API key
        lectures: List of lecture dicts
        get_transcript_fn: Function to get transcript
        concurrency_limit: Max concurrent API calls
        save_callback: Optional callback to save intermediate results
        save_interval: How often to call save_callback
        
    Returns:
        Tuple of (results, stats)
    """
    extractor = AsyncConceptExtractor(
        api_key=api_key,
        concurrency_limit=concurrency_limit
    )
    
    print(f"\n  Parallel extraction with concurrency={concurrency_limit}")
    print(f"  Model capacity: {extractor.rotator.get_remaining()} requests")
    
    results = await extractor.extract_batch(
        lectures=lectures,
        get_transcript_fn=get_transcript_fn
    )
    
    stats = extractor.get_stats()
    print(f"\n  Extraction complete:")
    print(f"    Successful: {stats['successful']}")
    print(f"    Failed: {stats['failed']}")
    print(f"    Rate limited events: {stats['rate_limited']}")
    
    return results, stats
