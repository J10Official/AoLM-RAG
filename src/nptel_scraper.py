"""
NPTEL scraper module using Playwright for dynamic content.
"""
import asyncio
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from playwright.async_api import async_playwright, Page, Browser, TimeoutError as PlaywrightTimeout


@dataclass
class Lecture:
    """Represents a lecture in a course."""
    name: str
    week: int
    lecture_num: int
    youtube_url: Optional[str] = None
    
    
@dataclass
class CourseDetails:
    """Represents detailed course information."""
    abstract: str = ""
    professor: str = ""
    institute: str = ""
    lectures: List[Lecture] = field(default_factory=list)
    error: Optional[str] = None


class NPTELScraper:
    """Scraper for NPTEL course pages using Playwright."""
    
    def __init__(self, headless: bool = True, slow_mo: int = 100):
        self.headless = headless
        self.slow_mo = slow_mo
        self._browser: Optional[Browser] = None
        self._playwright = None
        
    async def __aenter__(self):
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def start(self):
        """Start the browser."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo
        )
        
    async def close(self):
        """Close the browser."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
            
    async def get_course_details(self, nptel_url: str) -> CourseDetails:
        """
        Scrape course details including abstract and lectures.
        
        Args:
            nptel_url: URL to the NPTEL course page
            
        Returns:
            CourseDetails object with abstract and lectures
        """
        details = CourseDetails()
        
        if not self._browser:
            details.error = "Browser not started"
            return details
            
        page = await self._browser.new_page()
        
        try:
            # Navigate to the course page
            print(f"    Navigating to {nptel_url}", flush=True)
            await page.goto(nptel_url, wait_until='networkidle', timeout=45000)
            await asyncio.sleep(1)  # Wait for dynamic content
            
            # Get course abstract from About section
            details.abstract = await self._get_course_abstract(page)
            
            # Get lectures from week tabs
            details.lectures = await self._get_lectures(page)
            
        except PlaywrightTimeout:
            details.error = "Timeout loading page"
        except Exception as e:
            details.error = f"Error: {str(e)}"
        finally:
            await page.close()
            
        return details
    
    async def _get_course_abstract(self, page: Page) -> str:
        """Extract course abstract from the About section."""
        abstract = ""
        
        try:
            # Try clicking on "About" tab if it exists
            about_selectors = [
                'text=About',
                'a:has-text("About")',
                'button:has-text("About")',
                '[data-tab="about"]',
            ]
            
            for selector in about_selectors:
                try:
                    about_btn = page.locator(selector).first
                    if await about_btn.is_visible(timeout=2000):
                        await about_btn.click()
                        await asyncio.sleep(0.5)
                        break
                except:
                    continue
            
            # Try to find abstract content in various possible containers
            content_selectors = [
                '.course-about',
                '.about-content', 
                '.course-abstract',
                '#about',
                '[class*="about"]',
                '.course-description',
                'p',  # fallback to any paragraph
            ]
            
            for selector in content_selectors:
                try:
                    elements = page.locator(selector)
                    count = await elements.count()
                    if count > 0:
                        texts = []
                        for i in range(min(count, 5)):  # Get up to 5 paragraphs
                            text = await elements.nth(i).inner_text()
                            if text and len(text) > 50:  # Skip short texts
                                texts.append(text.strip())
                        if texts:
                            abstract = ' '.join(texts[:3])  # Join first 3 meaningful paragraphs
                            if len(abstract) > 100:
                                break
                except:
                    continue
                    
        except Exception as e:
            print(f"    Warning: Could not get abstract: {e}")
            
        return abstract[:2000] if abstract else ""  # Limit length
    
    async def _get_lectures(self, page: Page) -> List[Lecture]:
        """Extract lecture list directly from page HTML (no clicking needed)."""
        lectures = []
        
        try:
            # Get HTML content - lectures are embedded in JavaScript data
            html = await page.content()
            seen_lectures = {}  # youtube_id -> Lecture (deduplicate by youtube_id)
            
            # Primary method: Find all lesson objects with youtube_id
            # Find all lessons:[ ... ] blocks
            lessons_blocks = re.findall(r'lessons:\[([^\]]+)\]', html)
            
            lecture_num = 0
            for block in lessons_blocks:
                # Within each block, find individual lessons
                lesson_pattern = r'\{id:(\d+),name:"([^"]+)"[^}]*?youtube_id:"([a-zA-Z0-9_-]*)"'
                lesson_matches = re.findall(lesson_pattern, block)
                
                for lec_id, lec_name, youtube_id in lesson_matches:
                    if youtube_id and youtube_id not in seen_lectures:
                        lecture_num += 1
                        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
                        
                        # Try to extract week from name
                        week_match = re.search(r'[Ww]eek[\s_-]*(\d+)', lec_name)
                        week_num = int(week_match.group(1)) if week_match else ((lecture_num - 1) // 7) + 1
                        
                        lecture = Lecture(
                            name=lec_name.strip(),
                            week=week_num,
                            lecture_num=lecture_num,
                            youtube_url=youtube_url
                        )
                        seen_lectures[youtube_id] = lecture
            
            # Fallback: scan entire HTML for any youtube_id with associated name
            if not seen_lectures:
                # Look for patterns like name:"...",youtube_id:"..."
                all_pattern = r'name:"([^"]{3,100})"[^}]*?youtube_id:"([a-zA-Z0-9_-]{11})"'
                all_matches = re.findall(all_pattern, html)
                
                for lec_name, youtube_id in all_matches:
                    if youtube_id not in seen_lectures:
                        lecture_num += 1
                        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
                        week_match = re.search(r'[Ww]eek[\s_-]*(\d+)', lec_name)
                        week_num = int(week_match.group(1)) if week_match else ((lecture_num - 1) // 7) + 1
                        
                        lecture = Lecture(
                            name=lec_name.strip(),
                            week=week_num,
                            lecture_num=lecture_num,
                            youtube_url=youtube_url
                        )
                        seen_lectures[youtube_id] = lecture
            
            # Convert to sorted list
            lectures = sorted(seen_lectures.values(), key=lambda x: x.lecture_num)
            print(f"      Extracted {len(lectures)} unique lectures", flush=True)
                    
        except Exception as e:
            print(f"    Warning: Could not get lectures: {e}", flush=True)
            
        return lectures
    
    async def _get_lectures_direct(self, page: Page) -> List[Lecture]:
        """Alternative method: get lectures directly from the page."""
        lectures = []
        
        try:
            # Look for any links containing lecture-related text
            all_links = page.locator('a')
            count = await all_links.count()
            
            seen_names = set()
            for i in range(count):
                try:
                    link = all_links.nth(i)
                    text = await link.inner_text()
                    href = await link.get_attribute('href') or ""
                    
                    # Check if this looks like a lecture link
                    if any(kw in text.lower() for kw in ['lecture', 'lec ', 'module', 'video']):
                        name = text.strip()
                        if name and name not in seen_names and len(name) > 3:
                            seen_names.add(name)
                            
                            # Try to extract week number from text or href
                            week_match = re.search(r'week[_\s]*(\d+)', text.lower() + href.lower())
                            week_num = int(week_match.group(1)) if week_match else 1
                            
                            # Try to extract lecture number
                            lec_match = re.search(r'(?:lecture|lec)[_\s]*(\d+)', text.lower())
                            lec_num = int(lec_match.group(1)) if lec_match else len(lectures) + 1
                            
                            # Extract YouTube URL if embedded
                            youtube_url = None
                            if 'youtube.com' in href or 'youtu.be' in href:
                                youtube_url = href
                            
                            lecture = Lecture(
                                name=name,
                                week=week_num,
                                lecture_num=lec_num,
                                youtube_url=youtube_url
                            )
                            lectures.append(lecture)
                except:
                    continue
                    
        except Exception as e:
            print(f"    Warning: Direct lecture extraction failed: {e}")
            
        return lectures
    
    def _extract_week_number(self, text: str, default: int) -> int:
        """Extract week number from text."""
        match = re.search(r'week[_\s]*(\d+)', text.lower())
        return int(match.group(1)) if match else default


async def scrape_course(nptel_url: str, headless: bool = True) -> CourseDetails:
    """
    Convenience function to scrape a single course.
    
    Args:
        nptel_url: URL to the NPTEL course page
        headless: Whether to run browser in headless mode
        
    Returns:
        CourseDetails object
    """
    async with NPTELScraper(headless=headless) as scraper:
        return await scraper.get_course_details(nptel_url)


if __name__ == '__main__':
    # Test the scraper
    import sys
    
    test_url = "https://nptel.ac.in/courses/101101805"
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    
    print(f"Testing scraper with URL: {test_url}")
    print("-" * 50)
    
    async def main():
        details = await scrape_course(test_url, headless=True)
        
        print(f"\nAbstract: {details.abstract[:500]}..." if details.abstract else "No abstract found")
        print(f"\nLectures found: {len(details.lectures)}")
        
        if details.lectures:
            print("\nFirst 5 lectures:")
            for lec in details.lectures[:5]:
                print(f"  Week {lec.week}, Lec {lec.lecture_num}: {lec.name}")
                
        if details.error:
            print(f"\nError: {details.error}")
    
    asyncio.run(main())
