"""Presentation-only fighter portraits, isolated from prediction features."""

from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from threading import Lock
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .fighter_matching import canonical_fighter_name

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'data' / 'fighter_portraits'
_locks = {}
_guard = Lock()


class PortraitParser(HTMLParser):
    def __init__(self, name):
        super().__init__()
        self.name = canonical_fighter_name(name)
        self.image = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if (tag == 'img' and 'hero-profile__image' in values.get('class', '')
                and canonical_fighter_name(values.get('alt', '')) == self.name):
            self.image = values.get('src')


def _fetch(url):
    request = Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': '*/*'})
    with urlopen(request, timeout=12) as response:
        return response.read(8_000_000)


def portrait(name, espn_id=None):
    slug = canonical_fighter_name(name).replace(' ', '-')
    if not slug or len(slug) > 100:
        return None
    with _guard:
        lock = _locks.setdefault(slug, Lock())
    with lock:
        CACHE.mkdir(parents=True, exist_ok=True)
        target = CACHE / f'{slug}.png'
        metadata = CACHE / f'{slug}.json'
        if target.exists():
            return target
        if metadata.exists() and time.time() - metadata.stat().st_mtime < 86400:
            return None
        profile = f'https://www.ufc.com/athlete/{slug}'
        candidates = []
        try:
            parser = PortraitParser(name)
            parser.feed(_fetch(profile).decode('utf-8'))
            if parser.image:
                url = urlparse(parser.image)
                if url.scheme == 'https' and url.hostname in {'ufc.com', 'www.ufc.com'}:
                    candidates.append((parser.image, profile, 'UFC'))
        except (OSError, ValueError):
            pass
        if espn_id and re.fullmatch(r'\d{1,12}', str(espn_id)):
            candidates.append((f'https://a.espncdn.com/i/headshots/mma/players/full/{espn_id}.png',
                               f'https://www.espn.com/mma/fighter/_/id/{espn_id}', 'ESPN'))
        for image_url, source_url, source in candidates:
            try:
                content = _fetch(image_url)
                if not content.startswith(b'\x89PNG\r\n\x1a\n'):
                    continue
                temporary = target.with_suffix('.tmp')
                temporary.write_bytes(content)
                temporary.replace(target)
                metadata.write_text(json.dumps(dict(name=name, source=source, source_url=source_url,
                    image_url=image_url, fetched_at=datetime.now(timezone.utc).isoformat()), indent=2), encoding='utf-8')
                return target
            except (OSError, ValueError):
                continue
        metadata.write_text(json.dumps(dict(name=name, unavailable=True)), encoding='utf-8')
        return None
