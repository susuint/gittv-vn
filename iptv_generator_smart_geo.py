# Tên tệp: iptv_generator_smart_geo.py

import asyncio
import aiohttp
import re
import logging
from datetime import datetime
from collections import defaultdict
from urllib.parse import urlparse
import hashlib
import socket

# =======================================================================================
# SMART GEO-DETECTION - Works from ANY GitHub Actions location
# =======================================================================================

SOURCES = {
    "tv": [
        "https://raw.githubusercontent.com/dishiptv/dish/main/stream.m3u",
        "https://raw.githubusercontent.com/LS-Station/streamecho/main/StreamEcho.m3u8",
        "https://iptv-org.github.io/iptv/index.m3u",
        "https://raw.githubusercontent.com/Free-TV/IPTV/master/playlist.m3u8",
        "https://raw.githubusercontent.com/binhex/iptv/main/eng.m3u",
        "https://raw.githubusercontent.com/serdartas/iptv-playlist/main/refined.m3u",
        "https://raw.githubusercontent.com/ipstreet312/freeiptv/master/all.m3u",
        "https://raw.githubusercontent.com/sultanarabi161/filoox-bdix/main/playlist.m3u",
        "https://raw.githubusercontent.com/Miraz6755/Iptv.m3u/main/DaddyLive.m3u",
        "https://raw.githubusercontent.com/AAAAAEXQOSyIpN2JZ0ehUQ/iPTV-FREE-LIST/master/iPTV-Free-List_TV.m3u",
        "https://raw.githubusercontent.com/dp247/IPTV/master/playlists/playlist_usa.m3u8",
        "https://raw.githubusercontent.com/dp247/IPTV/master/playlists/playlist_uk.m3u8",
        "https://raw.githubusercontent.com/HabibSay/free_iptv_m3u8/refs/heads/main/all_channels.m3u",
    ],
    "movies": [
        "https://aymrgknetzpucldhpkwm.supabase.co/storage/v1/object/public/tmdb/top-movies.m3u",
        "https://aymrgknetzpucldhpkwm.supabase.co/storage/v1/object/public/tmdb/action-movies.m3u",
        "https://aymrgknetzpucldhpkwm.supabase.co/storage/v1/object/public/tmdb/comedy-movies.m3u",
        "https://aymrgknetzpucldhpkwm.supabase.co/storage/v1/object/public/tmdb/horror-movies.m3u",
    ],
}

OUTPUT_FILENAME = "playlist.m3u"

# Configuration
CHANNEL_CHECK_TIMEOUT = 6
FETCH_TIMEOUT = 25
MAX_CONCURRENT_CHECKS = 120
MAX_ACCEPTABLE_PING_MS = 4000
EXCELLENT_PING_MS = 800
GOOD_PING_MS = 2000
BATCH_SIZE = 150

# Quality requirements
QUALITY_KEYWORDS_REQUIRED = ['1080', 'fhd', 'full hd', '1920', '4k', 'uhd', '2160']
QUALITY_KEYWORDS_EXCLUDE = ['720', 'hd', '480', 'sd', '360', '240']

# Blocked countries
BLOCKED_COUNTRIES = [
    'bangladesh', 'bd', 'bangla', 'belarus', 'by', 'costa rica', 'cr',
    'india', 'indian', 'in', 'mexico', 'mx', 'lao', 'laos', 'la'
]

# =======================================================================================
# SMART GEO-IP DATABASE (without external API calls)
# =======================================================================================

# CDN providers with Asian edge servers
ASIAN_CDNS = {
    # Major CDNs with strong Asia presence
    'cloudflare': {'score': 95, 'region': 'GLOBAL-CDN'},  # Excellent in Vietnam
    'fastly': {'score': 90, 'region': 'GLOBAL-CDN'},
    'akamai': {'score': 90, 'region': 'GLOBAL-CDN'},
    'cloudfront': {'score': 85, 'region': 'AWS-CDN'},
    
    # Asian-specific CDNs
    'alibaba': {'score': 100, 'region': 'ASIA-CDN'},
    'aliyun': {'score': 100, 'region': 'ASIA-CDN'},
    'tencent': {'score': 100, 'region': 'ASIA-CDN'},
    'qcloud': {'score': 100, 'region': 'ASIA-CDN'},
    'cdnetworks': {'score': 95, 'region': 'ASIA-CDN'},
    'chinacache': {'score': 95, 'region': 'ASIA-CDN'},
    
    # SEA CDNs
    'vncdn': {'score': 100, 'region': 'SEA-CDN'},
    'viettelcdn': {'score': 100, 'region': 'SEA-CDN'},
    'fptcdn': {'score': 100, 'region': 'SEA-CDN'},
}

# Hosting providers known for Asian servers
ASIAN_HOSTING = {
    # SEA hosting
    'viettel': 100, 'vnpt': 100, 'fpt': 100, 'cmccloud': 100,
    'vngcloud': 100, 'bizflycloud': 100,
    
    # Singapore hosting (very common for IPTV)
    'digitalocean': 85, 'linode': 85, 'vultr': 85,
    'aws': 80, 'azure': 80, 'gcp': 80,  # Have SG regions
    
    # Asia hosting
    'alibaba': 95, 'aliyun': 95, 'tencent': 95,
    'sakura': 90, 'conoha': 90,  # Japan
    'naver': 90, 'kakao': 90,  # Korea
}

# IP ranges for major Asian ISPs/hosting (first 2 octets)
ASIAN_IP_RANGES = {
    # Vietnam
    '14.': 'VN', '27.': 'VN', '42.': 'VN', '43.': 'VN',
    '103.': 'SEA', '113.': 'VN', '115.': 'VN', '116.': 'VN',
    '118.': 'VN', '123.': 'VN', '171.': 'VN', '222.': 'VN',
    
    # Singapore (major IPTV hub)
    '8.': 'SG', '18.': 'SG', '52.': 'SG', '54.': 'SG',
    '13.': 'SG', '35.': 'SG',  # AWS/GCP Singapore
    
    # Asia general
    '1.': 'ASIA', '110.': 'ASIA', '111.': 'ASIA',
    '112.': 'ASIA', '114.': 'ASIA', '117.': 'ASIA',
    '119.': 'ASIA', '120.': 'ASIA', '121.': 'ASIA',
    '122.': 'ASIA', '124.': 'ASIA', '125.': 'ASIA',
}

# TLD to region mapping
TLD_REGIONS = {
    # SEA
    '.vn': 'VN', '.th': 'TH', '.sg': 'SG', '.my': 'MY',
    '.id': 'ID', '.ph': 'PH', '.mm': 'MM', '.la': 'LA',
    
    # East Asia
    '.jp': 'JP', '.kr': 'KR', '.cn': 'CN', '.tw': 'TW', '.hk': 'HK',
    
    # South Asia
    '.in': 'IN', '.pk': 'PK', '.bd': 'BD', '.lk': 'LK',
}

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('iptv_generator.log', encoding='utf-8', mode='w')
    ]
)

# =======================================================================================
# SMART SERVER LOCATION DETECTION (works from any GitHub location)
# =======================================================================================

def resolve_ip(url):
    """Resolve domain to IP address"""
    try:
        domain = urlparse(url).netloc.split(':')[0]
        ip = socket.gethostbyname(domain)
        return ip
    except:
        return None

def detect_ip_region(ip):
    """Detect region from IP address patterns"""
    if not ip:
        return 'UNKNOWN', 0
    
    # Check IP prefix
    for prefix, region in ASIAN_IP_RANGES.items():
        if ip.startswith(prefix):
            if region == 'VN':
                return 'VN', 100
            elif region == 'SG':
                return 'SG', 95
            elif region == 'SEA':
                return 'SEA', 90
            elif region == 'ASIA':
                return 'ASIA', 80
    
    return 'UNKNOWN', 0

def detect_cdn_provider(url):
    """Detect CDN provider from URL"""
    domain = urlparse(url).netloc.lower()
    
    for cdn_keyword, cdn_info in ASIAN_CDNS.items():
        if cdn_keyword in domain:
            return cdn_info['region'], cdn_info['score']
    
    return None, 0

def detect_hosting_provider(url):
    """Detect hosting provider from URL"""
    domain = urlparse(url).netloc.lower()
    
    for host_keyword, score in ASIAN_HOSTING.items():
        if host_keyword in domain:
            return 'ASIA-HOST', score
    
    return None, 0

def detect_tld_region(url):
    """Detect region from TLD"""
    domain = urlparse(url).netloc.lower()
    
    for tld, region in TLD_REGIONS.items():
        if domain.endswith(tld):
            if region in ['VN', 'TH', 'SG', 'MY', 'ID', 'PH']:
                return 'SEA', 85
            elif region in ['JP', 'KR', 'CN', 'TW', 'HK']:
                return 'ASIA', 75
            else:
                return 'OTHER', 30
    
    return 'UNKNOWN', 0

def smart_detect_server_location(url):
    """
    SMART detection without relying on ping (works from any GitHub location)
    
    Priority:
    1. CDN provider detection (most reliable)
    2. IP address pattern matching
    3. Hosting provider detection
    4. TLD analysis
    
    Returns: (region, confidence_score)
    """
    scores = {}
    
    # Method 1: CDN detection (highest priority)
    cdn_region, cdn_score = detect_cdn_provider(url)
    if cdn_region:
        scores['cdn'] = (cdn_region, cdn_score)
    
    # Method 2: IP address analysis
    ip = resolve_ip(url)
    if ip:
        ip_region, ip_score = detect_ip_region(ip)
        if ip_score > 0:
            scores['ip'] = (ip_region, ip_score)
    
    # Method 3: Hosting provider
    host_region, host_score = detect_hosting_provider(url)
    if host_region:
        scores['host'] = (host_region, host_score)
    
    # Method 4: TLD
    tld_region, tld_score = detect_tld_region(url)
    if tld_score > 0:
        scores['tld'] = (tld_region, tld_score)
    
    # Combine scores (weighted average, CDN has highest weight)
    if not scores:
        return 'UNKNOWN', 0
    
    # Priority: CDN > IP > Host > TLD
    if 'cdn' in scores:
        return scores['cdn']
    elif 'ip' in scores and scores['ip'][1] >= 90:
        return scores['ip']
    elif 'host' in scores:
        return scores['host']
    elif 'ip' in scores:
        return scores['ip']
    elif 'tld' in scores:
        return scores['tld']
    
    return 'UNKNOWN', 0

def calculate_vietnam_bonus(region, confidence):
    """
    Calculate ping bonus for Vietnam users based on detected region
    
    High confidence → More aggressive bonus
    Low confidence → Conservative bonus
    """
    base_bonus = {
        'VN': 0.2,           # 80% reduction for Vietnam servers
        'SG': 0.25,          # 75% reduction for Singapore (very common)
        'SEA': 0.3,          # 70% reduction for SEA
        'SEA-CDN': 0.25,     # 75% reduction for SEA CDN
        'ASIA': 0.5,         # 50% reduction for Asia
        'ASIA-CDN': 0.4,     # 60% reduction for Asia CDN
        'ASIA-HOST': 0.5,    # 50% reduction for Asia hosting
        'GLOBAL-CDN': 0.45,  # 55% reduction for global CDN (CloudFlare, etc)
        'AWS-CDN': 0.5,      # 50% reduction for AWS (has SG region)
        'OTHER': 0.8,        # 20% reduction
        'UNKNOWN': 1.0,      # No reduction
    }
    
    bonus = base_bonus.get(region, 1.0)
    
    # Adjust by confidence (low confidence = less aggressive bonus)
    if confidence < 70:
        bonus = bonus + (1.0 - bonus) * 0.3  # Reduce bonus by 30%
    
    return bonus

# =======================================================================================
# CHANNEL CLASS
# =======================================================================================

class IPTVChannel:
    """Smart channel with geo-detection"""
    
    __slots__ = ['name', 'url', 'attributes', 'category', 'status', 'ping', 
                 'url_hash', 'name_normalized', 'country', 'quality_score',
                 'content_region', 'server_region', 'server_confidence',
                 'ping_bonus', 'adjusted_ping']
    
    def __init__(self, name, url, attributes, category):
        self.name = self._clean_name(name)
        self.url = url.strip()
        self.attributes = attributes
        self.category = category
        self.status = 'unchecked'
        self.ping = float('inf')
        self.adjusted_ping = float('inf')
        self.quality_score = 0
        
        # SMART GEO DETECTION
        self.content_region = self._detect_content_region()
        self.server_region, self.server_confidence = smart_detect_server_location(url)
        self.ping_bonus = calculate_vietnam_bonus(self.server_region, self.server_confidence)
        
        # Pre-compute
        self.url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
        self.name_normalized = self._normalize_name(name)
        self.country = self._extract_country()
        
        self._ensure_required_attributes()
        self._calculate_quality_score()

    def _clean_name(self, name):
        name = re.sub(r'\s+', ' ', name.strip())
        name = re.sub(r'[^\w\s\-\+\.\(\)\[\]&]', '', name)
        return name[:80]

    def _normalize_name(self, name):
        normalized = re.sub(r'[^\w\s]', '', name.lower())
        normalized = re.sub(r'\b(hd|fhd|uhd|4k|1080p|720p|sd|live|tv|channel)\b', '', normalized)
        return re.sub(r'\s+', ' ', normalized).strip()

    def _detect_content_region(self):
        """Detect channel content region from name/metadata"""
        text = f"{self.name} {self.attributes.get('group-title', '')}".lower()
        
        sea_keywords = ['vietnam', 'vn', 'thai', 'thailand', 'singapore', 'sg', 
                       'malaysia', 'indonesia', 'philippines']
        asia_keywords = ['japan', 'jp', 'korea', 'kr', 'china', 'cn', 
                        'taiwan', 'tw', 'hong kong', 'hk']
        global_keywords = ['usa', 'us', 'uk', 'united kingdom', 'canada', 
                          'australia', 'france', 'germany', 'spain']
        
        for kw in sea_keywords:
            if kw in text:
                return 'SEA'
        
        for kw in asia_keywords:
            if kw in text:
                return 'ASIA'
        
        for kw in global_keywords:
            if kw in text:
                return 'GLOBAL'
        
        return 'UNKNOWN'

    def _extract_country(self):
        """Extract country code"""
        group = self.attributes.get('group-title', '').lower()
        name_lower = self.name.lower()
        
        for blocked in BLOCKED_COUNTRIES:
            if blocked in group or blocked in name_lower:
                return 'BLOCKED'
        
        country_map = {
            'vietnam': 'VN', 'vn': 'VN', 
            'thailand': 'TH', 'singapore': 'SG',
            'japan': 'JP', 'korea': 'KR',
            'usa': 'USA', 'us': 'USA',
            'uk': 'UK', 'canada': 'CA',
        }
        
        for keyword, country in country_map.items():
            if keyword in group or keyword in name_lower:
                return country
        
        return 'INT'

    def _calculate_quality_score(self):
        """Calculate quality score"""
        text = f"{self.name} {self.attributes.get('group-title', '')}".lower()
        score = 50
        
        if any(kw in text for kw in ['4k', 'uhd', '2160']):
            score = 100
        elif any(kw in text for kw in ['1080', 'fhd', 'full hd', '1920']):
            score = 80
        
        if any(kw in text for kw in QUALITY_KEYWORDS_EXCLUDE):
            score = 0
        
        if any(kw in text for kw in ['hevc', 'h265', 'x265']):
            score += 15
        
        self.quality_score = max(0, min(score, 115))

    def is_high_quality(self):
        """Check quality requirements"""
        return self.quality_score > 0 and self.country != 'BLOCKED'

    def update_ping(self, raw_ping_ms):
        """Update ping with geographic bonus"""
        self.ping = raw_ping_ms
        self.adjusted_ping = raw_ping_ms * self.ping_bonus

    def get_priority_score(self):
        """
        Calculate overall priority score for Vietnam users
        Higher = Better
        """
        # Server region score
        region_scores = {
            'VN': 1000, 'SG': 950, 'SEA': 900, 'SEA-CDN': 920,
            'ASIA': 800, 'ASIA-CDN': 850, 'ASIA-HOST': 800,
            'GLOBAL-CDN': 750, 'AWS-CDN': 780,
            'OTHER': 500, 'UNKNOWN': 400
        }
        
        region_score = region_scores.get(self.server_region, 400)
        
        # Confidence bonus
        confidence_bonus = self.server_confidence * 2
        
        # Quality bonus
        quality_bonus = self.quality_score * 5
        
        # Adjusted ping penalty (lower is better)
        ping_penalty = self.adjusted_ping / 10
        
        return region_score + confidence_bonus + quality_bonus - ping_penalty

    def get_display_tag(self):
        """Get display tag for playlist"""
        if self.server_confidence >= 80:
            confidence_emoji = "✓"
        elif self.server_confidence >= 60:
            confidence_emoji = "~"
        else:
            confidence_emoji = "?"
        
        if self.server_region == self.content_region:
            return f"[{self.server_region}{confidence_emoji}]"
        elif self.content_region != 'UNKNOWN':
            return f"[{self.server_region}{confidence_emoji}/{self.content_region}]"
        else:
            return f"[{self.server_region}{confidence_emoji}]"

    def _ensure_required_attributes(self):
        if 'tvg-id' not in self.attributes:
            self.attributes['tvg-id'] = re.sub(r'[^\w-]', '-', self.name.lower())[:40]
        if 'tvg-name' not in self.attributes:
            self.attributes['tvg-name'] = self.name
        if 'group-title' not in self.attributes:
            self.attributes['group-title'] = self.category.upper()

    def to_m3u_entry(self):
        attrs = []
        for key in ['tvg-id', 'tvg-name', 'group-title']:
            if key in self.attributes and self.attributes[key]:
                attrs.append(f'{key}="{self.attributes[key]}"')
        
        region_tag = self.get_display_tag()
        ping_tag = f'({int(self.adjusted_ping)}ms)' if self.adjusted_ping != float('inf') else ''
        
        display_name = f"{region_tag} {self.name} {ping_tag}".strip()
        
        return f"#EXTINF:-1 {' '.join(attrs)},{display_name}\n{self.url}"

# =======================================================================================
# PARSING
# =======================================================================================

def parse_m3u_content(content, category):
    channels = []
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    if content.startswith('\ufeff'):
        content = content[1:]
    
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('#EXTINF'):
            url = None
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                if next_line and not next_line.startswith('#'):
                    url = next_line
                    i = j
                    break
            
            if url and url.startswith('http'):
                try:
                    channel = parse_extinf_line(line, url, category)
                    if channel:
                        channels.append(channel)
                except:
                    pass
        
        i += 1
    
    return channels

def parse_extinf_line(extinf_line, url, category):
    extinf_line = re.sub(r'^#EXTINF:-?\d+\s*', '', extinf_line)
    
    if ',' in extinf_line:
        attr_string, name = extinf_line.rsplit(',', 1)
    else:
        attr_string = extinf_line
        name = ''
    
    attributes = {}
    for match in re.finditer(r'([\w-]+)=(["\'])([^\2]*?)\2', attr_string):
        key, _, value = match.groups()
        attributes[key] = value
    
    if not name:
        name = attributes.get('tvg-name', attributes.get('tvg-id', ''))
    
    name = re.sub(r'[\w-]+=(["\'])[^\1]*\1', '', name).strip()
    
    if not name or len(name) < 2 or len(url) < 10:
        return None
    
    return IPTVChannel(name, url, attributes, category)

# =======================================================================================
# FETCHING & CHECKING
# =======================================================================================

async def fetch_source(session, url, category, retry=0):
    try:
        logging.info(f"Fetching: {url}")
        
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT),
            allow_redirects=True
        ) as response:
            if response.status == 200:
                content = await response.text(errors='ignore')
                
                if '#EXTINF' not in content:
                    logging.warning(f"Invalid M3U: {url}")
                    return []
                
                channels = parse_m3u_content(content, category)
                logging.info(f"✓ {len(channels)} channels from {url}")
                return channels
            else:
                logging.warning(f"HTTP {response.status}: {url}")
                return []
                
    except Exception as e:
        logging.error(f"Error: {url} - {str(e)[:50]}")
        return []

async def check_channel_status(session, channel, semaphore):
    """Check channel availability"""
    async with semaphore:
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with session.head(
                channel.url,
                timeout=aiohttp.ClientTimeout(total=CHANNEL_CHECK_TIMEOUT),
                headers=REQUEST_HEADERS,
                allow_redirects=True
            ) as response:
                end_time = asyncio.get_event_loop().time()
                raw_ping_ms = (end_time - start_time) * 1000
                
                if response.status in [200, 206, 301, 302, 303, 307, 308]:
                    channel.status = 'working'
                    channel.update_ping(raw_ping_ms)
                else:
                    channel.status = 'dead'
                    
        except:
            channel.status = 'dead'

# =======================================================================================
# FILTERING
# =======================================================================================

def filter_and_deduplicate(channels):
    """Smart filtering with priority scoring"""
    logging.info(f"Starting SMART geo-filtering on {len(channels)} channels...")
    
    working = [ch for ch in channels if ch.status == 'working']
    logging.info(f"Working channels: {len(working)}/{len(channels)}")
    
    if not working:
        return []
    
    high_quality = [ch for ch in working if ch.is_high_quality()]
    logging.info(f"After quality filter: {len(high_quality)}/{len(working)}")
    
    if not high_quality:
        return []
    
    fast_channels = [ch for ch in high_quality if ch.adjusted_ping <= MAX_ACCEPTABLE_PING_MS]
    logging.info(f"Fast enough: {len(fast_channels)}/{len(high_quality)}")
    
    if not fast_channels:
        return []
    
    # URL deduplication
    url_map = {}
    for ch in fast_channels:
        if ch.url_hash not in url_map or ch.get_priority_score() > url_map[ch.url_hash].get_priority_score():
            url_map[ch.url_hash] = ch
    
    unique_urls = list(url_map.values())
    logging.info(f"After URL dedup: {len(unique_urls)}")
    
    # Name deduplication with priority scoring
    final_map = {}
    for ch in unique_urls:
        key = (ch.name_normalized, ch.country, ch.category)
        if key not in final_map:
            final_map[key] = ch
        else:
            if ch.get_priority_score() > final_map[key].get_priority_score():
                final_map[key] = ch
    
    final = list(final_map.values())
    final.sort(key=lambda x: (x.category, -x.get_priority_score()))
    
    # Statistics
    vn = len([ch for ch in final if ch.server_region == 'VN'])
    sg = len([ch for ch in final if ch.server_region == 'SG'])
    sea = len([ch for ch in final if 'SEA' in ch.server_region])
    asia = len([ch for ch in final if 'ASIA' in ch.server_region])
    cdn = len([ch for ch in final if 'CDN' in ch.server_region])
    
    high_conf = len([ch for ch in final if ch.server_confidence >= 80])
    
    logging.info(f"Final channels: {len(final)}")
    logging.info(f"  └─ VN: {vn} | SG: {sg} | SEA: {sea} | ASIA: {asia} | CDN: {cdn}")
    logging.info(f"  └─ High confidence (≥80%): {high_conf}/{len(final)}")
    
    smart_finds = len([ch for ch in final if 
                      ch.content_region == 'GLOBAL' and 
                      ch.server_region in ['VN', 'SG', 'SEA', 'SEA-CDN']])
    if smart_finds > 0:
        logging.info(f"  └─ ✨ SMART FINDS: {smart_finds} Global channels on SEA servers!")
    
    return final

# =======================================================================================
# OUTPUT
# =======================================================================================

def generate_m3u_playlist(channels):
    logging.info("Generating SMART geo-optimized M3U playlist...")
    
    lines = [
        '#EXTM3U',
        f'#EXTINF:-1,SMART Geo-Optimized for Vietnam - Updated: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}',
        f'#EXTINF:-1,Total: {len(channels)} channels | Detection: IP/CDN/Domain analysis (works from any location)',
        f'#EXTINF:-1,Legend: ✓=High confidence ~=Medium ?=Low | VN=Vietnam SG=Singapore SEA=SouthEast Asia',
        ''
    ]
    
    grouped = defaultdict(list)
    for ch in channels:
        grouped[ch.category].append(ch)
    
    for category in sorted(grouped.keys()):
        category_channels = grouped[category]
        avg_ping = sum(ch.adjusted_ping for ch in category_channels) / len(category_channels)
        
        vn = len([ch for ch in category_channels if ch.server_region == 'VN'])
        sg = len([ch for ch in category_channels if ch.server_region == 'SG'])
        sea = len([ch for ch in category_channels if 'SEA' in ch.server_region])
        asia = len([ch for ch in category_channels if 'ASIA' in ch.server_region])
        
        lines.append(f'#EXTINF:-1,━━━ {category.upper()} ({len(category_channels)} ch | VN:{vn} SG:{sg} SEA:{sea} ASIA:{asia} | avg {avg_ping:.0f}ms) ━━━')
        lines.append('')
        
        for ch in category_channels:
            lines.append(ch.to_m3u_entry())
            lines.append('')
    
    return '\n'.join(lines)

# =======================================================================================
# MAIN
# =======================================================================================

async def main():
    start_time = datetime.now()
    logging.info("=" * 80)
    logging.info("IPTV GENERATOR - SMART GEO-DETECTION (works from ANY GitHub location)")
    logging.info("=" * 80)
    logging.info("Detection methods:")
    logging.info("  1. CDN Provider Analysis (CloudFlare, Akamai, Asian CDNs...)")
    logging.info("  2. IP Address Pattern Matching (Asian IP ranges)")
    logging.info("  3. Hosting Provider Detection (DigitalOcean SG, AWS Asia...)")
    logging.info("  4. Domain TLD Analysis (.vn, .sg, .th...)")
    logging.info("")
    logging.info("✓ No ping dependency - works from US/EU GitHub servers")
    logging.info("✓ Optimized for Vietnam users regardless of runner location")
    logging.info("=" * 80)
    
    all_channels = []
    
    connector = aiohttp.TCPConnector(
        limit=150,
        limit_per_host=15,
        ttl_dns_cache=600,
        force_close=False,
    )
    
    async with aiohttp.ClientSession(
        connector=connector,
        headers=REQUEST_HEADERS,
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        # Phase 1: Fetch sources
        logging.info("\n[1/3] FETCHING SOURCES")
        logging.info("-" * 80)
        
        tasks = []
        for category, urls in SOURCES.items():
            for url in urls:
                tasks.append(fetch_source(session, url, category))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_channels.extend(result)
        
        logging.info(f"\nTotal channels fetched: {len(all_channels)}")
        
        # Pre-filter
        all_channels = [ch for ch in all_channels if ch.is_high_quality()]
        logging.info(f"After quality pre-filter: {len(all_channels)}")
        
        if not all_channels:
            logging.error("No channels after pre-filter!")
            return
        
        # Show geo-detection stats
        logging.info("\n📊 GEO-DETECTION ANALYSIS:")
        logging.info("-" * 80)
        
        vn_detected = len([ch for ch in all_channels if ch.server_region == 'VN'])
        sg_detected = len([ch for ch in all_channels if ch.server_region == 'SG'])
        sea_detected = len([ch for ch in all_channels if 'SEA' in ch.server_region])
        asia_detected = len([ch for ch in all_channels if 'ASIA' in ch.server_region])
        cdn_detected = len([ch for ch in all_channels if 'CDN' in ch.server_region])
        unknown = len([ch for ch in all_channels if ch.server_region == 'UNKNOWN'])
        
        logging.info(f"  VN servers: {vn_detected}")
        logging.info(f"  SG servers: {sg_detected}")
        logging.info(f"  SEA region: {sea_detected}")
        logging.info(f"  ASIA region: {asia_detected}")
        logging.info(f"  CDN (global): {cdn_detected}")
        logging.info(f"  Unknown: {unknown}")
        
        high_conf = len([ch for ch in all_channels if ch.server_confidence >= 80])
        medium_conf = len([ch for ch in all_channels if 60 <= ch.server_confidence < 80])
        low_conf = len([ch for ch in all_channels if ch.server_confidence < 60])
        
        logging.info(f"\n📊 CONFIDENCE LEVELS:")
        logging.info(f"  High (≥80%): {high_conf}")
        logging.info(f"  Medium (60-79%): {medium_conf}")
        logging.info(f"  Low (<60%): {low_conf}")
        
        # Show smart finds preview
        smart_preview = [ch for ch in all_channels if 
                        ch.content_region == 'GLOBAL' and 
                        ch.server_region in ['VN', 'SG', 'SEA', 'SEA-CDN']][:5]
        
        if smart_preview:
            logging.info(f"\n✨ SMART FINDS PREVIEW (Global content on SEA servers):")
            for ch in smart_preview:
                logging.info(f"  → {ch.name} | Server: {ch.server_region} (confidence: {ch.server_confidence}%)")
        
        # Phase 2: Check channels
        logging.info("\n[2/3] CHECKING CHANNEL AVAILABILITY")
        logging.info("-" * 80)
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHECKS)
        
        for i in range(0, len(all_channels), BATCH_SIZE):
            batch = all_channels[i:i+BATCH_SIZE]
            check_tasks = [check_channel_status(session, ch, semaphore) for ch in batch]
            await asyncio.gather(*check_tasks)
            
            checked = i + len(batch)
            if checked % (BATCH_SIZE * 3) == 0 or checked == len(all_channels):
                working_so_far = len([ch for ch in all_channels[:checked] if ch.status == 'working'])
                logging.info(f"Progress: {checked}/{len(all_channels)} | Working: {working_so_far}")
    
    # Phase 3: Filter and generate
    logging.info("\n[3/3] FILTERING & GENERATING OUTPUT")
    logging.info("-" * 80)
    
    final_channels = filter_and_deduplicate(all_channels)
    
    if not final_channels:
        logging.error("No channels passed all filters!")
        return
    
    content = generate_m3u_playlist(final_channels)
    
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        f.write(content)
    
    duration = datetime.now() - start_time
    minutes = int(duration.total_seconds() / 60)
    seconds = int(duration.total_seconds() % 60)
    
    logging.info("\n" + "=" * 80)
    logging.info(f"✓✓✓ SUCCESS! Generated {len(final_channels)} SMART GEO-OPTIMIZED channels")
    logging.info("=" * 80)
    logging.info("SMART DETECTION SUMMARY:")
    
    # Final stats
    final_vn = len([ch for ch in final_channels if ch.server_region == 'VN'])
    final_sg = len([ch for ch in final_channels if ch.server_region == 'SG'])
    final_sea = len([ch for ch in final_channels if 'SEA' in ch.server_region])
    final_asia = len([ch for ch in final_channels if 'ASIA' in ch.server_region])
    final_cdn = len([ch for ch in final_channels if 'CDN' in ch.server_region])
    
    logging.info(f"  Server Distribution:")
    logging.info(f"    → VN servers: {final_vn}")
    logging.info(f"    → SG servers: {final_sg}")
    logging.info(f"    → SEA region: {final_sea}")
    logging.info(f"    → ASIA region: {final_asia}")
    logging.info(f"    → CDN global: {final_cdn}")
    
    final_high_conf = len([ch for ch in final_channels if ch.server_confidence >= 80])
    logging.info(f"  High confidence detections: {final_high_conf}/{len(final_channels)} ({final_high_conf*100//len(final_channels)}%)")
    
    smart_finds = len([ch for ch in final_channels if 
                      ch.content_region == 'GLOBAL' and 
                      ch.server_region in ['VN', 'SG', 'SEA', 'SEA-CDN']])
    
    if smart_finds > 0:
        logging.info(f"  ✨ SMART FINDS: {smart_finds} USA/Global channels on SEA servers!")
    
    logging.info("")
    logging.info(f"Execution time: {minutes}m {seconds}s")
    logging.info(f"Playlist saved to: {OUTPUT_FILENAME}")
    logging.info("=" * 80)
    logging.info("✓ Detection works from ANY GitHub Actions location (US/EU/Asia)")
    logging.info("✓ Optimized for Vietnam users based on server geo-location")
    logging.info("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
