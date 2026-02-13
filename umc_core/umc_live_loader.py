import urllib.request
import xml.etree.ElementTree as ET
import time
import ssl

class LiveWorldLoader:
    """
    Загрузчик реальных данных из RSS и YouTube с балансировкой WEST vs EAST.
    """
    def __init__(self):
        self.rss_sources_west = [
            "https://www.reddit.com/r/geopolitics/hot/.rss",
            "https://cyberscoop.com/feed/",
            "https://www.bellingcat.com/feed/",
            "https://thediplomat.com/feed/"
        ]
        self.rss_sources_east = [
            "https://tass.ru/rss/v2.xml",
            "https://ria.ru/export/rss2/archive/index.xml",
            "https://www.interfax.ru/rss.asp"
        ]
        self.youtube_ids = [
            "UC9RM-iSvTu1uPJb8X5yp3EQ", # PERUN
            "UCdeMVChrumySxV9N1w0Au-w"  # TASK AND PURPOSE
        ]
        self.ssl_context = ssl._create_unverified_context()

    def _fetch_url(self, url):
        try:
            # ТАСС и РИА могут требовать более специфические заголовки
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/rss+xml, application/xml, text/xml'
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=self.ssl_context, timeout=15) as response:
                return response.read()
        except Exception as e:
            return None

    def get_rss_events(self, sources, label):
        events = []
        for url in sources:
            content = self._fetch_url(url)
            if content:
                try:
                    # Попытка декодирования (РИА и ТАСС могут использовать windows-1251)
                    try:
                        text = content.decode('utf-8')
                    except UnicodeDecodeError:
                        text = content.decode('windows-1251', errors='ignore')
                    
                    root = ET.fromstring(text)
                    for item in root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                        title_node = item.find('title') or item.find('{http://www.w3.org/2005/Atom}title')
                        if title_node is not None:
                            events.append({
                                "type": "fast",
                                "source": f"{label}_{url.split('/')[2]}",
                                "event": title_node.text.strip() if title_node.text else "No Title",
                                "tags": ["rss", label.lower()]
                            })
                except Exception as e:
                    pass
        return events

    def get_youtube_events(self):
        events = []
        for channel_id in self.youtube_ids:
            url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
            content = self._fetch_url(url)
            if content:
                try:
                    root = ET.fromstring(content)
                    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                        title = entry.find('{http://www.w3.org/2005/Atom}title').text
                        events.append({
                            "type": "fast",
                            "source": f"YT_{channel_id[:5]}",
                            "event": title,
                            "tags": ["youtube", "expert_analysis"]
                        })
                except Exception as e:
                    pass
        return events[:10]

    def get_all_realtime_data(self):
        print(f"[LIVE] Загрузка сбалансированных данных (West vs East)...")
        west_data = self.get_rss_events(self.rss_sources_west, "WEST")
        east_data = self.get_rss_events(self.rss_sources_east, "EAST")
        
        # Балансировка: берем равное количество (макс по 15 с каждой стороны)
        limit = min(len(west_data), len(east_data), 15) if west_data and east_data else 10
        
        balanced = west_data[:limit] + east_data[:limit]
        
        # Добавляем YouTube
        yt_data = self.get_youtube_events()
        
        return balanced + yt_data

if __name__ == "__main__":
    loader = LiveWorldLoader()
    results = loader.get_all_realtime_data()
    for r in results:
        print(f"[{r['source']}] {r['event']}")
