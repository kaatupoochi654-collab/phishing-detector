import re
import urllib.parse
import socket
import ssl
import requests
from bs4 import BeautifulSoup
import whois
from datetime import datetime
import tldextract
import warnings
warnings.filterwarnings('ignore')

class URLFeatureExtractor:
    def __init__(self):
        self.feature_names = [
            'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
            'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
            'Domain_registeration_length', 'Favicon', 'Port', 'HTTPS_token', 'Request_URL',
            'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
            'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
            'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
            'Statistical_report'
        ]
        self.shorteners = ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly', 'buff.ly', 'is.gd', 'adf.ly']
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})

    def extract(self, url):
        features = {}
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc or url

        # URL-based features (instant, no network)
        features['having_IP_Address'] = 1 if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', url) else -1
        features['URL_Length'] = 1 if len(url) > 75 else (0 if len(url) >= 54 else -1)
        features['Shortining_Service'] = 1 if any(s in url.lower() for s in self.shorteners) else -1
        features['having_At_Symbol'] = 1 if '@' in url else -1
        features['double_slash_redirecting'] = 1 if url.count('//') > 1 else -1
        features['Prefix_Suffix'] = 1 if '-' in domain else -1

        ext = tldextract.extract(domain)
        sub_count = ext.subdomain.count('.') + (1 if ext.subdomain else 0)
        features['having_Sub_Domain'] = 1 if sub_count > 1 else (0 if sub_count == 1 else -1)

        features['Port'] = 1 if parsed.port and parsed.port not in [80, 443] else -1
        features['HTTPS_token'] = 1 if 'https' in domain.lower() else -1

        # Network-based features (with 2-second timeout each)
        features['SSLfinal_State'] = self._safe_run(lambda: self._check_ssl(url), 1)
        features['Domain_registeration_length'] = self._safe_run(lambda: self._check_domain_reg(domain), 1)
        features['Favicon'] = self._safe_run(lambda: self._check_favicon(url, domain), 0)
        features['Request_URL'] = self._safe_run(lambda: self._check_request_url(url, domain), 0)
        features['URL_of_Anchor'] = self._safe_run(lambda: self._check_anchors(url, domain), 0)
        features['Links_in_tags'] = self._safe_run(lambda: self._check_links_in_tags(url, domain), 0)
        features['SFH'] = self._safe_run(lambda: self._check_sfh(url, domain), 0)
        features['Submitting_to_email'] = self._safe_run(lambda: self._check_submit_email(url), 0)
        features['Abnormal_URL'] = self._safe_run(lambda: self._check_abnormal(domain), 1)
        features['Redirect'] = self._safe_run(lambda: self._check_redirect(url), 0)
        features['on_mouseover'] = self._safe_run(lambda: self._check_mouseover(url), 0)
        features['RightClick'] = self._safe_run(lambda: self._check_rightclick(url), 0)
        features['popUpWidnow'] = self._safe_run(lambda: self._check_popup(url), 0)
        features['Iframe'] = self._safe_run(lambda: self._check_iframe(url), 0)
        features['age_of_domain'] = self._safe_run(lambda: self._check_domain_age(domain), 1)
        features['DNSRecord'] = self._safe_run(lambda: self._check_dns(domain), 1)
        features['web_traffic'] = 0
        features['Page_Rank'] = 0
        features['Google_Index'] = -1 if any(domain.endswith(t) for t in ['.com', '.org', '.net', '.edu', '.gov']) else 0
        features['Links_pointing_to_page'] = 0
        features['Statistical_report'] = 1 if any(k in domain.lower() for k in ['secure', 'login', 'verify', 'bank', 'account', 'update', 'confirm']) else -1

        return [features[f] for f in self.feature_names]

    def _safe_run(self, func, default):
        """Run function with exception handling"""
        try:
            return func()
        except:
            return default

    def _check_ssl(self, url):
        if not url.startswith('https'): return 1
        hostname = urllib.parse.urlparse(url).netloc.split(':')[0]
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=2) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                return -1 if ssock.getpeercert() else 1

    def _check_domain_reg(self, domain):
        w = whois.whois(domain)
        if w.expiration_date:
            exp = w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date
            return 1 if (exp - datetime.now()).days < 365 else -1
        return 1

    def _check_favicon(self, url, domain):
        r = self.session.get(url, timeout=2, allow_redirects=True)
        soup = BeautifulSoup(r.text, 'html.parser')
        icon = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
        if icon and icon.get('href', '').startswith('http'):
            return 1 if domain not in icon['href'] else -1
        return -1

    def _check_request_url(self, url, domain):
        r = self.session.get(url, timeout=2, allow_redirects=True)
        soup = BeautifulSoup(r.text, 'html.parser')
        total, external = 0, 0
        for tag in ['img', 'video', 'audio', 'source']:
            for item in soup.find_all(tag):
                total += 1
                src = item.get('src', '')
                if src.startswith('http') and domain not in src: external += 1
        if total == 0: return 0
        ratio = external / total
        return 1 if ratio > 0.61 else (0 if ratio >= 0.22 else -1)

    def _check_anchors(self, url, domain):
        r = self.session.get(url, timeout=2, allow_redirects=True)
        soup = BeautifulSoup(r.text, 'html.parser')
        anchors = soup.find_all('a')
        total = len(anchors)
        if total == 0: return 0
        susp = sum(1 for a in anchors if a.get('href', '').startswith('#') or 'javascript:' in a.get('href', '').lower() or (a.get('href', '').startswith('http') and domain not in a.get('href', '')))
        ratio = susp / total
        return 1 if ratio > 0.67 else (0 if ratio >= 0.31 else -1)

    def _check_links_in_tags(self, url, domain):
        r = self.session.get(url, timeout=2, allow_redirects=True)
        soup = BeautifulSoup(r.text, 'html.parser')
        tags = soup.find_all(['meta', 'script', 'link'])
        total = len(tags)
        if total == 0: return 0
        ext = sum(1 for t in tags for a in ['href', 'src', 'content', 'url'] if t.get(a, '').startswith('http') and domain not in t.get(a, ''))
        ratio = ext / total if total > 0 else 0
        return 1 if ratio > 0.81 else (0 if ratio >= 0.17 else -1)

    def _check_sfh(self, url, domain):
        r = self.session.get(url, timeout=2, allow_redirects=True)
        soup = BeautifulSoup(r.text, 'html.parser')
        for form in soup.find_all('form'):
            action = form.get('action', '')
            if not action or action.lower() == 'about:blank': return -1
            if action.startswith('http') and domain not in action: return 1
        return 0

    def _check_submit_email(self, url):
        r = self.session.get(url, timeout=2, allow_redirects=True)
        return 1 if 'mailto:' in r.text.lower() else -1

    def _check_abnormal(self, domain):
        w = whois.whois(domain)
        return -1 if w.domain_name else 1

    def _check_redirect(self, url):
        r = self.session.get(url, timeout=2, allow_redirects=True)
        return 0 if len(r.history) <= 1 else (1 if len(r.history) < 4 else -1)

    def _check_mouseover(self, url):
        r = self.session.get(url, timeout=2, allow_redirects=True)
        return 1 if 'onmouseover' in r.text.lower() else -1

    def _check_rightclick(self, url):
        r = self.session.get(url, timeout=2, allow_redirects=True)
        t = r.text.lower()
        return 1 if 'event.button==2' in t or 'oncontextmenu' in t else -1

    def _check_popup(self, url):
        r = self.session.get(url, timeout=2, allow_redirects=True)
        t = r.text.lower()
        return 1 if 'window.open' in t and ('<input' in t or 'prompt' in t) else -1

    def _check_iframe(self, url):
        r = self.session.get(url, timeout=2, allow_redirects=True)
        return 1 if '<iframe' in r.text.lower() else -1

    def _check_domain_age(self, domain):
        w = whois.whois(domain)
        if w.creation_date:
            creation = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
            return 1 if (datetime.now() - creation).days < 180 else -1
        return 1

    def _check_dns(self, domain):
        socket.gethostbyname(domain)
        return -1
