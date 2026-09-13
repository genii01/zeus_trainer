"""Validate publishable HTML's local assets, anchors and interactive references."""
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1] / 'docs'


class Document(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.links = []
        self.controls = []
        self.lang = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'html':
            self.lang = attrs.get('lang')
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        for attr in ('href', 'src'):
            if attrs.get(attr):
                self.links.append(attrs[attr])
        if attrs.get('aria-controls'):
            self.controls.extend(attrs['aria-controls'].split())


def main():
    files = list(ROOT.rglob('*.html'))
    assert files, 'No HTML documents found'
    for path in files:
        doc = Document()
        doc.feed(path.read_text())
        assert doc.lang == 'ko', f'{path}: expected Korean document language'
        assert all(count == 1 for count in Counter(doc.ids).values()), f'{path}: duplicate ID'
        for control in doc.controls:
            assert control in doc.ids, f'{path}: missing controlled element {control}'
        for link in doc.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            assert target.is_relative_to(ROOT), f'{path}: link outside Pages directory: {link}'
            assert target.is_file(), f'{path}: missing local link: {link}'
            if url.fragment and target.suffix == '.html':
                target_doc = Document()
                target_doc.feed(target.read_text())
                assert unquote(url.fragment) in target_doc.ids, f'{path}: missing anchor {link}'
        print(f'OK: {path.relative_to(ROOT)} ({len(doc.links)} links, {len(doc.ids)} IDs)')


if __name__ == '__main__':
    main()
