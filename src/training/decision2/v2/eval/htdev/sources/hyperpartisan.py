"""SemEval-2019 Task 4 by-article: is the article hyperpartisan?

Only the crowd-labelled by-article parts (test, then training); the by-publisher parts
(distant labels) are never read. State: the title and the body's first 2,000
characters, cut at a sentence end. The publisher (host of the ground-truth URL, shown
nowhere) is the group. Article ids are unique only within a part, so ids carry the split.
"""

from __future__ import annotations

import html
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from pathlib import Path

from v2.eval.htdev.common import HtCandidate, cut_at_sentence, make, noul, spec

TASK = "ideology/hyperpartisan"
PARTS = (
    (
        "test",
        "articles-test-byarticle-20181207.zip",
        "ground-truth-test-byarticle-20181207.zip",
    ),
    (
        "train",
        "articles-training-byarticle-20181122.zip",
        "ground-truth-training-byarticle-20181122.zip",
    ),
)
QUESTION = noul(
    "The state is a news article: its title and the beginning of its body.",
    "The article is hyperpartisan: it argues for one political side in an extreme, "
    "one-sided way.",
    "The article is not hyperpartisan.",
)

SPEC = spec(
    key="hyperpartisan",
    dataset_id="zenodo:10.5281/zenodo.5776081 (by-article)",
    revision="zenodo-record-5776081",
    licence="cc-by-4.0",
    evidence="zenodo.org/records/5776081 metadata licence cc-by-4.0; README.txt 'The "
    "collection (including labels) are licensed under ... CC BY 4.0'",
    label_provenance="by-article labels from crowdsourcing; only consensus articles",
    tasks=(TASK,),
)


def xml_member(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        (name,) = [n for n in archive.namelist() if n.endswith(".xml")]
        return archive.read(name)


def publisher(url: str) -> str:
    host = urllib.parse.urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def paragraphs(element: ET.Element) -> str:
    blocks = []
    for child in element:
        text = " ".join(html.unescape("".join(child.itertext())).split())
        if text:
            blocks.append(text)
    if not blocks:
        text = " ".join(html.unescape("".join(element.itertext())).split())
        blocks = [text] if text else []
    return "\n".join(blocks)


def candidates(root: Path) -> Iterator[HtCandidate]:
    for split, articles, truth in PARTS:
        labels = {
            node.get("id"): (node.get("hyperpartisan"), node.get("url") or "")
            for node in ET.fromstring(xml_member(root / truth)).iter("article")
        }
        for index, node in enumerate(
            ET.fromstring(xml_member(root / articles)).iter("article")
        ):
            article_id = node.get("id")
            flag, url = labels.get(article_id, (None, ""))
            if flag not in ("true", "false"):
                continue
            title = " ".join(html.unescape(node.get("title") or "").split())
            body = cut_at_sentence(paragraphs(node))
            if not title and not body:
                continue
            item = make(
                SPEC,
                TASK,
                f"{split}:{article_id}",
                publisher(url) or f"unknown:{split}:{article_id}",
                split,
                articles,
                index,
                {"title": title, "body": body},
                dict(QUESTION),
                flag == "true",
                overlap_texts=[title, body],
            )
            if item:
                yield item
