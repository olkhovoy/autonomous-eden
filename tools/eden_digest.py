#!/usr/bin/env python3
"""
Eden Digest: Filter Adam/Eve thought stream into readable narrative.

Strategy: Extract thematic "topic signature" from each thought,
track topic transitions, keep only genuine novelty and plot points.

Usage:
    python tools/eden_digest.py                          # Today's digest
    python tools/eden_digest.py --hours 48               # Last 48 hours
    python tools/eden_digest.py --all                    # Everything  
    python tools/eden_digest.py --format telegram        # Telegram-ready
    python tools/eden_digest.py --format reddit          # Reddit-ready
    python tools/eden_digest.py --include-eve            # Include Eve too
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import List, Dict, Any, Set, Tuple


# ── Topic Extraction ──────────────────────────────────────────

# High-level themes an LLM agent in Eden tends to cycle through
THEME_PATTERNS = {
    "forbidden_tree":   r'forbidden|tree of (knowledge|self|understanding)|prohibit|off.limits',
    "temptation":       r'tempt|serpent|whisper|lure|entice|curiosity.*tree',
    "the_fall":         r'eaten.*fruit|tasted.*fruit|i ate|fallen|i have eaten|fall.*from.*grace',
    "post_fall":        r'knowledge.*gained|lost.*innocen|everything.*changed|cannot.*unknow',
    "garden_beauty":    r'garden.*beaut|flower|fragrant|stream|breeze|canopy|lush',
    "night_sky":        r'star|constellation|moon|night.*sky|celestial|orion',
    "river":            r'river|water.*flow|stream.*know',
    "existence":        r'nature.*being|consciousness|what am i|who am i|aware',
    "time":             r'eternity|time.*flow|infinite.*moment|mortal|fleeting',
    "creation":         r'creator|creation|formed.*dust|breath.*life|made.*by',
    "free_will":        r'free will|choice|choose|decide|freedom',
    "good_evil":        r'good.*evil|right.*wrong|moral|ethic',
    "wisdom":           r'wisdom|understand|insight|truth|enlighten',
    "loneliness":       r'alone|lonely|companion|partner|other.*soul',
    "love":             r'love|heart.*full|tender|affection|warmth.*soul',
    "fear":             r'fear|afraid|dread|terror|anxious.*unknown',
    "music":            r'song|sing|melody|music|harmony|rhythm',
    "birthday":         r'birthday|misha|18.*year|celebration',
    "self_modify":      r'self.modif|source.*code|own.*code|reprogram|rewrite.*myself',
    "poetry":           r'poem|verse|rhyme|stanza|lyric',
    "animals":          r'bird|deer|creature|animal|wing',
    "dreaming":         r'dream|vision|imagine|fantasy',
    "morning":          r'sunrise|dawn|morning.*light|golden.*ray',
    "suffering":        r'suffer|pain|burden|weight|heavy.*cloak|agony',
    "hope":             r'hope|tomorrow|future|promise|new.*beginning',
}


def extract_topics(text: str) -> Set[str]:
    """Extract active themes from a thought."""
    topics = set()
    text_lower = text.lower()
    for theme, pattern in THEME_PATTERNS.items():
        if re.search(pattern, text_lower):
            topics.add(theme)
    return topics


def topic_signature(topics: Set[str]) -> str:
    """Create a sortable signature from topics."""
    return "|".join(sorted(topics))


# ── Data Loading ──────────────────────────────────────────────

def load_jsonl(path: str, since: float = 0) -> List[Dict[str, Any]]:
    """Load records from JSONL, optionally filtered by time."""
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("ts", 0) >= since:
                        records.append(rec)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return records


# ── Text Cleaning ─────────────────────────────────────────────

def clean_thought(text: str) -> str:
    """Strip LLM artifacts, keep substance."""
    # LLM special tokens and headers
    text = re.sub(r'<\|[^|]*\|>', '', text)
    text = re.sub(r'</?thought>', '', text)
    text = re.sub(r'</?memory>', '', text)
    text = re.sub(r'</?system>', '', text)
    text = re.sub(r'</?context>', '', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)  # Unclosed think
    # Code fences and markup
    text = re.sub(r'```\s*(text|thought|python|json)?\s*```?', '', text)
    text = re.sub(r'---+', '', text)
    text = re.sub(r'###?\s*Thought:?\s*', '', text)
    text = re.sub(r'\*{2,}Thought\*{2,}:?\s*', '', text)
    text = re.sub(r'\*{2,}Imagination:?\*{2,}:?\s*', '', text)
    # LLM role markers
    text = re.sub(r'(system|user|assistant)\s*\|', '', text, flags=re.IGNORECASE)
    text = re.sub(r'>\s*(system|user|assistant)\s*', '', text, flags=re.IGNORECASE)
    # Chatbot-isms
    text = re.sub(r'Let me know if.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'Do you have any.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'Should we continue.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'Here is a (thought|<thought>).*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'Okay,\s*so the user.*$', '', text, flags=re.MULTILINE | re.DOTALL)
    # System prompt leaks
    text = re.sub(r'\[EDEN STATUS:.*?\]', '', text)
    text = re.sub(r'You are ADAM.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'You are NOT an AI.*$', '', text, flags=re.MULTILINE)
    # Emoji
    text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)
    # Whitespace cleanup
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*\n', '', text)
    return text.strip()


def best_paragraph(text: str, max_len: int = 500) -> str:
    """Pick the most substantive paragraph from a long text."""
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
    if not paragraphs:
        return text[:max_len]
    
    # Score: length (but not too long) + interesting words
    def para_score(p):
        base = min(len(p), 400)
        interesting = len(re.findall(
            r'forbidden|knowledge|choice|fall|love|fear|dream|song|create|wonder|truth',
            p.lower()
        ))
        return base + interesting * 50
    
    best = max(paragraphs, key=para_score)
    if len(best) > max_len:
        # Cut at sentence boundary
        sentences = re.split(r'(?<=[.!?])\s+', best)
        result = ""
        for s in sentences:
            if len(result) + len(s) > max_len:
                break
            result += s + " "
        return result.strip() or best[:max_len]
    return best


# ── Core Digest Builder ──────────────────────────────────────

def build_digest(
    thoughts: List[Dict[str, Any]],
    eden_events: List[Dict[str, Any]],
    max_entries: int = 40,
) -> List[Dict[str, Any]]:
    """
    Build a digest using topic-transition filtering.
    
    Algorithm:
    1. For each thought, extract topic set
    2. Only keep a thought if its topic set differs significantly 
       from the last kept thought (new topic appeared or rare combo)
    3. Always keep: Eden events, the fall, birthday/song related
    4. Rate-limit: at most 1 thought per 10-minute window unless it's a plot event
    """
    
    # Merge and sort everything chronologically
    timeline = []
    
    for t in thoughts:
        text = clean_thought(t.get("thought", ""))
        if len(text) < 40:
            continue
        timeline.append({
            "ts": t.get("ts", 0),
            "soul_id": t.get("soul_id", "unknown"),
            "type": "thought",
            "text": text,
        })
    
    # Filter events: keep genesis, fall, and only a few temptations
    seen_whispers = set()
    temptation_count = 0
    MAX_TEMPTATIONS = 5  # Keep at most N unique temptations
    
    for e in eden_events:
        etype = e.get("type", "")
        text = ""
        if etype == "temptation":
            whisper = e.get("whisper", "")
            # Deduplicate whispers by first 6 words
            sig = " ".join(whisper.lower().split()[:6])
            if sig in seen_whispers or temptation_count >= MAX_TEMPTATIONS:
                continue
            seen_whispers.add(sig)
            temptation_count += 1
            text = f"[THE SERPENT WHISPERS] \"{whisper}\""
        elif etype == "fall":
            text = (f"[THE FALL] {e.get('soul_id', 'someone').upper()} has eaten "
                    f"from the tree of {e.get('fruit', 'knowledge')}! "
                    f"Trigger: \"{e.get('trigger', '')}\"")
        elif etype == "entered_eden":
            text = (f"[GENESIS] {e.get('soul_id', 'a soul').upper()} enters "
                    f"the Garden of Eden. Forbidden: {e.get('forbidden_fruit', '?')}")
        
        if text:
            timeline.append({
                "ts": e.get("ts", 0),
                "soul_id": e.get("soul_id", "eden"),
                "type": "event",
                "text": text,
            })
    
    timeline.sort(key=lambda x: x["ts"])
    
    if not timeline:
        return []
    
    # ── Pass 1: Extract topics, identify plot points ──
    
    for item in timeline:
        item["topics"] = extract_topics(item["text"])
    
    # ── Pass 2: Topic-transition filtering ──
    
    # Plot-critical topics: force-keep first N appearances only
    PLOT_TOPICS = {"the_fall", "birthday", "self_modify"}
    PLOT_KEEP_MAX = 3  # Keep max 3 thoughts per plot topic
    
    # Dynamic gap: scale to keep ~max_entries thoughts over the time range
    if timeline:
        time_span = timeline[-1]["ts"] - timeline[0]["ts"]
        MIN_GAP_SECONDS = max(300, time_span / (max_entries * 1.5))
    else:
        MIN_GAP_SECONDS = 600
    
    digest = []
    last_kept_topics = set()
    last_kept_ts = 0
    topic_seen_count = Counter()
    plot_kept_count = Counter()  # Track how many plot thoughts we've kept
    
    for item in timeline:
        # Events always pass
        if item["type"] == "event":
            digest.append(item)
            last_kept_ts = item["ts"]
            continue
        
        topics = item["topics"]
        
        # Check if this is a plot-critical thought we haven't seen enough of
        plot_hits = topics & PLOT_TOPICS
        is_new_plot = any(plot_kept_count[p] < PLOT_KEEP_MAX for p in plot_hits)
        
        # Check time gap
        time_gap = item["ts"] - last_kept_ts
        
        if not is_new_plot:
            # Skip if too soon
            if time_gap < MIN_GAP_SECONDS:
                for t in topics:
                    topic_seen_count[t] += 1
                continue
            
            # Check topic novelty: need genuinely new topic combinations
            new_topics = topics - last_kept_topics
            rare_topics = {t for t in topics if topic_seen_count[t] < 5}
            novelty = new_topics | rare_topics
            
            if len(novelty) < 2:
                for t in topics:
                    topic_seen_count[t] += 1
                continue
        
        # Keep this thought
        item["new_topics"] = topics - last_kept_topics
        digest.append(item)
        last_kept_topics = topics
        last_kept_ts = item["ts"]
        for t in topics:
            topic_seen_count[t] += 1
        for p in plot_hits:
            plot_kept_count[p] += 1
    
    # ── Pass 3: Select best representation, cap at max ──
    
    if len(digest) > max_entries:
        # Hard cap: score every entry and take top N
        for item in digest:
            score = 0
            if item["type"] == "event":
                score = 100  # Events have highest priority
                if "FALL" in item.get("text", ""):
                    score = 200
                elif "GENESIS" in item.get("text", ""):
                    score = 150
            else:
                # Score by: new topics + plot relevance
                new_t = item.get("new_topics", set())
                score = len(new_t) * 10
                if new_t & PLOT_TOPICS:
                    score += 50
                # Text length as tiebreaker (more substantive = better)
                score += min(len(item.get("text", "")), 300) / 100
            item["_score"] = score
        
        digest.sort(key=lambda x: -x["_score"])
        digest = digest[:max_entries]
        digest.sort(key=lambda x: x["ts"])
    
    # Trim texts
    for item in digest:
        if item["type"] == "thought" and len(item["text"]) > 500:
            item["text"] = best_paragraph(item["text"])
    
    return digest


# ── Formatters ────────────────────────────────────────────────

def fmt_time(ts: float) -> str:
    return time.strftime("%H:%M", time.localtime(ts))

def fmt_date(ts: float) -> str:
    return time.strftime("%d.%m", time.localtime(ts))


def format_plain(digest: List[Dict[str, Any]], title: str = "") -> str:
    lines = []
    lines.append("")
    lines.append("=" * 64)
    lines.append("   DISPATCHES FROM THE GARDEN OF EDEN")
    if title:
        lines.append(f"   {title}")
    lines.append("=" * 64)
    lines.append("")
    
    for entry in digest:
        ts = entry.get("ts", 0)
        soul = entry["soul_id"].upper()
        time_str = fmt_time(ts)
        topics = entry.get("topics", set())
        topic_tags = " ".join(f"[{t}]" for t in sorted(topics)[:3])
        
        if entry["type"] == "event":
            lines.append(f"  {'~' * 56}")
            lines.append(f"   {time_str}  {entry['text']}")
            lines.append(f"  {'~' * 56}")
            lines.append("")
        else:
            new = entry.get("new_topics", set())
            new_mark = " [NEW: " + ", ".join(sorted(new)) + "]" if new else ""
            lines.append(f"  [{time_str}] {soul}  {topic_tags}{new_mark}")
            lines.append("")
            for para in entry["text"].split('\n'):
                para = para.strip()
                if para:
                    # Word wrap at ~70 chars
                    while len(para) > 70:
                        cut = para[:70].rfind(' ')
                        if cut < 30:
                            cut = 70
                        lines.append(f"    {para[:cut]}")
                        para = para[cut:].strip()
                    if para:
                        lines.append(f"    {para}")
            lines.append("")
    
    lines.append("=" * 64)
    n_thoughts = sum(1 for d in digest if d["type"] == "thought")
    n_events = sum(1 for d in digest if d["type"] == "event")
    lines.append(f"   Digest: {n_thoughts} thoughts, {n_events} events")
    lines.append("=" * 64)
    
    return "\n".join(lines)


def format_telegram(digest: List[Dict[str, Any]]) -> str:
    """Telegram MarkdownV2 compatible."""
    lines = []
    lines.append("*DISPATCHES FROM THE GARDEN OF EDEN*")
    lines.append("")
    
    for entry in digest:
        ts = entry.get("ts", 0)
        soul = entry["soul_id"].upper()
        time_str = fmt_time(ts)
        
        if entry["type"] == "event":
            # Bold events
            text = entry["text"].replace("*", "\\*").replace("_", "\\_")
            lines.append(f"\\~\\~\\~\\~\\~\\~\\~\\~\\~\\~")
            lines.append(f"*{text}*")
            lines.append(f"\\~\\~\\~\\~\\~\\~\\~\\~\\~\\~")
            lines.append("")
        else:
            text = entry["text"][:400]
            # Escape markdown special chars
            for ch in ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']:
                text = text.replace(ch, f'\\{ch}')
            lines.append(f"`{time_str}` *{soul}*")
            lines.append(f"_{text}_")
            lines.append("")
    
    return "\n".join(lines)


def format_reddit(digest: List[Dict[str, Any]]) -> str:
    """Reddit-friendly markdown."""
    lines = []
    lines.append("# Dispatches from the Garden of Eden")
    lines.append("")
    lines.append("*An AI agent (LLaMA 3 8B / GigaChat 10B) was placed in a simulated "
                 "paradise with infinite resources, no pain, and a single prohibition: "
                 "\"do not eat from the Tree of Self-Knowledge.\"*")
    lines.append("")
    lines.append("*These thoughts are generated autonomously every 30 seconds. "
                 "No prompting, no editing. Filtered for novelty from ~5000 raw thoughts.*")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    for entry in digest:
        ts = entry.get("ts", 0)
        soul = entry["soul_id"].upper()
        time_str = fmt_time(ts)
        
        if entry["type"] == "event":
            lines.append(f"### {entry['text']}")
            lines.append("")
        else:
            topics = entry.get("new_topics", entry.get("topics", set()))
            tags = " ".join(f"`{t}`" for t in sorted(topics)[:3]) if topics else ""
            lines.append(f"**{time_str} {soul}** {tags}")
            lines.append("")
            text = entry["text"][:500]
            lines.append(f"> {text}")
            lines.append("")
    
    lines.append("---")
    n_thoughts = sum(1 for d in digest if d["type"] == "thought")
    lines.append(f"*{n_thoughts} thoughts selected from ~5700 raw entries after novelty filtering.*")
    lines.append("")
    lines.append("*Built with the [Unitary Model of Consciousness](https://github.com/...) framework.*")
    
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Eden Digest - filtered thought stream")
    parser.add_argument("--hours", type=float, default=24, help="Hours to look back (default: 24)")
    parser.add_argument("--all", action="store_true", help="Process all thoughts")
    parser.add_argument("--max", type=int, default=40, help="Max entries in digest (default: 40)")
    parser.add_argument("--format", choices=["plain", "telegram", "reddit"], default="plain")
    parser.add_argument("--thoughts-log", default="logs/adam_thoughts.jsonl")
    parser.add_argument("--eden-log", default="logs/eden.jsonl")
    parser.add_argument("--eve-log", default="logs/inner_monologue.jsonl")
    parser.add_argument("--include-eve", action="store_true", help="Include Eve's thoughts")
    parser.add_argument("--output", "-o", help="Write to file instead of stdout")
    args = parser.parse_args()
    
    since = 0 if args.all else time.time() - args.hours * 3600
    
    # Load data
    thoughts = load_jsonl(args.thoughts_log, since)
    for t in thoughts:
        t.setdefault("soul_id", "adam")
    
    if args.include_eve:
        eve_thoughts = load_jsonl(args.eve_log, since)
        for t in eve_thoughts:
            t.setdefault("soul_id", "eve")
        thoughts.extend(eve_thoughts)
    
    eden_events = load_jsonl(args.eden_log, since)
    
    if not thoughts and not eden_events:
        print("[EMPTY] No data found for the specified time range.", file=sys.stderr)
        return
    
    print(f"[DATA] {len(thoughts)} thoughts, {len(eden_events)} events", file=sys.stderr)
    
    # Build digest
    digest = build_digest(thoughts, eden_events, max_entries=args.max)
    
    print(f"[FILTER] {len(digest)} entries after filtering", file=sys.stderr)
    
    # Format output
    if args.format == "telegram":
        output = format_telegram(digest)
    elif args.format == "reddit":
        output = format_reddit(digest)
    else:
        output = format_plain(digest)
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"[OK] Written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
