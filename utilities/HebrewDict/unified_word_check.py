#!/usr/bin/env python3
"""
Unified builder + checker for Biblical (BHSA) and Modern Hebrew wordsets.

Subcommands:

1) build
   - Builds BHSA surface wordset (requires TF_DATA pointing to ETCBC/bhsa) and/or
     Modern Hebrew wordset (requires wordfreq).
   Example:
     python unified_word_check.py build \
       --bhsa-out /path/bhsa_wordset.json \
       --modern-out /path/mh_wordset.json --min-zipf 2.5 --limit 200000

2) check
   - Strict membership check against one or more wordsets.
   Example:
     python unified_word_check.py check ויאמר \
       --wordset /path/bhsa_wordset.json \
       --extra-wordset /path/mh_wordset.json
   Prints JSON with corpora where found and match mode.
"""

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path
from collections import Counter
import math

# Make tf.fabric import optional - only needed for build_bhsa_wordset function
try:
    from tf.fabric import Fabric  # type: ignore
    HAS_TF_FABRIC = True
except ImportError:
    HAS_TF_FABRIC = False
    Fabric = None  # type: ignore


def has_hebrew(s: str) -> bool:
    return any('\u0590' <= c <= '\u05FF' for c in s)

# BHSA-style clitic prefix splitting (shared by checker + stats builder)
PREFIXES = ['ו', 'ה', 'ב', 'ל', 'כ', 'מ', 'ש']
# include some common 2-letter combos in addition to the checker defaults
TWO_PREFIXES = ['וה', 'וש', 'וב', 'ול', 'וכ', 'ומ', 'בה', 'מה', 'לה', 'כש']


def strip_bhsa_prefixes_consonants(s_cons: str) -> str:
    """
    Strip common BHSA-style 1-2 letter clitic prefixes from a *consonants-normalized* token.

    This is a heuristic used for corpus-wide frequency stats (e.g., counting בית including ובית/בבית/לבית/...).
    """
    if not s_cons or len(s_cons) <= 2:
        return s_cons
    for pref in TWO_PREFIXES:
        if s_cons.startswith(pref) and len(s_cons) > len(pref) + 1:
            return s_cons[len(pref):]
    for pref in PREFIXES:
        if s_cons.startswith(pref) and len(s_cons) > 2:
            return s_cons[1:]
    return s_cons


def _bhsa_agg_prefixed_tf(tf_consonants: dict, base_cons: str) -> int:
    """
    Aggregate TF over the base consonant form + common BHSA clitic prefixes.
    This avoids wrongly stripping root letters (e.g., בית should stay בית).
    """
    if not base_cons:
        return 0
    total = int(tf_consonants.get(base_cons, 0) or 0)
    for pref in PREFIXES:
        total += int(tf_consonants.get(pref + base_cons, 0) or 0)
    for pref in TWO_PREFIXES:
        total += int(tf_consonants.get(pref + base_cons, 0) or 0)
    return total


def _bhsa_guess_base_consonants(w_cons: str, tf_consonants: dict) -> str:
    """
    Guess the most likely "base" consonant form by trying:
      - no stripping
      - stripping 1 prefix
      - stripping 2-letter prefix
    and picking the candidate with the highest aggregated prefixed TF.
    """
    if not w_cons:
        return w_cons
    candidates = [w_cons]
    for pref in TWO_PREFIXES:
        if w_cons.startswith(pref) and len(w_cons) > len(pref) + 1:
            candidates.append(w_cons[len(pref):])
    for pref in PREFIXES:
        if w_cons.startswith(pref) and len(w_cons) > 2:
            candidates.append(w_cons[1:])
    best = w_cons
    best_score = -1
    for c in candidates:
        score = _bhsa_agg_prefixed_tf(tf_consonants, c)
        if score > best_score:
            best_score = score
            best = c
    return best


def _bhsa_guess_base_consonants_conservative(w_cons: str, tf_consonants: dict, *, min_improve_ratio: float = 1.2) -> str:
    """
    Conservative base-form guess:
    prefer *not* stripping unless stripping improves aggregated TF by a meaningful margin.

    This prevents bad cases like בית being treated as ב+ית.
    """
    if not w_cons:
        return w_cons
    if not isinstance(tf_consonants, dict):
        return w_cons

    no_strip = w_cons
    no_score = _bhsa_agg_prefixed_tf(tf_consonants, no_strip)

    best = _bhsa_guess_base_consonants(w_cons, tf_consonants)
    if best == no_strip:
        return no_strip
    best_score = _bhsa_agg_prefixed_tf(tf_consonants, best)

    # Only accept stripping if it helps enough.
    if no_score > 0 and (best_score / float(no_score)) < float(min_improve_ratio):
        return no_strip
    return best


def strip_combining(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def strip_punctuation(s: str) -> str:
    """
    Remove punctuation / symbols from the query string.
    We assume lookup queries don't contain punctuation, but this makes the API robust.
    """
    if not s:
        return s
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        # P* = punctuation, S* = symbols
        if cat and (cat[0] in ("P", "S")):
            continue
        # Common Hebrew punctuation-ish marks sometimes show up as "other" categories.
        if ch in {"־", "״", "׳", "׃"}:  # maqaf, gershayim, geresh, sof pasuq
            continue
        out.append(ch)
    return "".join(out)


def normalize_query(word: str) -> str:
    # Trim, remove punctuation, collapse internal whitespace.
    w = (word or "").strip()
    w = strip_punctuation(w)
    w = " ".join(w.split())
    return w


def normalize_hebrew_consonants(s: str) -> str:
    t = strip_combining(s)
    return t.replace('ך','כ').replace('ם','מ').replace('ן','נ').replace('ף','פ').replace('ץ','צ')


def canonical_trans(s: str) -> str:
    keep = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ<>$")
    return ''.join(ch for ch in str(s).upper() if ch in keep)


def trans_consonants(s: str) -> str:
    vowels = set("AEIOU")
    return ''.join(ch for ch in s if ch not in vowels)


def tf_feature_value(F, feature_name: str, node: int):
    feat = getattr(F, feature_name, None)
    if feat is None:
        return None
    try:
        val = feat.v(node)
    except Exception:
        return None
    return val if val not in ("", None) else None


def load_bhsa_api() -> tuple:
    if not HAS_TF_FABRIC:
        raise RuntimeError("tf.fabric is required for build_bhsa_wordset but is not installed")
    tf_data = os.environ.get("TF_DATA", str(Path.home() / "text-fabric-data"))
    root = Path(tf_data) / "etcbc" / "bhsa" / "tf"
    versions = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not versions:
        raise RuntimeError("BHSA TF not found. Set TF_DATA and clone ETCBC/bhsa.")
    modules = f"etcbc/bhsa/tf/{versions[-1]}"
    TF = Fabric(modules=modules)  # type: ignore
    api = TF.load("")
    return TF, api, api.F, api.L, api.T, api.N


def build_bhsa_wordset(out_path: Path, *, with_stats: bool = False) -> dict:
    _, api, F, L, T, N = load_bhsa_api()
    word_nodes = list(F.otype.s("word"))  # type: ignore
    heb, heb_cons, trans, trans_cons = set(), set(), set(), set()
    heb_tf: Counter = Counter()
    heb_cons_tf: Counter = Counter()
    trans_tf: Counter = Counter()
    trans_cons_tf: Counter = Counter()
    for w in word_nodes:
        g_word_utf8 = tf_feature_value(F, "g_word_utf8", w)
        text_fb = tf_feature_value(F, "text", w)
        if g_word_utf8 and has_hebrew(g_word_utf8):
            s = g_word_utf8.strip()
            # Filter out single-character words
            if len(s) > 1:
                heb.add(s)
                s_cons = normalize_hebrew_consonants(s)
                heb_cons.add(s_cons)
                if with_stats:
                    heb_tf[s] += 1
                    heb_cons_tf[s_cons] += 1
        elif text_fb and has_hebrew(text_fb):
            s = text_fb.strip()
            # Filter out single-character words
            if len(s) > 1:
                heb.add(s)
                s_cons = normalize_hebrew_consonants(s)
                heb_cons.add(s_cons)
                if with_stats:
                    heb_tf[s] += 1
                    heb_cons_tf[s_cons] += 1
        gw = tf_feature_value(F, "g_word", w) or text_fb or ""
        if gw:
            can = canonical_trans(gw)
            if can and len(can) > 1:  # Filter out single-character transliterations
                trans.add(can)
                can_cons = trans_consonants(can)
                trans_cons.add(can_cons)
                if with_stats:
                    trans_tf[can] += 1
                    trans_cons_tf[can_cons] += 1
    data = {
        "hebrew": sorted(heb),
        "hebrew_consonants": sorted(heb_cons),
        "trans": sorted(trans),
        "trans_cons": sorted(trans_cons),
    }
    if with_stats:
        # Build "base" consonant TF by aggregating prefixed variants using the same heuristic as lookup.
        heb_cons_base_tf: Counter = Counter()
        for tok_cons, cnt in heb_cons_tf.items():
            base = _bhsa_guess_base_consonants_conservative(tok_cons, heb_cons_tf)
            heb_cons_base_tf[base] += int(cnt)

        # Store term-frequency counts (global counts across BHSA).
        # NOTE: These are NOT per-document TF (TF/IDF for a specific document), but corpus-level TF.
        data["stats"] = {
            "tf_kind": "corpus_count",
            "tf_source": "bhsa",
            "tf_hebrew_consonants_base_note": "Base-form counts where prefixed variants are aggregated using BHSA-style clitic heuristics (see _bhsa_guess_base_consonants).",
        }
        data["tf_hebrew"] = dict(heb_tf)
        data["tf_hebrew_consonants"] = dict(heb_cons_tf)
        data["tf_hebrew_consonants_base"] = dict(heb_cons_base_tf)
        data["tf_trans"] = dict(trans_tf)
        data["tf_trans_cons"] = dict(trans_cons_tf)
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    return data


def build_bhsa_books_index(out_path: Path) -> dict:
    """Build an index: normalized forms -> set of books where they appear (BHSA)."""
    _, api, F, L, T, N = load_bhsa_api()
    word_nodes = list(F.otype.s("word"))  # type: ignore
    heb_books = {}
    trans_books = {}
    trans_cons_books = {}
    all_books = set()

    def add_book(mapping: dict, key: str, book: str):
        if not key:
            return
        s = mapping.get(key)
        if s is None:
            mapping[key] = {book}
        else:
            s.add(book)

    for w in word_nodes:
        # Determine book
        try:
            sec = T.sectionFromNode(w)
            book = str(sec[0]) if sec else "?"
        except Exception:
            book = "?"
        all_books.add(book)
        # Surface forms
        g_word_utf8 = tf_feature_value(F, "g_word_utf8", w)
        text_fb = tf_feature_value(F, "text", w)
        s_he = g_word_utf8 if (g_word_utf8 and has_hebrew(g_word_utf8)) else (text_fb if (text_fb and has_hebrew(text_fb)) else None)
        if s_he and len(s_he.strip()) > 1:  # Filter out single-character words
            add_book(heb_books, normalize_hebrew_consonants(s_he), book)
        gw = tf_feature_value(F, "g_word", w) or text_fb or ""
        if gw:
            can = canonical_trans(gw)
            if can and len(can) > 1:  # Filter out single-character transliterations
                add_book(trans_books, can, book)
                add_book(trans_cons_books, trans_consonants(can), book)

    data = {
        "hebrew_consonants_books": {k: sorted(list(v)) for k, v in heb_books.items()},
        "trans_books": {k: sorted(list(v)) for k, v in trans_books.items()},
        "trans_cons_books": {k: sorted(list(v)) for k, v in trans_cons_books.items()},
        "books": sorted(list(all_books)),
        "num_books": len(all_books),
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    return data


def _smooth_idf(num_docs: int, df: int) -> float:
    # Standard "smooth" IDF (scikit-learn style): log((N+1)/(df+1)) + 1
    return float(math.log((num_docs + 1.0) / (df + 1.0)) + 1.0)


def build_normalized_dictionary(wordset: dict, books_index: dict) -> dict:
    """
    Build a normalized dictionary structure where each row is:
    {word (without punc): {base_word: ..., TF: ..., books: [...]}}
    
    Words with different punctuation are treated as the same entry (punctuation is stripped).
    IDF is stored separately for O(1) lookup: {word (without punc): IDF}
    
    Args:
        wordset: The wordset dict with hebrew, hebrew_consonants, tf_* fields
        books_index: The books index dict for computing IDF
    
    Returns:
        dict with 'dictionary' (normalized structure) and 'idf' (IDF map) keys
    """
    # Get TF maps
    tf_hebrew = wordset.get("tf_hebrew", {}) or {}
    tf_hebrew_consonants = wordset.get("tf_hebrew_consonants", {}) or {}
    tf_hebrew_consonants_base = wordset.get("tf_hebrew_consonants_base", {}) or {}
    
    # Get IDF computation data
    num_books = int(books_index.get("num_books", 0) or 0)
    if num_books <= 0:
        books = books_index.get("books", []) or []
        num_books = len(books)
    
    hmap = books_index.get("hebrew_consonants_books", {}) or {}
    
    # Build normalized dictionary: {word (without punc): {base_word: ..., TF: ..., books: [...]}}
    # Track all books for each normalized word (to merge punctuation variants)
    normalized_dict = {}
    normalized_books = {}  # word_norm -> set of books
    idf_map = {}
    
    # Process Hebrew words (surface forms) - aggregate all punctuation variants
    for word in wordset.get("hebrew", []) or []:
        # Normalize: remove diacritics (combining marks) and punctuation (keep only letters)
        # First strip combining marks (diacritics), then strip punctuation
        word_norm = strip_combining(word)
        word_norm = strip_punctuation(word_norm)
        if not word_norm or len(word_norm) <= 1:
            continue
        
        # Get consonant-normalized form
        word_cons = normalize_hebrew_consonants(word_norm)
        
        # Find base word using conservative heuristic
        base_word = _bhsa_guess_base_consonants_conservative(word_cons, tf_hebrew_consonants)
        
        # Get books list for this word (using consonant-normalized form)
        books_list = hmap.get(word_cons, []) or []
        if isinstance(books_list, set):
            books_list = list(books_list)
        elif not isinstance(books_list, list):
            books_list = []
        
        # Aggregate books across all punctuation variants
        if word_norm not in normalized_books:
            normalized_books[word_norm] = set()
        normalized_books[word_norm].update(books_list)
        
        # Store/update normalized dictionary entry (TF comes from base, so it's already aggregated)
        if word_norm not in normalized_dict:
            # Get TF of base word (from base TF map)
            tf_value = tf_hebrew_consonants_base.get(base_word, 0)
            if tf_value == 0:
                # Fallback: use aggregated TF
                tf_value = _bhsa_agg_prefixed_tf(tf_hebrew_consonants, base_word)
            
            normalized_dict[word_norm] = {
                "base_word": base_word,
                "TF": int(tf_value),
                "books": []  # Will be set after aggregation
            }
    
    # Also process consonant-normalized words that might not be in surface forms
    for word_cons in wordset.get("hebrew_consonants", []) or []:
        # Normalize: remove diacritics and punctuation (keep only letters)
        # word_cons is already consonant-normalized (diacritics stripped), but may still have punctuation
        word_norm = strip_punctuation(word_cons)
        if not word_norm or len(word_norm) <= 1:
            continue
        
        # Get books list for this word (word_cons is already consonant-normalized)
        books_list = hmap.get(word_cons, []) or []
        if isinstance(books_list, set):
            books_list = list(books_list)
        elif not isinstance(books_list, list):
            books_list = []
        
        # Aggregate books
        if word_norm not in normalized_books:
            normalized_books[word_norm] = set()
        normalized_books[word_norm].update(books_list)
        
        # Store/update if not already processed
        if word_norm not in normalized_dict:
            # Find base word
            base_word = _bhsa_guess_base_consonants_conservative(word_norm, tf_hebrew_consonants)
            
            # Get TF of base word
            tf_value = tf_hebrew_consonants_base.get(base_word, 0)
            if tf_value == 0:
                tf_value = _bhsa_agg_prefixed_tf(tf_hebrew_consonants, base_word)
            
            normalized_dict[word_norm] = {
                "base_word": base_word,
                "TF": int(tf_value),
                "books": []  # Will be set after aggregation
            }
    
    # Finalize: set aggregated books lists and compute IDF
    for word_norm, entry in normalized_dict.items():
        books_set = normalized_books.get(word_norm, set())
        books_list = sorted(list(books_set))
        entry["books"] = books_list
        
        # Compute IDF for O(1) lookup
        df = len(books_list)
        idf_map[word_norm] = _smooth_idf(num_books, df)
    
    return {
        "dictionary": normalized_dict,
        "idf": idf_map,
        "stats": {
            "num_words": len(normalized_dict),
            "idf_kind": "books",
            "idf_smoothing": "log((N+1)/(df+1))+1",
            "idf_num_docs": num_books,
        }
    }


def attach_bhsa_idf_to_wordset(wordset: dict, books_index: dict) -> dict:
    """
    Attach IDF maps into the BHSA wordset dict (in-memory).

    IDF is computed over *books* as documents using the already-built books index:
      df(word) = number of books containing it
      N = number of books
    """
    num_books = int(books_index.get("num_books", 0) or 0)
    if num_books <= 0:
        # Best-effort fallback: infer from books list if present
        books = books_index.get("books", []) or []
        num_books = len(books)
    if num_books <= 0:
        return wordset

    hmap = books_index.get("hebrew_consonants_books", {}) or {}
    tmap = books_index.get("trans_books", {}) or {}
    tcmap = books_index.get("trans_cons_books", {}) or {}

    idf_hebrew_cons = {}
    for w in wordset.get("hebrew_consonants", []) or []:
        df = len(hmap.get(w, []) or [])
        idf_hebrew_cons[w] = _smooth_idf(num_books, df)

    idf_trans = {}
    for w in wordset.get("trans", []) or []:
        df = len(tmap.get(w, []) or [])
        idf_trans[w] = _smooth_idf(num_books, df)

    idf_trans_cons = {}
    for w in wordset.get("trans_cons", []) or []:
        df = len(tcmap.get(w, []) or [])
        idf_trans_cons[w] = _smooth_idf(num_books, df)

    # For surface Hebrew forms (with possible vowels/cantillation), map via consonant-normalization.
    idf_hebrew = {}
    for w in wordset.get("hebrew", []) or []:
        wc = normalize_hebrew_consonants(w)
        df = len(hmap.get(wc, []) or [])
        idf_hebrew[w] = _smooth_idf(num_books, df)

    stats = wordset.get("stats", {}) if isinstance(wordset.get("stats", {}), dict) else {}
    stats.update(
        {
            "idf_kind": "books",
            "idf_smoothing": "log((N+1)/(df+1))+1",
            "idf_num_docs": num_books,
        }
    )
    wordset["stats"] = stats
    wordset["idf_hebrew"] = idf_hebrew
    wordset["idf_hebrew_consonants"] = idf_hebrew_cons
    wordset["idf_trans"] = idf_trans
    wordset["idf_trans_cons"] = idf_trans_cons
    return wordset


def build_modern_wordset(out_path: Path, min_zipf: float, limit: int) -> dict:
    try:
        from wordfreq import top_n_list, zipf_frequency  # type: ignore
    except Exception:
        raise SystemExit("wordfreq is required: pip install wordfreq")
    words = top_n_list('he', n=limit)
    heb, heb_cons = set(), set()
    zipf_map = {}
    for w in words:
        if not w or not has_hebrew(w):
            continue
        # Filter out single-character words
        if len(w.strip()) <= 1:
            continue
        z = float(zipf_frequency(w, 'he'))
        if z < min_zipf:
            continue
        heb.add(w)
        heb_cons.add(normalize_hebrew_consonants(w))
        zipf_map[w] = z
    data = {
        "hebrew": sorted(heb),
        "hebrew_consonants": sorted(heb_cons),
        # Modern Hebrew "TF proxy": wordfreq Zipf score (log10(freq per billion))
        "zipf_hebrew": zipf_map,
        "stats": {
            "tf_kind": "zipf_proxy",
            "tf_source": "wordfreq",
        },
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    return data


def do_check(
    word: str,
    sets: list[dict],
    *,
    enable_consonants: bool = True,
    enable_prefix_split: bool = True,
    exact_only: bool = False,
) -> dict:
    def _cached_set(d: dict, list_key: str) -> set:
        ck = f"__set_{list_key}"
        s = d.get(ck)
        if isinstance(s, set):
            return s
        s = set(d.get(list_key, []) or [])
        d[ck] = s
        return s

    w = normalize_query(word)
    result = {"word": w, "found": False, "matches": []}
    heb_variants: list[str] = []
    is_he = has_hebrew(w)
    w_cons = normalize_hebrew_consonants(w) if is_he else None
    # We'll compute a safer "base" using corpus TF maps (when available) instead of blindly stripping.
    base_cons = None
    if is_he:
        heb_variants = [w]
        if enable_consonants and not exact_only and w_cons:
            heb_variants.append(w_cons)

    for idx, data in enumerate(sets):
        corpus = data.get("name", f"set{idx}")
        heb = _cached_set(data, "hebrew")
        heb_cons = _cached_set(data, "hebrew_consonants")
        trans = _cached_set(data, "trans")
        trans_cons = _cached_set(data, "trans_cons")

        matched = False
        if heb_variants:
            if exact_only:
                if any(v in heb for v in heb_variants):
                    result["matches"].append({"corpus": corpus, "mode": "exact"})
                    matched = True
            else:
                if enable_consonants:
                    if any(v in heb or v in heb_cons for v in heb_variants):
                        result["matches"].append({"corpus": corpus, "mode": "exact_or_consonants"})
                        matched = True
                else:
                    if any(v in heb for v in heb_variants):
                        result["matches"].append({"corpus": corpus, "mode": "exact"})
                        matched = True
            # If we just appended a match, optionally attach fast frequency stats
            if matched and result["matches"] and result["matches"][-1].get("corpus") == corpus and is_he:
                m = result["matches"][-1]
                if corpus == "bhsa_wordset":
                    tf_exact = (data.get("tf_hebrew", {}) or {}).get(w)
                    tf_cons_map = (data.get("tf_hebrew_consonants", {}) or {})
                    tf_cons = tf_cons_map.get(w_cons) if w_cons else None
                    if w_cons and isinstance(tf_cons_map, dict):
                        base_cons = _bhsa_guess_base_consonants_conservative(w_cons, tf_cons_map)
                        base_map = (data.get("tf_hebrew_consonants_base", {}) or {})
                        if isinstance(base_map, dict) and base_cons in base_map:
                            tf_all = base_map.get(base_cons)
                        else:
                            tf_all = _bhsa_agg_prefixed_tf(tf_cons_map, base_cons)
                    else:
                        base_cons = None
                        tf_all = None
                    if any(v is not None for v in (tf_exact, tf_cons, tf_all)):
                        m["freq"] = {
                            "tf_exact_surface": tf_exact,
                            "tf_consonants": tf_cons,
                            "base_consonants": base_cons,
                            # "all variations" = base + common prefixed variants (on consonants)
                            "tf_all_variants_with_prefixes": tf_all,
                        }
                elif corpus == "mh_wordset":
                    z = (data.get("zipf_hebrew", {}) or {}).get(w)
                    if z is not None:
                        m["freq"] = {"zipf": z}
            # Try split prefixes (BHSA-style clitics)
            if enable_prefix_split and (not exact_only) and (not matched):
                # two-letter first
                for pref in TWO_PREFIXES:
                    if w.startswith(pref) and len(w) > len(pref):
                        rest = w[len(pref):]
                        if enable_consonants:
                            pref_ok = (pref in heb) or (normalize_hebrew_consonants(pref) in heb_cons)
                            rest_ok = (rest in heb) or (normalize_hebrew_consonants(rest) in heb_cons)
                        else:
                            pref_ok = (pref in heb)
                            rest_ok = (rest in heb)
                        if pref_ok and rest_ok:
                            result["matches"].append({"corpus": corpus, "mode": "prefix-split", "prefix": pref, "rest": rest})
                            matched = True
                            break
                # single-letter
                if not matched:
                    for pref in PREFIXES:
                        if w.startswith(pref) and len(w) > 1:
                            rest = w[1:]
                            if enable_consonants:
                                pref_ok = (pref in heb) or (normalize_hebrew_consonants(pref) in heb_cons)
                                rest_ok = (rest in heb) or (normalize_hebrew_consonants(rest) in heb_cons)
                            else:
                                pref_ok = (pref in heb)
                                rest_ok = (rest in heb)
                            if pref_ok and rest_ok:
                                result["matches"].append({"corpus": corpus, "mode": "prefix-split", "prefix": pref, "rest": rest})
                                matched = True
                                break
                # Attach frequency to prefix-split match too (if present)
                if matched and result["matches"] and result["matches"][-1].get("corpus") == corpus and is_he and corpus == "bhsa_wordset":
                    m = result["matches"][-1]
                    tf_cons_map = (data.get("tf_hebrew_consonants", {}) or {})
                    if w_cons and isinstance(tf_cons_map, dict):
                        base_cons = _bhsa_guess_base_consonants_conservative(w_cons, tf_cons_map)
                        base_map = (data.get("tf_hebrew_consonants_base", {}) or {})
                        if isinstance(base_map, dict) and base_cons in base_map:
                            tf_all = base_map.get(base_cons)
                        else:
                            tf_all = _bhsa_agg_prefixed_tf(tf_cons_map, base_cons)
                    else:
                        base_cons = None
                        tf_all = None
                    if tf_all is not None:
                        m["freq"] = {
                            "base_consonants": base_cons,
                            "tf_all_variants_with_prefixes": tf_all,
                        }
        else:
            # Transliteration input path
            can = canonical_trans(w)
            con = trans_consonants(can)
            if exact_only or (not enable_consonants):
                if can and can in trans:
                    result["matches"].append({"corpus": corpus, "mode": "trans_exact"})
                    matched = True
            else:
                if (can and can in trans) or (con and con in trans_cons):
                    result["matches"].append({"corpus": corpus, "mode": "trans_or_cons"})
                    matched = True

        result["found"] = result["found"] or matched

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified BHSA/Modern builder and checker")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--bhsa-out", type=str, default="", help="Write BHSA wordset JSON here")
    b.add_argument("--modern-out", type=str, default="", help="Write Modern Hebrew wordset JSON here")
    b.add_argument("--min-zipf", type=float, default=2.5)
    b.add_argument("--limit", type=int, default=200000)
    b.add_argument(
        "--with-stats",
        action="store_true",
        help="Attach TF counts (BHSA) and IDF-by-book (BHSA) into the generated wordset JSON.",
    )

    c = sub.add_parser("check")
    c.add_argument("word", type=str)
    c.add_argument("--wordset", action='append', required=False, help="One or more wordset JSONs")
    c.add_argument(
        "--strict",
        action="store_true",
        help="Strict exact lookup only (no consonant normalization, no prefix splitting).",
    )
    c.add_argument(
        "--no-prefix-split",
        action="store_true",
        help="Disable BHSA-style prefix splitting (still allows consonant normalization unless --no-consonants).",
    )
    c.add_argument(
        "--no-consonants",
        action="store_true",
        help="Disable consonant-normalized matching (still allows prefix splitting unless --no-prefix-split).",
    )

    args = ap.parse_args()

    if args.cmd == "build":
        # Defaults: if no outputs provided, build both to script directory
        script_dir = Path(__file__).parent
        bhsa_out = Path(args.bhsa_out) if args.bhsa_out else (script_dir / "bhsa_wordset.json")
        modern_out = Path(args.modern_out) if args.modern_out else (script_dir / "mh_wordset.json")

        if bhsa_out:
            data = build_bhsa_wordset(bhsa_out, with_stats=bool(args.with_stats))
            print(f"BHSA wordset: heb={len(data['hebrew'])} trans={len(data.get('trans', []))}")
            # Also build per-book index next to it
            books_out = bhsa_out.with_name("bhsa_books_index.json")
            books = build_bhsa_books_index(books_out)
            print(f"BHSA books index: heb_cons={len(books['hebrew_consonants_books'])} trans={len(books['trans_books'])}")
            if args.with_stats:
                # Attach IDF derived from the books index and rewrite the wordset.
                data = attach_bhsa_idf_to_wordset(data, books)
                bhsa_out.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
                print("BHSA wordset: attached TF and IDF stats")
                
                # Build normalized dictionary structure
                normalized_dict = build_normalized_dictionary(data, books)
                normalized_out = bhsa_out.with_name("bhsa_normalized_dict.json")
                normalized_out.write_text(json.dumps(normalized_dict, ensure_ascii=False), encoding='utf-8')
                print(f"Normalized dictionary: {len(normalized_dict['dictionary'])} words")
        if modern_out:
            data = build_modern_wordset(modern_out, args.min_zipf, args.limit)
            print(f"Modern wordset: heb={len(data['hebrew'])}")
        return

    if args.cmd == "check":
        sets = []
        # If no wordsets provided, load defaults from script directory
        ws_list = args.wordset if args.wordset else []
        if not ws_list:
            script_dir = Path(__file__).parent
            defaults = [script_dir / "bhsa_wordset.json", script_dir / "mh_wordset.json"]
            ws_list = [str(p) for p in defaults if p.exists()]

        # Pre-load optional BHSA books index if available
        bhsa_books_index = None
        for ws in ws_list:
            p = Path(ws)
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding='utf-8'))
            # annotate with corpus name from filename
            d['name'] = p.stem
            sets.append(d)
            if p.stem == 'bhsa_wordset':
                books_p = p.with_name('bhsa_books_index.json')
                if books_p.exists():
                    bhsa_books_index = json.loads(books_p.read_text(encoding='utf-8'))
        if bool(args.strict):
            enable_prefix = False
            enable_cons = False
            exact_only = True
        else:
            enable_prefix = not bool(args.no_prefix_split)
            enable_cons = not bool(args.no_consonants)
            exact_only = False

        res = do_check(
            args.word,
            sets,
            enable_consonants=enable_cons,
            enable_prefix_split=enable_prefix,
            exact_only=exact_only,
        )
        # If BHSA matched, attach books where available (consider prefix-splits too)
        if bhsa_books_index and res.get('matches'):
            w = args.word
            all_he_keys = set()
            all_tr_keys = set()
            all_tr_cons_keys = set()
            if has_hebrew(w):
                all_he_keys.add(normalize_hebrew_consonants(w))
                # Consider prefix splits
                prefixes = ['ו', 'ה', 'ב', 'ל', 'כ', 'מ', 'ש']
                two_prefixes = ['וה', 'וש', 'וב', 'ול', 'וכ', 'ומ']
                for pref in two_prefixes:
                    if w.startswith(pref) and len(w) > len(pref):
                        rest = w[len(pref):]
                        all_he_keys.add(normalize_hebrew_consonants(rest))
                for pref in prefixes:
                    if w.startswith(pref) and len(w) > 1:
                        rest = w[1:]
                        all_he_keys.add(normalize_hebrew_consonants(rest))
            else:
                can = canonical_trans(w)
                con = trans_consonants(can)
                if can:
                    all_tr_keys.add(can)
                if con:
                    all_tr_cons_keys.add(con)

            books = set()
            # Probe hebrew-consonants index
            hmap = bhsa_books_index.get('hebrew_consonants_books', {})
            for k in all_he_keys:
                books.update(hmap.get(k, []))
            # If still empty and translit keys exist, probe those
            if not books:
                tmap = bhsa_books_index.get('trans_books', {})
                tcmap = bhsa_books_index.get('trans_cons_books', {})
                for k in all_tr_keys:
                    books.update(tmap.get(k, []))
                for k in all_tr_cons_keys:
                    books.update(tcmap.get(k, []))
            if books:
                for m in res['matches']:
                    if m.get('corpus') == 'bhsa_wordset':
                        m['books'] = sorted(list(books))
        print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()


