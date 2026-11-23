"""ResumeBench scorer v5: canonical comparison so that presentation-only differences neither need label
corrections nor cost a system points, without crediting genuinely wrong answers.

Both the label and the system output are turned into a canonical view; the released strict scorer then scores the
two views, with these internal changes:
  * text equality: exact after normalisation, or equal ignoring case when one side is ALL CAPS (styling)
  * lines (bullets, summary/prose lines, custom-section lines) compared as a bag; the same text broken into lines
    differently counts as equal (prose) or as min(lines) (bullet arrays)
  * city/state/country compared as a multiset of parts (slot-free)
  * education degree compared as level + degree type + field: an equal degree earns all its units however it is split
    over studyType/area; one field holding the whole degree is right only when the whole degree is equal; levels must
    agree; the type is a wildcard when one side omits it
  * list markers (glyphs, symbols, emoji, dashes, numbering) ignored at the start of line fields, and of the lines of a
    multi-line text value when at least two lines carry one
  * the composed name compared as a multiset of tokens (which part holds a token, and their order, are presentation)
  * entities matched by identity without years; ties between equal identities broken by years and bullet overlap
  * flags count only when true
Every rule can be switched off (RULES) so its effect can be measured on its own.
"""

from __future__ import annotations

import copy
import re
import threading
import unicodedata
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

import resume_bench.grading.strict as S
from resume_bench.schema.sections import SECTIONS

ALL_RULES = (
    "caps_tolerant",  # equal ignoring case when one side is ALL CAPS (rendered capitals)
    "markers",  # list markers stripped from line fields (bullets, custom lines, skills) and from multi-line text when >=2 lines carry one
    "edge_punct",  # trailing . , ; : and leading , ; : ignored (a leading '.' before a letter is kept: .NET)
    "degree",  # education degree = level + type + field, abbreviation/possessive/plural/accents folded
    "degree_identity",  # education matched on institution + degree level (level wildcard for pairing when empty or 'other')
    "lone_date",  # a single date sits in the end slot on both sides; start==end collapses to end
    "flags",  # currently*/inProgress/isRemote/hasPersonalPhoto: only true counts
    "location_set",  # city/state/country compared as a multiset of parts
    "language_parse",  # "Hindi (Fluent)" -> name + level; level scored only when the label has one
    "lines_bag",  # bag-of-lines comparison with re-wrapping tolerance
    "prose_merge",  # prose written as the first bullet line(s) is moved to roleDescription/summary when, joined, it equals the other side's prose
    "url",  # URLs compared href-to-href on host+path (host case-insensitive)
    "no_url_units",  # URL fields left out of the score
    "skill_split",  # skills split on every ',' and '|' (as the released scorer splits on ','); de-duplicated canonically
    "gpa_label",  # 'GPA'/'CGPA'/'Grade' label words around a score ignored
    "identity_nodates",  # entries matched without years (years still scored as fields; they break ties)
    "name_order",  # the composed name compared as a multiset of tokens (given/family order is presentation)
)
RULES = set(ALL_RULES)

_ORIG = {
    "normalize": S.normalize,
    "_split_comma_skills": S._split_comma_skills,
    "_score_record": S._score_record,
    "_field_passes": S._field_passes,
    "_str_match": S._str_match,
    "_direct_match": S._direct_match,
    "_identity_pass_rate": S._identity_pass_rate,
    "_rb_cell_mismatch_count": S._rb_cell_mismatch_count,
    "IDENTITY_FIELDS": copy.deepcopy(S.IDENTITY_FIELDS),
    "SECTION_SCHEMA_FIELDS": copy.deepcopy(S.SECTION_SCHEMA_FIELDS),
}
GLYPH_MARKS = "•▪●◦‣∙·➢►▸❖✔✓➤■»›⮚→➔○★‣⁃▪●"
SYMBOL_MARKS = "\u2190-\u21ff\u2300-\u23ff\u25a0-\u25ff\u2600-\u27bf\u2b00-\u2bff\ue000-\uf8ff\U0001f300-\U0001faff"  # arrows, shapes, dingbats, private-use bullets, emoji
MARK_RE = re.compile(
    r"^[ \t]*(?:[" + GLYPH_MARKS + SYMBOL_MARKS + r"]\uFE0F?[ \t]*|[-*–—‐‑‒―−][ \t]+|\(?\d{1,2}[.)][ \t]+(?=[^\d\s]))"
)
URL_FIELDS = {"linkedinUrl", "url", "companyUrl"}
FLAG_FIELDS = {
    "currentlyWorkHere",
    "currentlyStudyHere",
    "currentlyVolunteerHere",
    "inProgress",
    "isRemote",
    "hasPersonalPhoto",
}
LOC = ("city", "state", "country")
CASE_SENSITIVE_HOSTS = ()  # paths case-sensitive except on these hosts
CI_HOSTS = (
    "linkedin.com",
    "github.com",
    "gitlab.com",
    "twitter.com",
    "x.com",
    "medium.com",
    "behance.net",
    "dribbble.com",
)


# ------------------------------------------------------------------ text
def norm_text(s) -> str:
    """Normalise a value for comparison (case is kept; see eq())."""
    if s is None:
        return ""
    t = str(s).replace("″", '"').replace("‶", '"')
    t = unicodedata.normalize("NFKC", t)
    t = t.translate(S._GLYPHS).translate(S._LIGATURES)
    t = re.sub("[​‌‍⁠﻿]", "", t)
    t = t.replace("**", "").replace("__", "")
    t = re.sub(r"\s+", " ", t).strip()
    if "edge_punct" in RULES:
        t = re.sub(r"[\s.,;:]+$", "", t)
        t = re.sub(r"^[\s,;:]+", "", t)
        t = re.sub(r"^\.(?![A-Za-z])\s*", "", t)
    return t


def _tr_lower(s: str) -> str:
    """Turkish lower case: capital I becomes dotless i, capital dotted I becomes i."""
    return s.replace("I", "ı").replace("İ", "i").lower()


def same_ignoring_case(a: str, b: str) -> bool:
    """Equal ignoring case, also under Turkish casing (capital I and dotless i, capital dotted I and i);
    a different letter (i against dotless i in two lower-case spellings) is not a case difference."""
    if a.casefold() == b.casefold() or _tr_lower(a) == _tr_lower(b):
        return True
    # text in capitals may drop the Turkish dots ('ISTANBUL' for 'İstanbul'): fold i/ı/İ/I only then
    return (is_caps(a) or is_caps(b)) and _cf(a) == _cf(b)


def _cf(s: str) -> str:
    """casefold, with Turkish dotted/dotless i folded to i ('YILMAZ' equals the Turkish lower-case spelling)."""
    return s.casefold().replace("i\u0307", "i").replace("ı", "i")


def is_caps(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def eq(a: str, b: str) -> bool:
    if a == b:
        return True
    if "casefold_all" in RULES:
        return same_ignoring_case(a, b)
    return bool("caps_tolerant" in RULES and (is_caps(a) or is_caps(b)) and same_ignoring_case(a, b))


def str_match_v5(a, b):
    return 1.0 if eq(norm_text(a), norm_text(b)) else 0.0


def strip_marker(line: str) -> str:
    if "markers" not in RULES:
        return line
    return MARK_RE.sub("", line, count=1)


def strip_line_markers(s):
    """Multi-line text value: when at least two lines start with a list marker, ignore the markers."""
    if "markers" not in RULES or not isinstance(s, str) or "\n" not in s:
        return s
    ls = s.split("\n")
    if sum(1 for ln in ls if MARK_RE.match(ln)) >= 2:
        ls = [MARK_RE.sub("", ln, count=1) for ln in ls]
    return "\n".join(ls)


# ------------------------------------------------------------------ urls
def canon_url(v):
    if isinstance(v, dict):
        v = v.get("href") or ""
    t = str(v or "").strip()
    if not t or " " in t:
        return t
    t = re.sub(r"^[A-Za-z][A-Za-z0-9+.-]*://", "", t)
    t = re.sub(r"^www\.", "", t, flags=re.I)
    host, _, path = t.partition("/")
    host = host.casefold()
    if any(host == h or host.endswith("." + h) for h in CI_HOSTS) or (host and "." not in host):
        path = path.casefold()  # known case-insensitive hosts, and handles / partial paths such as 'in/Jane-Doe'
    return (host + ("/" + path if path else "")).rstrip("/")


def url_label(v):
    return (v.get("label") if isinstance(v, dict) else None) or ""


# ------------------------------------------------------------------ field comparison
GPA_RE = re.compile(
    r"^\s*(?:c?gpa|cpi|sgpa|grade|percentage|marks|score)\s*[:\-–]?\s*|\s*[\-–(]?\s*(?:c?gpa|cpi|sgpa)\s*\)?\s*$", re.I
)


def field_passes_v5(fname, gv, pv):
    if fname in URL_FIELDS and "url" in RULES:
        gh, ph = canon_url(gv), canon_url(pv)
        if gh or ph:
            return bool(gh) and gh == ph
        return eq(norm_text(url_label(gv)), norm_text(url_label(pv)))
    if fname in FLAG_FIELDS and "flags" in RULES:
        return bool(gv) == bool(pv)
    if fname == "studyType" and "degree" in RULES and isinstance(gv, str) and gv.startswith("DEG|"):
        return degree_equal(gv, pv)
    if fname == "score" and "gpa_label" in RULES:
        return eq(norm_text(GPA_RE.sub("", str(gv or ""))), norm_text(GPA_RE.sub("", str(pv or ""))))
    if isinstance(gv, bool) or isinstance(pv, bool):
        return bool(gv) == bool(pv)
    if isinstance(gv, (int, float)) and not isinstance(gv, bool):
        return gv == pv
    if isinstance(gv, dict) and isinstance(pv, dict):
        keys = set(gv) | set(pv)
        return all(field_passes_v5(fname, gv.get(k), pv.get(k)) for k in keys)
    gv, pv = strip_line_markers(gv), strip_line_markers(pv)
    gs, ps = norm_text(gv if gv is not None else ""), norm_text(pv if pv is not None else "")
    if fname in S.CONTACT_FIELDS or fname in S.LOCATION_FIELDS or fname == "network":
        return same_ignoring_case(gs, ps)
    return eq(gs, ps)


def rb_mismatch_v5(gt_ent, pred_ent, s_fields):
    """Hungarian cost for keyword sections using v5 equality."""
    mism = tot = 0
    for f in s_fields:
        gv, pv = gt_ent.get(f), pred_ent.get(f)
        ge, pe = S._is_empty(gv), S._is_empty(pv)
        tot += 1
        if ge and pe:
            continue
        if ge or pe or not field_passes_v5(f, gv, pv):
            mism += 1
    return mism, tot


# ------------------------------------------------------------------ lines
def _bag_match(gb, pb):
    """Matched count between two lists of normalised lines: exact multiset, then caps-tolerant pairs."""
    cg, cp = Counter(gb), Counter(pb)
    inter = cg & cp
    c = sum(inter.values())
    rg = list((cg - inter).elements())
    rp = list((cp - inter).elements())
    used = set()
    for a in rg:
        for j, b in enumerate(rp):
            if j not in used and eq(a, b):
                used.add(j)
                c += 1
                break
    return c


def _breaks(lines):
    """Line breaks as word positions in the joined text (punctuation trimmed per line cannot shift them)."""
    return {len(_joined(lines[:k]).split()) for k in range(1, len(lines))}


def _breaks_nested(a, b):
    """True when one side's line breaks are a subset of the other's (a real merge or split, not a moved break)."""
    ba, bb = _breaks([x for x in a if norm_text(x)]), _breaks([x for x in b if norm_text(x)])
    return ba <= bb or bb <= ba


def _joined(lines):
    return norm_text(" ".join(lines))


PROSE_FIELDS = ("roleDescription", "summary")


def _route_prose(g, p, pf):
    """If one side wrote the prose as bullet line(s) and the other in the prose field, move it across."""
    g, p = dict(g), dict(p)
    for a, b in ((g, p), (p, g)):
        aR, bR = norm_text(a.get(pf)), norm_text(b.get(pf))
        if aR and not bR and b is g and len(S._bullets(g)) > 1:
            # the output wrote the label's bullet lines as one paragraph: score it as one merged line
            gl = [strip_marker(str(x)) for x in S._bullets(g)]
            if any(eq(_joined(gl[:k]), aR) for k in range(2, len(gl) + 1)):
                p["description"] = [p.get(pf)] + [str(x) for x in S._bullets(p)]
                p[pf] = None
                p.pop("descriptions", None)
                continue
        if aR and not bR:
            bb = [str(x) for x in S._bullets(b)]
            ab = {norm_text(strip_marker(str(x))) for x in S._bullets(a)}
            if any(norm_text(strip_marker(x)) in ab for x in bb[:1]):
                continue  # the other side also lists these lines as bullets: not a routing difference
            for k in range(1, len(bb) + 1):
                if eq(_joined([strip_marker(x) for x in bb[:k]]), aR):
                    b[pf] = " ".join(strip_marker(x) for x in bb[:k])
                    b["description"] = bb[k:]
                    b.pop("descriptions", None)
                    break
            else:
                for k in range(len(bb)):
                    if eq(norm_text(strip_marker(bb[k])), aR):
                        b[pf] = strip_marker(bb[k])
                        b["description"] = bb[:k] + bb[k + 1 :]
                        b.pop("descriptions", None)
                        break
    return g, p


def _degree_units(g, p):
    gu, pu = g.get("_deg_units", 0), p.get("_deg_units", 0)
    gd, pd = g.get("studyType"), p.get("studyType")
    if not gu and not pu:
        return 0, 0, 0
    if not (gu and pu and isinstance(gd, str) and isinstance(pd, str)):
        return 0, gu, pu
    if degree_equal(gd, pd):
        return gu, gu, gu  # same degree, whatever the split between studyType and area
    _, gl, gt, gf = gd.split("|", 3)
    _, pl, pt, pf_ = pd.split("|", 3)
    type_ok = (not gl or not pl or gl == pl) and (not gt or not pt or set(gt.split()) == set(pt.split()))
    field_ok = bool(gf.split()) and gf.split() == pf_.split()
    if gu > 1 and pu > 1:
        c = int(type_ok) + int(field_ok)
    elif gu == 1 and pu == 1:
        c = 0  # one unit holding the degree is right only when the whole degree is
    else:  # one side split over studyType+area, the other wrote one field
        single, split = (gd, p) if gu == 1 else (pd, g)
        single_field = (gf if gu == 1 else pf_).split()
        if any(isinstance(x, str) and degree_equal(single, x) for x in split.get("_deg_parts") or ()):
            c = 1  # the one field equals one of the other side's two fields (as the released scorer credits)
        else:
            c = (
                int(type_ok) if not single_field else 0
            )  # a type-only unit is credited for the type; a whole-degree unit must be equal
    return min(c, gu, pu), gu, pu


CUR = ("currentlyWorkHere", "currentlyStudyHere", "currentlyVolunteerHere", "inProgress")


def _is_lone(e):
    if any(e.get(k) for k in CUR):
        return False
    sy, ey = e.get("startYear"), e.get("endYear")
    if sy and ey:
        return sy == ey and (e.get("startMonth") or None) == (e.get("endMonth") or None)
    return bool(sy) != bool(ey)


def _to_end(e):
    if e.get("startYear") and not e.get("endYear"):
        e["endYear"], e["endMonth"] = e.get("startYear"), e.get("startMonth")
    e["startYear"] = None
    e["startMonth"] = None
    return e


def score_record_v5(g, p, fields):
    c = e = q = 0
    loc_fields = [f for f in fields if f in LOC] if "location_set" in RULES else []
    pf = next((f for f in fields if f in PROSE_FIELDS), None)
    if pf and "prose_merge" in RULES:
        g, p = _route_prose(g, p, pf)
    if "lone_date" in RULES and ("startYear" in fields or "endYear" in fields) and _is_lone(g):
        g = _to_end(dict(g))
        if _is_lone(p):
            p = _to_end(dict(p))
    deg = "_deg_units" in g or "_deg_units" in p
    if deg:
        dc, de, dq = _degree_units(g, p)
        c += dc
        e += de
        q += dq
    for f in fields:
        if f in loc_fields or (deg and f in ("studyType", "area")):
            continue
        gv, pv = g.get(f), p.get(f)
        ge, pe = S._empty(gv), S._empty(pv)
        if fname_flag(f) and "flags" in RULES:
            ge, pe = not gv, not pv
        if f in ("companyUrl", "url") and "no_url_units" in RULES:
            continue
        if ge and pe:
            continue
        if not ge:
            e += 1
        if not pe:
            q += 1
        if not ge and not pe and S._field_passes(f, gv, pv):
            c += 1
    if loc_fields:
        G = [norm_text(g.get(f)) for f in loc_fields if norm_text(g.get(f))]
        P = [norm_text(p.get(f)) for f in loc_fields if norm_text(p.get(f))]
        left = list(P)
        for x in G:  # multiset match, ignoring case
            k = next((i for i, y in enumerate(left) if same_ignoring_case(x, y)), None)
            if k is not None:
                left.pop(k)
                c += 1
        e += len(G)
        q += len(P)
    sm = (lambda x: x) if (g.get("_prose") or p.get("_prose")) else strip_marker
    graw = [sm(str(b)) for b in S._bullets(g) if str(b).strip()]
    praw = [sm(str(b)) for b in S._bullets(p) if str(b).strip()]
    gb = [norm_text(b) for b in graw]
    pb = [norm_text(b) for b in praw]
    gb = [b for b in gb if b]
    pb = [b for b in pb if b]
    if gb or pb:
        if "lines_bag" in RULES:
            m = _bag_match(gb, pb)
            if (
                gb
                and pb
                and m < min(len(gb), len(pb))
                and eq(_joined(graw), _joined(praw))
                and (g.get("_prose") or _breaks_nested(graw, praw))
            ):
                m = len(gb) if g.get("_prose") else min(len(gb), len(pb))
                if g.get("_prose"):
                    pb = gb  # same prose, other line breaks: full credit
            c += m
            e += len(gb)
            q += len(pb)
        else:
            for i in range(max(len(gb), len(pb))):
                a = gb[i] if i < len(gb) else ""
                b = pb[i] if i < len(pb) else ""
                if a:
                    e += 1
                if b:
                    q += 1
                if a and b and eq(a, b):
                    c += 1
    return c, e, q


def fname_flag(f):
    return f in FLAG_FIELDS


def fname_url(f):
    return f in URL_FIELDS


def cnt_v5(ent, sf):
    n = ent.get("_deg_units", 0) if "_deg_units" in ent else 0
    for f in sf:
        if "_deg_units" in ent and f in ("studyType", "area"):
            continue
        v = ent.get(f)
        if f in FLAG_FIELDS and "flags" in RULES:
            n += 1 if v else 0
            continue
        if f in ("companyUrl", "url") and "no_url_units" in RULES:
            continue
        if not S._empty(v):
            n += 1
    for dk in ("description", "descriptions"):
        d = ent.get(dk, []) or []
        if isinstance(d, list):
            n += sum(1 for b in d if norm_text(strip_marker(str(b))))
    return n


# ------------------------------------------------------------------ identity
WILDCARD_ID = {"degreeLevel"}


def identity_pass_rate_v5(g, p, section_name):
    id_fields = S.IDENTITY_FIELDS.get(section_name, set())
    if not id_fields:
        return 1.0, 0
    ok = tot = 0
    for f in id_fields:
        gv, pv = g.get(f), p.get(f)
        ge, pe = S._empty(gv), S._empty(pv)
        if ge and pe:
            continue
        if f in WILDCARD_ID and (ge or pe or gv == "other" or pv == "other"):
            continue
        tot += 1
        if not ge and not pe and S._field_passes(f, gv, pv):
            ok += 1
    if tot == 0:
        return 1.0, 0
    return ok / tot, tot


def _agreement(g, p):
    s = 0.0
    gd, pd = g.get("studyType"), p.get("studyType")
    if isinstance(gd, str) and gd.startswith("DEG|") and isinstance(pd, str) and pd.startswith("DEG|"):
        if degree_equal(gd, pd):
            s += 6.0  # the same degree beats years and bullets when two entries share an institution
        elif gd.split("|")[1] == pd.split("|")[1]:
            s += 3.0  # the same level
    for a, b in (("startYear", "startYear"), ("endYear", "endYear")):
        if g.get(a) and g.get(a) == p.get(b):
            s += 1
    gb = {norm_text(x) for x in S._bullets(g)}
    pb = {norm_text(x) for x in S._bullets(p)}
    if gb and pb:
        s += len(gb & pb) / len(gb | pb)
    return s


def direct_match_v5(gi, pi, section_name, gate):
    if not gi or not pi:
        return [], list(range(len(gi))), list(range(len(pi)))
    n, m = len(gi), len(pi)
    W = np.full((n, m), -1e9)
    for i, ge in enumerate(gi):
        for j, pe in enumerate(pi):
            rate, _ = S._identity_pass_rate(ge, pe, section_name)
            if rate >= gate:
                W[i, j] = 10.0 + rate + _agreement(ge, pe)
    r, c = linear_sum_assignment(-W)
    matched = [(int(a), int(b)) for a, b in zip(r, c, strict=False) if W[a, b] > -1e8]
    mg = {a for a, _ in matched}
    mp = {b for _, b in matched}
    return matched, [i for i in range(n) if i not in mg], [j for j in range(m) if j not in mp]


# ------------------------------------------------------------------ skills
def split_skills_v5(skills):
    out = []
    for s in skills:
        if not isinstance(s, str):
            continue
        parts, _depth, cur = [], 0, ""
        for ch in s:
            if ch in ",|":
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        parts.append(cur)
        for p_ in parts:
            p_ = norm_text(strip_marker(p_.strip()))
            if p_:
                out.append(p_)
    seen, res = set(), []
    for x in out:
        k = x.casefold()
        if k not in seen:
            seen.add(k)
            res.append(x)
    return res


# ------------------------------------------------------------------ install
def install():
    S.normalize = norm_text
    S._str_match = str_match_v5
    S._field_passes = field_passes_v5
    S._rb_cell_mismatch_count = rb_mismatch_v5
    S._identity_pass_rate = identity_pass_rate_v5
    S._cnt = cnt_v5
    S._split_comma_skills = split_skills_v5 if "skill_split" in RULES else _ORIG["_split_comma_skills"]
    S._score_record = score_record_v5
    S._direct_match = direct_match_v5 if "identity_nodates" in RULES else _ORIG["_direct_match"]
    S.IDENTITY_FIELDS.clear()
    S.IDENTITY_FIELDS.update(copy.deepcopy(_ORIG["IDENTITY_FIELDS"]))
    if "identity_nodates" in RULES:
        S.IDENTITY_FIELDS["experience"] = {"company", "position"}
        S.IDENTITY_FIELDS["volunteering"] = {"organization", "position"}
        S.IDENTITY_FIELDS["projects"] = {"name"}
    if "degree_identity" in RULES or "degree" in RULES:
        S.IDENTITY_FIELDS["education"] = (
            {"institution", "degreeLevel"} if "degree_identity" in RULES else {"institution", "startYear", "endYear"}
        )


_CNT0 = S._cnt


def uninstall():
    S.normalize = _ORIG["normalize"]
    S._str_match = _ORIG["_str_match"]
    S._field_passes = _ORIG["_field_passes"]
    S._rb_cell_mismatch_count = _ORIG["_rb_cell_mismatch_count"]
    S._identity_pass_rate = _ORIG["_identity_pass_rate"]
    S._cnt = _CNT0
    S._split_comma_skills = _ORIG["_split_comma_skills"]
    S._score_record = _ORIG["_score_record"]
    S._direct_match = _ORIG["_direct_match"]
    S.IDENTITY_FIELDS.clear()
    S.IDENTITY_FIELDS.update(copy.deepcopy(_ORIG["IDENTITY_FIELDS"]))


# ------------------------------------------------------------------ degrees
ABBR = {
    "bs": "bachelor science",
    "bsc": "bachelor science",
    "ba": "bachelor arts",
    "ab": "bachelor arts",
    "be": "bachelor engineering",
    "beng": "bachelor engineering",
    "btech": "bachelor technology",
    "bba": "bachelor business administration",
    "bcom": "bachelor commerce",
    "bcomm": "bachelor commerce",
    "bca": "bachelor computer applications",
    "llb": "bachelor laws",
    "bfa": "bachelor fine arts",
    "bed": "bachelor education",
    "bpharm": "bachelor pharmacy",
    "bsn": "bachelor science|nursing",
    "barch": "bachelor architecture",
    "bms": "bachelor management studies",
    "mbbs": "bachelor medicine surgery",
    "mbchb": "bachelor medicine surgery",
    "bds": "bachelor dental surgery",
    "bse": "bachelor science|engineering",
    "basc": "bachelor applied science",
    "bcs": "bachelor computer science",
    "bbus": "bachelor business",
    "bsba": "bachelor science|business administration",
    "bdes": "bachelor design",
    "bmus": "bachelor music",
    "bsw": "bachelor social work",
    "bvsc": "bachelor veterinary science",
    "ms": "master science",
    "msc": "master science",
    "ma": "master arts",
    "mba": "master business administration",
    "meng": "master engineering",
    "me": "master engineering",
    "mtech": "master technology",
    "mpa": "master public administration",
    "mph": "master public health",
    "mfa": "master fine arts",
    "med": "master education",
    "llm": "master laws",
    "mca": "master computer applications",
    "mcom": "master commerce",
    "msw": "master social work",
    "mres": "master research",
    "mphil": "master philosophy",
    "mpp": "master public policy",
    "msn": "master science|nursing",
    "mse": "master science|engineering",
    "masc": "master applied science",
    "mcs": "master computer science",
    "mha": "master health administration",
    "mpharm": "master pharmacy",
    "mdes": "master design",
    "msf": "master science|finance",
    "mscs": "master science|computer science",
    "msee": "master science|electrical engineering",
    "msme": "master science|mechanical engineering",
    "msit": "master science|information technology",
    "msds": "master science|data science",
    "bscs": "bachelor science|computer science",
    "bsee": "bachelor science|electrical engineering",
    "bsme": "bachelor science|mechanical engineering",
    "bsce": "bachelor science|civil engineering",
    "bsit": "bachelor science|information technology",
    "bscn": "bachelor science|nursing",
    "eds": "specialist education",
    "phd": "doctor philosophy",
    "dphil": "doctor philosophy",
    "edd": "doctor education",
    "dba": "doctor business administration",
    "jd": "doctor juris",
    "md": "doctor medicine",
    "pharmd": "doctor pharmacy",
    "dnp": "doctor nursing practice",
    "dds": "doctor dental surgery",
    "dmd": "doctor dental medicine",
    "dvm": "doctor veterinary medicine",
    "psyd": "doctor psychology",
    "dsc": "doctor science",
    "drph": "doctor public health",
    "aas": "associate applied science",
    "aa": "associate arts",
    "as": "associate science",
    "hnd": "higher national diploma",
    "hnc": "higher national certificate",
    "pgd": "postgraduate diploma",
    "pgdm": "postgraduate diploma management",
    "pgce": "postgraduate certificate education",
    "hs": "high school",
    "ssc": "secondary school certificate",
    "hsc": "higher secondary certificate",
    "hssc": "higher secondary school certificate",
    "sslc": "secondary school leaving certificate",
    "puc": "preuniversity course",
    "ged": "general educational development",
    "bts": "brevet technicien superieur",
    "dut": "diplome universitaire technologie",
}
KNOWN = set(ABBR) | {"fsc", "llm"}
AMBIG = {"as", "me", "ma", "ms", "md", "be", "med", "bed", "ab", "aa", "hs", "it", "cs", "ee"}
FIELD_ABBR = {
    "me": "mechanical engineering",
    "cse": "computer science engineering",
    "ece": "electronics communication engineering",
    "eee": "electrical electronics engineering",
    "cs": "computer science",
    "it": "information technology",
    "ee": "electrical engineering",
    "ce": "civil engineering",
}
FOLD = {
    "bachelors": "bachelor",
    "masters": "master",
    "doctorate": "doctor",
    "doctoral": "doctor",
    "associates": "associate",
    "honors": "honours",
    "hons": "honours",
    "hon": "honours",
    "graduation": "graduate",
    "postgraduation": "postgraduate",
    "certificat": "certificate",
    "levels": "level",
    "minors": "minor",
    "gcses": "gcse",
    "igcses": "igcse",
    "diplome": "diploma",
    "baccalaureat": "baccalaureate",
    "licenciatura": "licence",
    "license": "licence",
}
STOP = {
    "of",
    "in",
    "and",
    "&",
    "the",
    "degree",
    "with",
    "on",
    "an",
    "for",
    "major",
    "majored",
    "majoring",
    "concentration",
    "specialization",
    "specialisation",
    "emphasis",
    "focus",
    "track",
    "option",
    "current",
    "incoming",
    "ongoing",
    "pursuing",
    "expected",
    "de",
    "en",
    "du",
    "des",
    "la",
    "le",
    "d",
    "l",
    "full",
    "time",
    "part",
}
LEVEL_HEAD = {"bachelor", "master", "doctor", "associate", "specialist"}
TYPE_WORDS = {
    "science",
    "arts",
    "engineering",
    "technology",
    "commerce",
    "business",
    "administration",
    "laws",
    "law",
    "philosophy",
    "medicine",
    "surgery",
    "education",
    "fine",
    "applied",
    "music",
    "design",
    "pharmacy",
    "nursing",
    "architecture",
    "juris",
    "public",
    "health",
    "policy",
    "social",
    "work",
    "research",
    "dental",
    "veterinary",
    "psychology",
    "management",
    "studies",
    "computer",
    "applications",
}


def _fold(s):
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.casefold().translate(S._GLYPHS)


def _dtokens(s):
    s = _fold(s)
    s = re.sub(r"['’]s\b", "", s)
    s = re.sub(r"\b(c?gpa|percentage)\b.*$", " ", s)
    s = re.sub(r"\b(19|20)\d\d\b", " ", s)

    def join(m):
        j = re.sub(r"[.\s]", "", m.group(0))
        if j in KNOWN:
            return j + " "
        dotted = re.match(r"(?:[a-z]{1,3}\.\s?){1,3}", m.group(0))  # 'B. S. in': do not swallow the next word
        if dotted:
            k = re.sub(r"[.\s]", "", dotted.group(0))
            if k in KNOWN:
                return k + " " + m.group(0)[dotted.end() :]
        return m.group(0)

    s = re.sub(r"\b(?:[a-z]{1,3}\.\s?){1,3}[a-z]{0,5}\.?(?=\W|$)", join, s)
    s = re.sub(r"(?<=\w)\.(?=\w)", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return [FOLD.get(t, t) for t in s.split()]


HEADS = {"bachelor", "master", "doctor", "associate", "specialist", "diploma", "certificate"}
SEPS = {",", "in", ":", "|", "-", "(", ")"}


def _seq(s):
    """Tokens with boundary markers kept: ',' for brackets/commas/colons/dashes; 'in' kept as a word."""
    s = _fold(s)
    s = re.sub(r"['’]s\b", "", s)
    s = re.sub(r"\b(c?gpa|percentage)\b.*$", " ", s)
    s = re.sub(r"\b(19|20)\d\d\b", " ", s)

    def join(m):
        j = re.sub(r"[.\s]", "", m.group(0))
        if j in KNOWN:
            return j + " "
        dotted = re.match(r"(?:[a-z]{1,3}\.\s?){1,3}", m.group(0))  # 'B. S. in': do not swallow the next word
        if dotted:
            k = re.sub(r"[.\s]", "", dotted.group(0))
            if k in KNOWN:
                return k + " " + m.group(0)[dotted.end() :]
        return m.group(0)

    s = re.sub(r"\b(?:[a-z]{1,3}\.\s?){1,3}[a-z]{0,5}\.?(?=\W|$)", join, s)
    s = re.sub(r"(?<=\w)\.(?=\w)", "", s)
    s = re.sub(r"[(),:|/]|\s[-–—]\s", " , ", s)
    s = re.sub(r"[^\w\s,]", " ", s)
    return [FOLD.get(t, t) for t in s.split()]


def _parse_study(seq):
    """-> (head words, type words, field words) from one studyType token sequence."""
    head, typ, field = [], [], []
    i = 0
    # leading abbreviation: 'BS', 'M.Tech', 'MBA'
    while i < len(seq) and seq[i] == ",":
        i += 1
    if i < len(seq) and seq[i] in ABBR and seq[i] not in HEADS and not (i + 1 < len(seq) and seq[i + 1] == "level"):
        tpart, _, fpart = ABBR[seq[i]].partition("|")
        exp = tpart.split()
        head, typ = exp[:1], exp[1:]
        field += fpart.split()  # an abbreviation that includes the field ('BSN' = Bachelor of Science in Nursing)
        i += 1
    else:
        while i < len(seq) and seq[i] not in HEADS and seq[i] != ",":
            field.append(seq[i])
            i += 1  # words before the head ('Honours Bachelor', 'Executive Master')
        if i < len(seq) and seq[i] in HEADS:
            head = [seq[i]]
            i += 1
            if i < len(seq) and seq[i] == "of":
                i += 1
                while i < len(seq) and seq[i] not in SEPS:
                    typ.append(seq[i])
                    i += 1
    rest = seq[i:]
    j = 0
    skip = set()
    while j < len(rest):  # '(Doctor of Pharmacy)' restating the degree
        if rest[j] in HEADS and (not head or rest[j] == head[0]):
            k = j + 1
            words = []
            if k < len(rest) and rest[k] == "of":
                k += 1
                while k < len(rest) and rest[k] not in SEPS:
                    words.append(rest[k])
                    k += 1
            if not typ or not words or set(words) <= set(typ):
                if not typ and words:
                    typ = [w for w in words]
                if not head:
                    head = [rest[j]]
                skip.update(range(j, k))
                j = k
                continue
        j += 1
    for j, t in enumerate(rest):
        if j in skip or t in SEPS or t in STOP:
            continue
        if (
            t in ABBR
            and head
            and ABBR[t].partition("|")[0].split()[:1] == head
            and (set(ABBR[t].partition("|")[0].split()[1:]) == set(typ))
        ):
            continue  # '(M.E.)' after 'Master of Engineering' restates the degree
        if t in FIELD_ABBR and t in AMBIG and head:
            field += FIELD_ABBR[t].split()
        elif t in ABBR and (t not in AMBIG or (j > 0 and rest[j - 1] == ",")):
            tpart, _, fpart = ABBR[t].partition("|")
            exp = tpart.split()
            # a restatement of the same degree: same head and the same type ('Bachelor of Science (B.S.)'); with no type
            # yet, the abbreviation may restate the field too ("Master's in Business Administration (MBA)")
            if exp[:1] == head and (set(exp[1:]) == set(typ) if typ else set(exp[1:]) <= set(field) or not exp[1:]):
                if not typ:
                    typ = exp[1:]
                    if Counter(typ) == Counter(field):  # "Master's in Business Administration (MBA)"
                        field = []
                if fpart and not set(fpart.split()) <= set(field):
                    field += fpart.split()
                continue  # '(BS)' restating the degree
            field += exp + fpart.split()
        elif t in FIELD_ABBR:
            field += FIELD_ABBR[t].split()
        else:
            field.append(t)
    # a later degree head ('B.S., MBA', 'BA, Master of ...') is kept as a marker, so a second printed degree counts
    return head, [t for t in typ if t not in STOP], [("+" + t if t in HEADS else t) for t in field if t not in STOP]


def degree_parts(study, area):
    """-> (level, type tokens, field tokens)."""
    h, typ, field = _parse_study(_seq(study))
    area_toks = [FIELD_ABBR.get(t, t) for t in _seq(area) if t not in SEPS and t not in STOP]
    area_toks = [w for t in area_toks for w in t.split()]
    if area_toks:
        if h and h[0] in HEADS:
            # read area as the continuation of studyType's degree (its head and type words): a restated degree is
            # dropped, a second degree ('MBA', 'Certificate in ...') is kept as a marked degree in printed order
            _, atyp, afield = _parse_study([h[0], *(["of", *typ] if typ else []), ",", *_seq(area)])
            typ = typ or atyp
            area_toks = afield
        elif h:  # an abbreviation whose expansion has no degree word in front ('HSC', 'GED', 'PGDM')
            ah, atyp, afield = _parse_study(_seq(area))
            area_toks = afield if (ah and ah == h) else area_toks
        else:
            ah, atyp, afield = _parse_study(_seq(area))
            if ah:  # the whole degree was written in area
                h, typ, area_toks = ah, atyp, afield
        cf, ca = Counter(field + typ), Counter(area_toks)
        if not (ca - cf):
            area_toks = []  # area restates what studyType already says
        elif not (cf - ca) and not typ:
            field = []
        field = field + area_toks
    HEAD_LEVEL = {
        "bachelor": "bachelor",
        "master": "master",
        "doctor": "doctorate",
        "associate": "associate",
        "specialist": "master",
    }
    lvl = HEAD_LEVEL.get(h[0]) if h else None
    if lvl is None or not h:
        lvl = degree_level(study, area)
    return lvl, tuple(typ), _seg_sort(field)  # type words in printed order (see degree_equal)


def _seg_sort(tokens):
    """Field words as a bag per segment: the main field, then each 'minor ...' part in printed order."""
    segs = [[]]
    for t in tokens:
        if t == "minor" or t.startswith("+"):
            segs.append([t])
        else:
            segs[-1].append(t)
    # each segment is a bag; segments keep their printed order, except that consecutive minors may come in any order
    out, run = [], []
    for seg in segs:
        body = sorted(seg[1:]) if seg[:1] and (seg[0] == "minor" or seg[0].startswith("+")) else sorted(seg)
        head = seg[:1] if seg[:1] and (seg[0] == "minor" or seg[0].startswith("+")) else []
        if head == ["minor"]:
            run.append(head + body)
            continue
        out += [w for m in sorted(run) for w in m]
        run = []
        out += head + body
    out += [w for m in sorted(run) for w in m]
    return tuple(out)


def _minus_main(field, words):
    """Remove ``words`` (once each) from the main segment of a field."""
    left = Counter(words)
    out, main = [], True
    for t in field:
        if t == "minor" or t.startswith("+"):
            main = False
        if main and left[t] > 0:
            left[t] -= 1
            continue
        out.append(t)
    return _seg_sort(out)


def canon_degree(study, area):
    lvl, typ, field = degree_parts(study, area)
    if not (lvl or typ or field):
        return None
    return "DEG|" + lvl + "|" + " ".join(typ) + "|" + " ".join(field)


def degree_equal(g, p):
    if not isinstance(p, str) or not p.startswith("DEG|"):
        return False
    _, gl, gt, gf = g.split("|", 3)
    _, pl, pt, pf = p.split("|", 3)
    if gl != pl:
        return False  # a different level, or a degree on one side only (no studyType)
    gts, pts = set(gt.split()), set(pt.split())
    gfa, pfa = tuple(gf.split()), tuple(pf.split())
    if gts and pts and gts != pts:
        # a field written right after the type with no separator ('Bachelor of Science Nursing', 'Bachelor of
        # Commerce-Accounting') is read as more type words after the same type: they count as field words.
        # Words in front of the type ('Bachelor of Applied Science') are a different type.
        gl_, pl_ = gt.split(), pt.split()
        if len(gl_) > len(pl_) and gl_[: len(pl_)] == pl_:
            return _seg_sort(gl_[len(pl_) :] + list(gfa)) == pfa
        if len(pl_) > len(gl_) and pl_[: len(gl_)] == gl_:
            return _seg_sort(pl_[len(gl_) :] + list(pfa)) == gfa
        return False  # 'BS' vs 'BA'
    if gts and not pts:  # the other side omitted the type, or wrote its words into the field
        return pfa == gfa or _minus_main(pfa, gts) == gfa
    if pts and not gts:
        return gfa == pfa or _minus_main(gfa, pts) == pfa
    return gfa == pfa


LEVELS = [
    ("bachelor", r"\b(mbbs|mbchb|bds|llb|bvsc)\b"),
    ("other", r"\b(postdoctoral|postdoc|residency|fellowship|exchange|semester|coursework|course work)\b"),
    ("doctorate", r"\b(doctor|doctorat|phd|dphil|edd|dba|juris|jd|md|pharmd|dnp|dds|dmd|dvm|psyd|dsc|jsd|sjd|drph)\b"),
    (
        "master",
        r"\b(master|mba|emba|msc|ms|ma|me|meng|mse|masc|mcs|mtech|mpa|mph|mfa|med|llm|mca|mcom|msw|mres|mphil|mpp|msn|mha|eds|specialist education|education specialist|diploma d ingenieur|diplom ingenieur|dipl ing|magister|laurea magistrale|m1|m2|first state examination|staatsexamen)\b",
    ),
    (
        "bachelor",
        r"\b(bachelor|bs|bsc|ba|ab|be|beng|bse|basc|bcs|bbus|bsba|btech|bba|bcom|bcomm|bca|bfa|bed|bpharm|bsn|barch|bms|bdes|bmus|bsw|licence|licenciado|laurea)\b",
    ),
    (
        "diploma_certificate",
        r"\b(certified|certification|ccna|ccnp|cisa|cissp|cism|cfe|pmp|itil|istqb|comptia|lead auditor)\b",
    ),
    ("secondary", r"\b(a|o|as) levels?\b|\b(i?gcses?|ib diploma|international baccalaureate)\b"),
    ("associate", r"\b(associate|aas|aa)\b|^as\b"),
    (
        "diploma_certificate",
        r"\b(postsecondary|graduate|postgraduate|pg)\b.*\b(diploma|certificate)\b|\b(pgd\w*|pgce|pgcert|pgdip)\b",
    ),
    (
        "secondary",
        r"\b(high school|secondary|ged|alevels?|olevels?|aslevels?|gcse|igcse|hsc|ssc|hssc|sslc|matric\w*|leaving certificate|baccalaureate|bac|abitur|matura|maturita|highers|hkale|hkdse|hkcee|grade 12|12th|10th|cbse|icse|intermediate|fsc|puc|preuniversity|pre university|regents|kcse|kcpe|primary education|primary school|middle school|junior high|hs|higher school certificate)\b",
    ),
    (
        "diploma_certificate",
        r"\b(diploma|certificate|cert|hnc|hnd|nvq|btec|bts|dut|cap|nitec|apprenticeship|executive programm?e?|executive education|national certificate)\b|^nc\b",
    ),
]


def degree_level(study, area=None):
    for text in (study, f"{study or ''} {area or ''}"):
        t = " ".join(_dtokens(text))
        if not t:
            continue
        for lvl, pat in LEVELS:
            if re.search(pat, t):
                return lvl
    return "other" if str(study or "").strip() else ""


# ------------------------------------------------------------------ languages
LANG_LEVEL_RES = (
    # the name may hold complete bracket groups but no open one, so a dash or colon inside brackets never splits
    re.compile(r"^\s*(?P<name>(?:[^():\[\]]|\([^()]*\)|\[[^\[\]]*\])+?)\s+[-–—]\s+(?P<lv>.+?)\s*$"),  # 'French - C1'
    re.compile(r"^\s*(?P<name>(?:[^():\[\]]|\([^()]*\)|\[[^\[\]]*\])+?)\s*:\s*(?P<lv>.+?)\s*$"),  # 'Spanish: native'
    re.compile(r"^\s*(?P<name>.+?)\s*\((?P<lv>[^()]*)\)\s*$"),  # 'Hindi (Fluent)', 'Chinese (Mandarin) (Native)'
)
PROFICIENCY_RE = re.compile(
    r"\b(native|mother[ -]?tongue|bilingual|fluent|fluency|proficien\w*|advanced|intermediate|upper|basic|beginners?|"
    r"elementary|conversational|working|professional|limited|full|business|good|excellent|moderate|fair|read\w*|writ\w*|"
    r"speak\w*|spoken|oral|[abc][12]|hsk|jlpt|n[1-5]|ielts|toefl|toeic|cet|dele|delf|dalf|goethe|level|\d+|"
    r"first language|mother language|nativ[oae]|natif|native speaker|materna|muttersprache|courant|fließend|flüssig)\b",
    re.I,
)


def split_language(name, level):
    if not name:
        return name, level
    if level:
        return name, level  # a stored level: the brackets are part of the name ('Chinese (Mandarin)')
    for rx in LANG_LEVEL_RES:
        m = rx.match(str(name))
        if m and PROFICIENCY_RE.search(m.group("lv")):
            return m.group("name").strip(), m.group("lv").strip()  # 'Hindi (Fluent)', 'French - C1', 'Spanish: native'
    return name, level


# ------------------------------------------------------------------ canonical view
def _items(x):
    if isinstance(x, dict):
        if "items" in x:
            return x["items"] if isinstance(x["items"], list) else []
        return [x] if x else []
    return x if isinstance(x, list) else []


def _lines(text):
    if not text or not isinstance(text, str):
        return []
    return [ln for ln in re.split(r"\n+", text) if ln.strip()]


def _list(v):
    return [x for x in v if isinstance(x, str)] if isinstance(v, list) else []


def canon_view(doc, is_label):
    d = copy.deepcopy(doc) if isinstance(doc, dict) else {}
    out = {k: v for k, v in d.items() if k != "layout"}
    b = out.get("basics") if isinstance(out.get("basics"), dict) else {}
    if b:
        b = dict(b)
        if "name_order" in RULES:
            toks = [norm_text(t) for k in ("prefix", "fname", "lname", "suffix") for t in str(b.get(k) or "").split()]
            toks = sorted((t for t in toks if t), key=lambda t: (_cf(t), t))
            b["fname"] = " ".join(toks) or None
            b["prefix"] = b["lname"] = b["suffix"] = None
        if "flags" in RULES and not b.get("hasPersonalPhoto"):
            b["hasPersonalPhoto"] = None
        out["basics"] = b
    for sec in (
        "experience",
        "education",
        "projects",
        "volunteering",
        "certifications",
        "awards",
        "publications",
        "languages",
        "interests",
        "profiles",
        "skills",
    ):
        if out.get(sec) is None:
            continue
        items = [dict(e) for e in _items(out.get(sec)) if isinstance(e, dict)]
        for e in items:
            for k in ("description", "descriptions"):
                if k in e:
                    e[k] = _list(e[k])
            if sec == "skills":
                e["skills"] = _list(e.get("skills"))
            if "flags" in RULES:
                for f in FLAG_FIELDS:
                    if f in e and not e.get(f):
                        e[f] = None
            if "no_url_units" in RULES:
                for f in ("companyUrl", "url"):
                    if f in e:
                        e[f] = None
            if sec == "education":
                e["degreeLevel"] = degree_parts(e.get("studyType"), e.get("area"))[0] or None
                if "degree" in RULES:
                    e["_deg_units"] = sum(1 for k in ("studyType", "area") if norm_text(e.get(k)))
                    e["_deg_parts"] = (canon_degree(e.get("studyType"), None), canon_degree(None, e.get("area")))
                    e["studyType"] = canon_degree(e.get("studyType"), e.get("area"))
                    e["area"] = None
            if sec == "languages" and "language_parse" in RULES:
                e["name"], e["level"] = split_language(e.get("name"), e.get("level"))
        out[sec] = {"items": items}
    cs = [dict(c) for c in _items(out.get("customSections")) if isinstance(c, dict)]
    for c in cs:
        for k in ("description", "descriptions"):
            if k in c:
                c[k] = _list(c[k])
    if cs or out.get("customSections") is not None:
        out["customSections"] = cs
    ps = out.get("personalSummary")
    text = ps.get("text") if isinstance(ps, dict) else (ps if isinstance(ps, str) else None)
    if "lines_bag" in RULES:
        ls = _lines(text)
        if "markers" in RULES and sum(1 for ln in ls if MARK_RE.match(ln)) >= 2:
            ls = [MARK_RE.sub("", ln, count=1) for ln in ls]
        out["personalSummary"] = {"text": None, "description": ls, "_prose": True} if text else None
    elif text is not None:
        out["personalSummary"] = {"text": text}
    return out


def _align_basics_locations(lv, pv):
    """Slot-free basics location: move output parts that equal a label part into that label's slot."""
    gb, pb = lv.get("basics") or {}, pv.get("basics") or {}
    if not isinstance(gb, dict) or not isinstance(pb, dict):
        return
    gvals = {f: norm_text(gb.get(f)) for f in LOC}
    pvals = [norm_text(pb.get(f)) for f in LOC if norm_text(pb.get(f))]
    new = {f: None for f in LOC}
    rest = []
    for v in pvals:
        slot = next((f for f in LOC if gvals[f] and same_ignoring_case(gvals[f], v) and new[f] is None), None)
        if slot:
            new[slot] = v
        else:
            rest.append(v)
    for f in LOC:
        if new[f] is None and rest and not gvals[f]:
            new[f] = rest.pop(0)
    for f in LOC:
        if new[f] is None and rest:
            new[f] = rest.pop(0)
    pb.update(new)


def _strip_lang_levels(lv, pv):
    labs = [(norm_text(e.get("name")), e.get("level")) for e in _items(lv.get("languages"))]
    for e in _items(pv.get("languages")):
        name = norm_text(e.get("name"))
        if not any(lev for n, lev in labs if same_ignoring_case(n, name)):
            e["level"] = None


_LOCK = threading.Lock()


def score(label, pred, gate=1.0):
    """Score one prediction against one ground truth with the v5 rules (``RULES``).

    v5 swaps comparison hooks into ``resume_bench.grading.strict`` for the duration of the call (under a lock);
    do not call the strict scorer from another thread while v5 scores.
    """
    with _LOCK:
        return _score(label, pred, gate)


def _score(label, pred, gate=1.0):
    if isinstance(pred, dict) and isinstance(pred.get("personalSummary"), str):
        pred = {**pred, "personalSummary": {"text": pred["personalSummary"]}}
    install()
    try:
        lv = canon_view(label, True)
        pv = canon_view(pred, False)
        if "location_set" in RULES:
            _align_basics_locations(lv, pv)
        if "language_parse" in RULES:
            _strip_lang_levels(lv, pv)
        return S.score_resume_strict(lv, pv, SECTIONS, gate=gate)
    finally:
        uninstall()


def score_resume_v5(ground_truth: dict, prediction: dict, gate: float = 1.0) -> dict:
    """Public entry point: same return shape as ``score_resume_strict`` (f1, precision, recall, sections, ...)."""
    return score(ground_truth, prediction, gate)
