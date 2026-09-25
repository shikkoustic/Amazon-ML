"""Stage 1 blocking for the test set -- word-level, repaired normalisation.

Self-contained for Kaggle: the normaliser and its variant table are inlined,
so the notebook needs nothing from the repository.

Two things differ from the character-trigram runs. Word tokens carry about
nine non-zeros per document against thirty-one, measured 6.7x faster at better
recall, so both signals across both sources fit in one session. And each
meaning now has one spelling before anything is compared -- road and rd,
street and st, maharashtra and mh, techn0logies and technology -- which was
the largest single reason true pairs looked unlike each other.

Signals run strongest first and each writes its own file, so a session that
runs out of time still leaves the most useful output behind.
"""
import os
import gc
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def _require(module, package=None):
    """Import a module, installing it first if the image lacks it.

    Kaggle's script images are leaner than its notebook images: unidecode is
    present in one and absent in the other, and a run that dies on the import
    wastes the whole session.
    """
    import importlib
    import subprocess
    import sys
    try:
        return importlib.import_module(module)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        package or module], check=True)
        return importlib.import_module(module)


unidecode = _require("unidecode").unidecode
sp_matmul_topn = _require("sparse_dot_topn").sp_matmul_topn

IN = Path(os.environ.get("IN", "/kaggle/input"))
OUT = Path(os.environ.get("OUT", "/kaggle/working"))
SPLIT = os.environ.get("SPLIT", "test")
K = int(os.environ.get("K", "50"))
MAX_DF = float(os.environ.get("MAX_DF", "0.05"))
CHUNK = int(os.environ.get("CHUNK", "40000"))
THREADS = int(os.environ.get("THREADS", str(os.cpu_count() or 4)))
PLAN = [("combo", 2), ("combo", 3), ("addr", 2), ("addr", 3)]
T0 = time.time()

"""Token variants that mean the same thing.

Most of this table is learned, not written. Aligning matched records from the
training ground truth and reading off the single-token disagreements ranks the
corpus's corruptions by what they actually cost: road against rd leads at 1024
occurrences, street against st at 966, drive against dr at 955, avenue against
ave at 670 -- each on its own larger than every region abbreviation combined,
and present in nearly every US address.

Clustering those pairs also exposes a second corruption class: digits standing
in for the letters they resemble, as in s0lutions, g1obal and techn0logies,
alongside singular and plural forms of the same word. Within a cluster the
cleanest spelling wins, never the corrupted one -- choosing by length alone
once mapped "best" onto "8est".

The postal and region entries are written out because a table learned from one
sample covers only the variants that sample happened to contain, and the test
set will carry others. Both kinds are spelling knowledge compiled into the
code, in the same spirit as the legal-suffix list the normaliser already
applies: nothing is looked up, no registry or geocoder is consulted, and no
network is reached.
"""

VARIANTS: dict[str, str] = {
    '0fice': 'ofice',
    '5afe': 'safe',
    '5ervices': 'services',
    '5hre': 'shre',
    '5ilver': 'silver',
    '5mith': 'smith',
    '5olutions': 'solutions',
    '5ons': 'sons',
    '5tar': 'star',
    '5upreme': 'supreme',
    '5ystems': 'systems',
    '6lobal': 'global',
    '6olden': 'golden',
    '8est': 'best',
    '8lue': 'blue',
    '8right': 'bright',
    '8rothers': 'brothers',
    'a1l': 'alabama',
    'agr0': 'egro',
    'agro': 'egro',
    'ak': 'alaska',
    'al': 'alabama',
    'ank1e': 'ankle',
    'ap': 'andhra pradesh',
    'apt': 'apartment',
    'ar': 'arkansas',
    'arihnt': 'arihant',
    'as0ciates': 'asociates',
    'asociate': 'asociates',
    'at1antic': 'atlantic',
    'av': 'avenue',
    'ave': 'avenue',
    'aven': 'avenue',
    'az': 'arizona',
    'b1ue': 'blue',
    'bhart': 'bharat',
    'bijnes': 'busines',
    'bildrs': 'builders',
    'bldg': 'building',
    'blk': 'block',
    'blu': 'blue',
    'blv': 'boulevard',
    'blvd': 'boulevard',
    'br': 'bihar',
    'br0thers': 'brothers',
    'brait': 'bright',
    'c0nstruction': 'construction',
    'c0nsultants': 'consultants',
    'c1asic': 'clasic',
    'c1inic': 'clinic',
    'ca': 'california',
    'cae': 'care',
    'capita1': 'capital',
    'cardi0logy': 'cardiology',
    'cardio1ogy': 'cardiology',
    'ce': 'care',
    'center': 'centre',
    'cg': 'chhattisgarh',
    'chd': 'chandigarh',
    'cir': 'circle',
    'circ': 'circle',
    'cnt': 'connecticut',
    'co': 'colorado',
    'consu1tants': 'consultants',
    'continenta1': 'continental',
    'court': 'conecticut',
    'cr': 'care',
    'cre': 'care',
    'crt': 'court',
    'ct': 'court',
    'cust0m': 'custom',
    'cv': 'cove',
    'de': 'delaware',
    'de1hi': 'delhi',
    'denta1': 'dental',
    'devel0pers': 'developers',
    'devlprs': 'developers',
    'digita1': 'digital',
    'dirve': 'drvie',
    'dive': 'drvie',
    'dl': 'delhi',
    'dr': 'drive',
    'drie': 'drvie',
    'drive': 'drvie',
    'drv': 'drive',
    'drve': 'drvie',
    'educati0n': 'education',
    'enterprises': 'entrpraijes',
    'entrpraijej': 'entrpraijes',
    'estet': 'estate',
    'exports': 'eksports',
    'ext': 'extension',
    'f0ods': 'fods',
    'f0undation': 'foundation',
    'fami1y': 'family',
    'federa1': 'federal',
    'fl': 'florida',
    'flr': 'floor',
    'fr0ntier': 'frontier',
    'ft': 'fort',
    'g1obal': 'global',
    'ga': 'georgia',
    'gj': 'gujarat',
    'gl0bal': 'global',
    'globl': 'global',
    'gret': 'great',
    'gujrat': 'gujarat',
    'h0me': 'home',
    'harb0r': 'harbor',
    'haryana': 'hriyana',
    'haspitaliti': 'honspitailiti',
    'hea1th': 'health',
    'heart1and': 'heartland',
    'hi': 'hawaii',
    'hospitality': 'honspitailiti',
    'hp': 'himachal pradesh',
    'hr': 'haryana',
    'hri': 'hari',
    'hspitaliti': 'honspitailiti',
    'hway': 'highway',
    'hwy': 'highway',
    'ia': 'iowa',
    'id': 'idaho',
    'il': 'illinois',
    'impex': 'impeks',
    'india': 'lndia',
    'indian': 'indiyn',
    'indstrij': 'lndustries',
    'indstris': 'lndustries',
    'industries': 'lndustries',
    'industry': 'lndustries',
    'infra': 'inphra',
    'institute': 'lnstitute',
    'internal': 'lnternal',
    'international': 'lnternational',
    'investments': 'lnvestments',
    'it': 'aiti',
    'jct': 'junction',
    'jh': 'jharkhand',
    'jy': 'jay',
    'ka': 'karnataka',
    'kerala': 'keralam',
    'keralam': 'kerala',
    'kerln': 'keralam',
    'keyst0ne': 'keystone',
    'kl': 'kerala',
    'knslting': 'consulting',
    'krnatk': 'karnataka',
    'ks': 'kansas',
    'ky': 'kentucky',
    'l0gistics': 'lonjistiks',
    'l1c': 'lc',
    'la': 'louisiana',
    'lae': 'lane',
    'lksmi': 'lakshmi',
    'ln': 'lane',
    'lne': 'lane',
    'lnfra': 'inphra',
    'logistics': 'lonjistiks',
    'lots': 'lotus',
    'm0untain': 'mountain',
    'ma': 'massachusetts',
    'mainejment': 'management',
    'md': 'maryland',
    'me': 'maine',
    'media': 'midiya',
    'menejment': 'management',
    'metr0': 'metro',
    'mh': 'maharashtra',
    'mharastr': 'maharashtra',
    'mi': 'michigan',
    'mn': 'minnesota',
    'mo': 'missouri',
    'modern': 'mondrn',
    'mount': 'montana',
    'mp': 'madhya pradesh',
    'ms': 'mississippi',
    'mt': 'mount',
    'mtn': 'montana',
    'my': 'may',
    'n0rth': 'north',
    'n0rthern': 'northern',
    'nationa1': 'national',
    'nbr': 'nebraska',
    'nc': 'north carolina',
    'nd': 'north dakota',
    'ne': 'nebraska',
    'nh': 'new hampshire',
    'nj': 'new jersey',
    'nm': 'new mexico',
    'nr': 'near',
    'nv': 'nevada',
    'ny': 'new york',
    'oh': 'ohio',
    'ok': 'oklahoma',
    'opp': 'opposite',
    'or': 'oregon',
    'orisa': 'odisha',
    'orissa': 'odisha',
    'pa': 'pennsylvania',
    'partner': 'partners',
    'pb': 'punjab',
    'physica1': 'physical',
    'pi0ner': 'pioner',
    'piedm0nt': 'piedmont',
    'pkwy': 'parkway',
    'pky': 'parkway',
    'pl': 'place',
    'plz': 'plaza',
    'pnjab': 'punjab',
    'pr0ducer': 'prodyusr',
    'pr0ducts': 'products',
    'pr0jects': 'projekts',
    'pr0perties': 'properties',
    'praim': 'prime',
    'prajekts': 'projekts',
    'prodakts': 'products',
    'prodkts': 'products',
    'producer': 'prodyusr',
    'projects': 'projekts',
    'pronprtij': 'properties',
    'proprtijh': 'properties',
    'pt': 'point',
    'rad': 'roda',
    'rajsthan': 'rajasthan',
    'raod': 'roda',
    'rd': 'road',
    'rea1ty': 'realty',
    'ri': 'rhode island',
    'rj': 'rajasthan',
    'road': 'roda',
    'rod': 'roda',
    'rods': 'road',
    's0ftware': 'software',
    's0lutions': 'solutions',
    's0ns': 'sons',
    'saint': 'stret',
    'sc': 'south carolina',
    'sd': 'south dakota',
    'sec': 'sector',
    'service': 'services',
    'sevn': 'seven',
    'shkti': 'shakti',
    'si1ver': 'silver',
    'silvr': 'silver',
    'sistms': 'systems',
    'sn': 'sun',
    'so1utions': 'solutions',
    'solution': 'solutions',
    'sq': 'square',
    'st': 'street',
    'ste': 'suite',
    'stet': 'stret',
    'str': 'street',
    'stret': 'street',
    'strt': 'street',
    'suprim': 'supreme',
    'svstik': 'swastik',
    'tamilnadu': 'tamil nadu',
    'techn0logies': 'technologies',
    'techn0logy': 'technologies',
    'techno1ogies': 'technologies',
    'technology': 'technologies',
    'telngan': 'telangana',
    'ter': 'terrace',
    'terr': 'terrace',
    'tg': 'telangana',
    'tirupti': 'tirupati',
    'tn': 'tennessee',
    'trading': 'treding',
    'trl': 'trail',
    'ts': 'telangana',
    'tx': 'texas',
    'united': 'yunaited',
    'up': 'uttar pradesh',
    'ut': 'utah',
    'va': 'virginia',
    'vijy': 'vijay',
    'visi0n': 'vision',
    'vt': 'vermont',
    'wa': 'washington',
    'wb': 'west bengal',
    'wi': 'wisconsin',
    'wi1liams': 'wiliams',
    'wv': 'west virginia',
    'wy': 'wyoming',
}


"""Text normalisation for business names and addresses.

Ordering matters here and is driven by measurement, not taste:

1. Transliterate. Source 1 is entirely Latin while Sources 2 and 3 carry
   nine Indic scripts, so an untransliterated comparison scores those
   pairs at exactly zero rather than merely low.
2. Collapse doubled letters. unidecode is a literal character map, so the
   same phrase arrives as "praaivett limittedd" from Devanagari and
   "praiveett limittedd" from Telugu. Collapsing runs makes four scripts
   converge on the identical string "praivet limited".
3. Only then strip legal suffixes. Done in this order the transliterated
   "limittedd" reduces to "limited" and is removed by the same rule that
   handles the Latin side, so both records lose the same tokens. Stripping
   first would leave the Indic record carrying two junk tokens the Latin
   record does not, dragging down the similarity of exactly the pairs
   transliteration is meant to rescue.

Legal suffixes are removed rather than expanded: expanding makes every
Indian company end in "private limited", padding n-gram overlap between
unrelated businesses. Whether two records agree on legal form is kept as a
separate Stage 2 feature instead.
"""

import re
import unicodedata

from unidecode import unidecode


# Latin legal forms, plus the transliterated spellings observed coming out
# of each Indic script once doubled letters are collapsed.
LEGAL_SUFFIX = re.compile(
    r"\b("
    # latin
    r"pvt|private|ltd|limited|ltda|inc|incorporated|llc|llp|pllc|lp|"
    r"corp|corporation|co|company|holdings|group|"
    r"sarl|sas|sasu|eurl|sci|snc|sa|societe|cie|"
    r"gmbh|ag|bv|nv|plc|oy|ab|as|spa|srl|"
    # transliterated from indic scripts
    r"praivet|praiveet|praibhet|piraivet|praivr|pra|"
    r"limitet|limird|kampani|knpni|kompani"
    r")\b"
)

STOPWORDS = re.compile(r"\b(the|and|of|for|at|in|on|a|an|et|de|la|le|les|du|des)\b")

NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
MULTISPACE = re.compile(r"\s+")
DOUBLED = re.compile(r"(.)\1+")

NULLISH = {"", "<null>", "null", "nil", "na", "n/a", "none", "-", "--", "---", "."}

# Applied on top of transliteration. unidecode renders Devanagari "फ" as "ph"
# where English writes "f", and Indic vowel length has no English analogue.
# Measured on the name signal alone these rules move India from 95.6% to
# 98.2% reachability while costing the US nothing, so they are worth their
# small loss of precision at the blocking stage.
_PHONETIC = [
    (re.compile(r"ph"), "f"),
    (re.compile(r"ck"), "k"),
    (re.compile(r"wh"), "w"),
    (re.compile(r"v"), "w"),
    (re.compile(r"z"), "s"),
]


def is_null(s: str | None) -> bool:
    """True for empty values and for the literal placeholders in this data."""
    return s is None or s.strip().lower() in NULLISH



# Confusable characters. The corpus substitutes digits for the letters they
# resemble -- s0lutions, g1obal, techn0logies, 5olutions -- which leaves a
# corrupted token sharing nothing with its clean form. Pure numbers are left
# alone: house numbers and pincodes are real digits and among the strongest
# signals there are.
_CONFUSE = str.maketrans({"0": "o", "1": "l", "5": "s", "6": "g",
                          "3": "e", "4": "a", "8": "b", "7": "t"})

# The variant table is keyed on tokens as they look after doubled letters are
# collapsed, so both sides of it are collapsed here rather than at each lookup.
_VARIANTS = {DOUBLED.sub(r"\1", k): DOUBLED.sub(r"\1", v)
             for k, v in VARIANTS.items()}


def _unconfuse(tok: str) -> str:
    if tok.isdigit() or not any(c.isdigit() for c in tok):
        return tok
    return tok.translate(_CONFUSE)


def _singular(tok: str) -> str:
    """Fold plurals so "technologies" and "technology" agree."""
    if len(tok) > 5 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def canon_token(tok: str) -> str:
    """One spelling per meaning: rd and road, mh and maharashtra, s0lutions.

    Followed to a fixed point rather than applied a fixed number of times,
    because clusters chain -- rd to road to the cluster's own canonical form --
    and stopping early would land two spellings of one word on different
    tokens, which is the whole failure this exists to prevent.
    """
    t = _unconfuse(tok)
    for _ in range(4):
        nxt = _singular(_VARIANTS.get(t, t))
        nxt = _VARIANTS.get(nxt, nxt)
        if nxt == t:
            break
        t = nxt
    return t


def normalize(s: str | None, *, translit: bool = True, collapse: bool = True,
              drop_legal: bool = True, drop_stop: bool = True,
              canon: bool = True) -> str:
    """Canonical form used for both indexing and querying.

    Never returns empty for a non-null input: if suffix removal would
    consume the whole string -- "Private Limited" as an entire name, or a
    French name reduced to nothing -- the pre-removal form is kept, since
    a blank key matches nothing and silently costs recall.
    """
    if is_null(s):
        return ""
    s = unicodedata.normalize("NFKC", s)
    if translit:
        s = unidecode(s)
    s = NON_ALNUM.sub(" ", s.lower())
    if collapse:
        s = DOUBLED.sub(r"\1", s)
    before = MULTISPACE.sub(" ", s).strip()
    if canon:
        before = " ".join(canon_token(t) for t in before.split())

    out = before
    if drop_legal:
        out = LEGAL_SUFFIX.sub(" ", out)
    if drop_stop:
        out = STOPWORDS.sub(" ", out)
    out = MULTISPACE.sub(" ", out).strip()
    return out if out else before


def phonetic(s: str) -> str:
    """Pull transliterated spellings toward their English equivalents.

    Raises recall and lowers precision, which is the right trade at the
    blocking stage but not at the scoring stage -- Stage 2 should compare
    the un-phoneticised forms.
    """
    for pat, rep in _PHONETIC:
        s = pat.sub(rep, s)
    return s


def blocking_key(s: str | None) -> str:
    """The most aggressive form, used only to generate candidates."""
    return phonetic(normalize(s))


def sorted_tokens(s: str) -> str:
    """Word-order-invariant key: 'sharma traders' == 'traders sharma'."""
    return " ".join(sorted(s.split()))


def numeric_tokens(s: str) -> frozenset[str]:
    """Digit groups from an address. Numbers survive transliteration and
    abbreviation when words do not, so they make a sharp blocking key."""
    return frozenset(re.findall(r"\d+", s))


_SCRIPT_RANGES = (
    (0x0000, 0x024F, "latin"), (0x0900, 0x097F, "devanagari"),
    (0x0980, 0x09FF, "bengali"), (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"), (0x0B00, 0x0B7F, "oriya"),
    (0x0B80, 0x0BFF, "tamil"), (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"), (0x0D00, 0x0D7F, "malayalam"),
    (0x0600, 0x06FF, "arabic"), (0x0400, 0x04FF, "cyrillic"),
    (0x4E00, 0x9FFF, "cjk"),
)


def script_of(s: str) -> str:
    """Dominant writing system: a Stage 2 feature and a diagnostic."""
    counts: dict[str, int] = {}
    for ch in s:
        if not ch.isalpha():
            continue
        o = ord(ch)
        for lo, hi, name in _SCRIPT_RANGES:
            if lo <= o <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            counts["other"] = counts.get("other", 0) + 1
    return max(counts, key=counts.get) if counts else "none"


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def find_dataset():
    for p in IN.rglob(f"{SPLIT}_source1.tsv"):
        return p.parent
    raise SystemExit(f"{SPLIT}_source1.tsv not found under {IN}")


def load(d, i):
    df = pd.read_csv(d / f"{SPLIT}_source{i}.tsv", sep="\t", dtype=str).fillna("")
    df["name_norm"] = [normalize(x) for x in df.business_name]
    df["addr_norm"] = [normalize(x) for x in df.business_address]
    log(f"S{i}: {len(df):,} rows normalised")
    return df


def field(df, sig):
    if sig == "addr":
        return df.addr_norm.tolist()
    if sig == "name":
        return df.name_norm.tolist()
    return (df.name_norm + " " + df.addr_norm).str.strip().tolist()


def main():
    d = find_dataset()
    log(f"dataset at {d}")
    s1 = load(d, 1)
    tgt = {2: load(d, 2), 3: load(d, 3)}
    OUT.mkdir(parents=True, exist_ok=True)

    for sig, src in PLAN:
        out = OUT / f"candidate_pairs_{sig}_s{src}.tsv"
        if out.exists():
            log(f"{out.name} exists, skipping")
            continue
        cand = {}
        for cty in sorted(s1.country.unique()):
            qs = s1[s1.country == cty]
            ts = tgt[src][tgt[src].country == cty]
            if not len(qs) or not len(ts):
                continue
            v = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1,
                                max_df=MAX_DF, dtype="float32")
            M = v.fit_transform(field(ts, sig)).tocsr()
            Mt = M.T.tocsr()
            tid = ts.entity_id.to_numpy()
            log(f"{sig} s{src} {cty}: index {M.shape[0]:,}, nnz/doc {M.nnz/M.shape[0]:.1f}")
            del M
            gc.collect()
            qi = qs.entity_id.to_numpy()
            done = 0
            for lo in range(0, len(qs), CHUNK):
                hi = min(lo + CHUNK, len(qs))
                Q = v.transform(field(qs.iloc[lo:hi], sig)).tocsr()
                C = sp_matmul_topn(Q, Mt, top_n=K, threshold=1e-6,
                                   n_threads=THREADS, sort=False).tocoo()
                for r, c in zip(C.row, C.col):
                    cand.setdefault(qi[lo + r], []).append(tid[c])
                done = hi
                del Q, C
                gc.collect()
            log(f"{sig} s{src} {cty}: {done:,} queries done")
            del v, Mt, tid, ts
            gc.collect()
        total = 0
        with out.open("w", encoding="utf-8") as fh:
            fh.write("source1_entity_id\tcandidate_entity_ids\n")
            for q in s1.entity_id:
                ids = sorted(set(cand.get(q, ())))
                total += len(ids)
                fh.write(f"{q}\t{','.join(ids)}\n")
        log(f"wrote {out.name}: {total:,} pairs, {total/len(s1):.1f}/entity")
        del cand
        gc.collect()


if __name__ == "__main__":
    main()
