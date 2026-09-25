"""Country-agnostic text normalisation for business names and addresses.

Every record gets:
  name_n   canonical name tokens (abbreviations/legal forms mapped to one spelling)
  name_c   "core" name: name_n without legal-form / filler tokens
  addr_n   canonical address tokens
  nums     digit tokens from the address (house numbers, postcodes, plot numbers)
  zip      last digit token of length >= 5 (ZIP / PIN / code postal), '' if none
  name_k   consonant skeleton of name_c (bridges transliteration variants)
  legal    sorted canonical legal-form tokens (inc, llc, pvt ltd, sarl, ...)
  name_alt the other half of a DBA name, '' if none
  is_dom   '1' when the raw name was a web domain ('ryfoods.com')
The maps below cover English (US/India) and French forms; nothing branches on the
country label, so an unseen country simply goes through the same generic path.
"""
import re
import unicodedata
from multiprocessing import Pool

from anyascii import anyascii

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGIT_ALPHA = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")


def _pairs(spec: str) -> dict:
    """Parse 'canon: v1 v2 v3; canon2: ...' into {variant: canon}."""
    out = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        canon, variants = part.split(":")
        canon = canon.strip()
        for v in variants.split() + [canon]:
            out[v] = canon
    return out


NAME_MAP = _pairs("""
inc: incorporated incorp incorporation; corp: corporation corpn corporate; co: company compagnie cie;
ltd: limited ltee; pvt: private pvte prv; llc: lc; llp: ; plc: ; pllc: ; lp: ; pc: ; opc: ;
and: n et; intl: international internatl intnl; mfg: manufacturing mfrs manufacturers mfr;
svc: service services svcs srvc srvcs; assoc: associates association assn assocs associated;
bros: brothers bro; ctr: center centre cntr; mgmt: management mgt; tech: technology technologies technol;
ent: enterprise enterprises ents enterprize; ind: industries industry inds industrial;
grp: group groupe; hosp: hospital; univ: university; natl: national; dept: department;
dr: doctor; st: saint; ste: sainte; mt: mount; ft: fort; shri: sri shree sree shiri; smt: shrimati;
sons: son; traders: trader trading; engg: engineering engineers engineer eng; elec: electrical electricals electric electricals;
constr: construction constructions; dist: distributors distributor distribution; sys: systems system;
sol: solutions solution; soc: societe society ste; ets: etablissements etablissement; freres: frere;
restr: restaurant resto; pharm: pharmacy pharma pharmaceuticals pharmaceutical; ins: insurance;
fin: financial finance; invest: investments investment; prop: properties property;
dev: development developers developer; cons: consulting consultants consultancy consultant;
mktg: marketing; lab: labs laboratory laboratories; med: medical; ed: education educational;
sons: fils;
""")

# Canonical legal-form tokens (kept separately: a legal-form conflict is a strong
# "different business" signal, e.g. 'UDE Service EI' vs 'UDE Service SAS').
LEGAL = set("""
inc corp co ltd pvt llc llp plc pllc lp pc opc pa sa sas sasu sarl eurl sci snc ei gmbh ag bv nv
""".split())
# Legal forms plus filler words dropped from the "core" name.
NAME_DROP = LEGAL | set("the and of a an le la les l de du des d ms".split())
ADDR_MAP = _pairs("""
st: street str strt saint; rd: road; ave: avenue av aven avn avnue; blvd: boulevard boul bd bld blv;
dr: drive drv; ln: lane; ct: court crt; pl: place plc; sq: square; hwy: highway hiway hway;
pkwy: parkway pky; cir: circle; ter: terrace; trl: trail; way: wy; ste: suite su sainte;
apt: apartment apartments; bldg: building bldng bld batiment bat; fl: floor flr; rm: room;
no: number num nr nbr; n: north; s: south; e: east; w: west; ne: northeast; nw: northwest;
se: southeast; sw: southwest; mt: mount; ft: fort; near: nr nar; opp: opposite oppo opps;
bhd: behind; nagar: ngr; colony: col clny; sector: sec sectr; extn: extension ext;
hno: hnumber; po: postoffice; chemin: ch chem; impasse: imp; allee: all; route: rte;
fbg: faubourg; res: residence; pt: point; hts: heights; jct: junction; ctr: center centre;
cross: crs; main: mn; layout: lyt; chowk: chk; marg: mrg; gali: gl;
rue: r; chaussee: chee; cite: ; sente: ;
""")

# Phrases replaced before tokenisation (multi-word forms).
_PHRASES = [
    (re.compile(r"\bh\s*no\b|\bhouse\s+no\b|\bhouse\s+number\b|\bdoor\s+no\b|\bd\s*no\b"), " hno "),
    (re.compile(r"\bpost\s+office\b|\bp\s+o\b"), " po "),
    (re.compile(r"\bprivate\s+limited\b"), " pvt ltd "),
    (re.compile(r"\bpublic\s+limited\b"), " plc "),
    (re.compile(r"\bn\s*(?=\d)"), " no "),  # 'N°16' -> 'no 16' (not 'north')
]
_DBA = re.compile(r"\b(?:doing business as|trading as|d\s*/?\s*b\s*/?\s*a|t\s*/\s*a|a\s*/?\s*k\s*/?\s*a)\b", re.I)
_PHONE = re.compile(r"\+?\d[\d\s\-]{6,}\d")
_DOMAIN = re.compile(r"^\s*(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|in|co|fr|biz|info|us|io)(?:\.[a-z]{2})?\s*$", re.I)

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
}
# Multi-word states are replaced as phrases; single-word ones are just token maps.
_STATE_PHRASES = []
for _full, _ab in {**US_STATES, **IN_STATES}.items():
    if " " in _full:
        _STATE_PHRASES.append((re.compile(r"\b" + _full + r"\b"), f" {_ab} "))
    else:
        ADDR_MAP.setdefault(_full, _ab)


_SIGNS = str.maketrans({"°": " ", "º": " ", "№": " no ", "ª": " "})


def base(s: str) -> str:
    """Unicode fold -> ASCII transliteration -> lowercase -> alnum tokens separated by spaces."""
    if not s:
        return ""
    if not s.isascii():
        s = anyascii(unicodedata.normalize("NFKC", s.translate(_SIGNS)))
    s = s.lower().replace("&", " and ").replace("'", "")
    s = _NON_ALNUM.sub(" ", s)
    return s.strip()


def _merge_initials(toks):
    """Join runs of single letters: 'l l c' -> 'llc', 'i b m' -> 'ibm'."""
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run)) if len(run) > 1 else out.append(run[0])
            run = []
        out.append(t)
    if run:
        out.append("".join(run)) if len(run) > 1 else out.append(run[0])
    return out


def _name_tokens(s: str):
    s = base(s)
    for pat, rep in _PHRASES:
        s = pat.sub(rep, s)
    toks = [NAME_MAP.get(t, t) for t in _merge_initials(s.split())]
    if toks and toks[0] == "ms" and len(toks) > 1:  # 'M/s Darsh Trading'
        toks = toks[1:]
    out = []
    for t in toks:  # collapse immediate repeats: 'limited limited', 'inc inc'
        if not out or out[-1] != t:
            out.append(t)
    return out


def norm_name(raw: str):
    """Returns (name_n, name_c, legal, name_alt, is_domain).

    DBA forms 'X doing business as Y' use Y as the main name and X as name_alt.
    Domain-style names ('ryfoods.com') are reduced to their stem.
    """
    raw = _PHONE.sub(" ", raw or "")
    is_dom = 0
    m = _DOMAIN.match(raw)
    if m:
        raw, is_dom = m.group(1), 1
    alt = ""
    parts = _DBA.split(raw, maxsplit=1)
    if len(parts) == 2 and parts[1].strip():
        alt_t = _name_tokens(parts[0])
        alt = " ".join(t for t in alt_t if t not in NAME_DROP)
        raw = parts[1]
    toks = _name_tokens(raw)
    legal = " ".join(sorted({t for t in toks if t in LEGAL}))
    core = [t for t in toks if t not in NAME_DROP]
    return " ".join(toks), " ".join(core if core else toks), legal, alt, is_dom


def norm_addr(raw: str):
    s = base(raw)
    for pat, rep in _PHRASES:
        s = pat.sub(rep, s)
    for pat, rep in _STATE_PHRASES:
        s = pat.sub(rep, s)
    s = _DIGIT_ALPHA.sub(" ", s)  # '12b' -> '12 b', 'sector5' -> 'sector 5'
    toks = [ADDR_MAP.get(t, t) for t in s.split()]
    nums = [t for t in toks if t.isdigit()]
    zips = [t for t in nums if len(t) >= 5]
    return " ".join(toks), " ".join(nums), (zips[-1] if zips else "")


_SKEL_SUBS = [(re.compile(p), r) for p, r in (
    (r"ph", "f"), (r"[ckq]", "k"), (r"w", "v"), (r"z", "j"), (r"x", "ks"),
    (r"(?<=.)[aeiouyh]", ""), (r"(.)\1+", r"\1"),
)]


_OCR = str.maketrans("0134578", "oleastb")


def skeleton_token(t: str) -> str:
    """Consonant skeleton used to bridge transliteration variants.

    Latin 'ganesh traders' and anyascii('गणेश ट्रेडर्स') = 'gnes tredrs' both map to 'gns trdrs'.
    Digits are kept as-is.
    """
    if t.isdigit():
        return t
    if any(c.isdigit() for c in t):  # OCR-style digit-for-letter swaps: 'heart1and', 'f0os'
        t = t.translate(_OCR)
    for pat, rep in _SKEL_SUBS:
        t = pat.sub(rep, t)
    return t


def skeleton(s: str) -> str:
    return " ".join(skeleton_token(t) for t in s.split())


def _norm_row(args):
    name, addr = args
    nn, nc, legal, alt, dom = norm_name(name)
    an, nums, z = norm_addr(addr)
    return nn, nc, an, nums, z, skeleton(nc), legal, alt, str(dom)


def normalize_frame(df, n_proc: int = 40):
    """Add name_n, name_c, addr_n, nums, zip, name_k, legal, name_alt, is_dom columns to a polars frame (parallel)."""
    import polars as pl
    rows = list(zip(df["business_name"].to_list(), df["business_address"].to_list()))
    with Pool(n_proc) as pool:
        res = pool.map(_norm_row, rows, chunksize=20000)
    names = ["name_n", "name_c", "addr_n", "nums", "zip", "name_k", "legal", "name_alt", "is_dom"]
    cols = list(zip(*res)) if res else [[]] * len(names)
    return df.with_columns(pl.Series(n, list(c), dtype=pl.Utf8) for n, c in zip(names, cols))
