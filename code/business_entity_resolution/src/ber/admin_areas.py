"""Administrative-area canonicalisation at the address-component level.

Sources write the top-level admin area differently: US/India use state names or codes
(handled token-wise in normalize.py), French records use either the *région*
('Hauts-de-France') or a *département* ('Nord', 'Pas-de-Calais'). A comma-separated
address component that is exactly a région or département name is mapped to one
canonical région, so both spellings agree. Street names are never touched because only
whole components are matched. This is static normalisation knowledge (like 'Rd' ->
'Road'), not an external lookup.
"""
import re
import unicodedata

from anyascii import anyascii

FR_REGIONS = {
    "ara": "Auvergne-Rhone-Alpes", "bfc": "Bourgogne-Franche-Comte", "bre": "Bretagne",
    "cvl": "Centre-Val de Loire", "cor": "Corse", "ges": "Grand Est", "hdf": "Hauts-de-France",
    "idf": "Ile-de-France", "nor": "Normandie", "naq": "Nouvelle-Aquitaine", "occ": "Occitanie",
    "pdl": "Pays de la Loire", "pac": "Provence-Alpes-Cote d'Azur",
}
FR_DEPTS = {
    "ara": "Ain;Allier;Ardeche;Cantal;Drome;Isere;Loire;Haute-Loire;Puy-de-Dome;Rhone;Savoie;Haute-Savoie",
    "bfc": "Cote-d'Or;Doubs;Jura;Nievre;Haute-Saone;Saone-et-Loire;Yonne;Territoire de Belfort",
    "bre": "Cotes-d'Armor;Finistere;Ille-et-Vilaine;Morbihan",
    "cvl": "Cher;Eure-et-Loir;Indre;Indre-et-Loire;Loir-et-Cher;Loiret",
    "cor": "Corse-du-Sud;Haute-Corse",
    "ges": "Ardennes;Aube;Marne;Haute-Marne;Meurthe-et-Moselle;Meuse;Moselle;Bas-Rhin;Haut-Rhin;Vosges",
    "hdf": "Aisne;Nord;Oise;Pas-de-Calais;Somme",
    "idf": "Seine-et-Marne;Yvelines;Essonne;Hauts-de-Seine;Seine-Saint-Denis;Val-de-Marne;Val-d'Oise",
    "nor": "Calvados;Eure;Manche;Orne;Seine-Maritime",
    "naq": "Charente;Charente-Maritime;Correze;Creuse;Dordogne;Gironde;Landes;Lot-et-Garonne;"
           "Pyrenees-Atlantiques;Deux-Sevres;Vienne;Haute-Vienne",
    "occ": "Ariege;Aude;Aveyron;Gard;Haute-Garonne;Gers;Herault;Lot;Lozere;Hautes-Pyrenees;"
           "Pyrenees-Orientales;Tarn;Tarn-et-Garonne",
    "pdl": "Loire-Atlantique;Maine-et-Loire;Mayenne;Sarthe;Vendee",
    "pac": "Alpes-de-Haute-Provence;Hautes-Alpes;Alpes-Maritimes;Bouches-du-Rhone;Var;Vaucluse",
}

_SEP = re.compile(r"[\s\-'’.]+")


def _key(s: str) -> str:
    s = anyascii(unicodedata.normalize("NFKC", s)) if not s.isascii() else s
    return _SEP.sub(" ", s.lower()).strip()


ADMIN = {}  # component key -> canonical display name
for code, name in FR_REGIONS.items():
    ADMIN[_key(name)] = name
    for d in FR_DEPTS[code].split(";"):
        ADMIN[_key(d)] = name


# base-normalised région name -> single code token (like US 'tn' / India 'gj')
REGION_CODE = {_key(name).replace("'", " "): code for code, name in FR_REGIONS.items()}
_REGION_RE = re.compile(r"\b(" + "|".join(sorted((re.escape(k) for k in REGION_CODE), key=len, reverse=True)) + r")\b")


def region_codes(s: str) -> str:
    """In an already base()-normalised address, replace région names by their code."""
    return _REGION_RE.sub(lambda m: REGION_CODE[m.group(1)], s)


def canon_admin(addr: str) -> str:
    """Replace whole address components that are a région/département by the région name."""
    if not addr or "," not in addr and _key(addr) not in ADMIN:
        return addr
    parts = addr.split(",")
    out = [ADMIN.get(_key(p), p.strip()) if _key(p) in ADMIN else p for p in parts]
    return ",".join(out)
