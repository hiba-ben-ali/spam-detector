import re


def nettoyer(texte):
    """Nettoie un message : minuscules, URL remplacées, ponctuation supprimée."""
    texte = texte.lower()
    texte = re.sub(r"http\S+|www\S+", " url ", texte)
    texte = re.sub(r"[^a-z0-9\s]", " ", texte)
    return re.sub(r"\s+", " ", texte).strip()