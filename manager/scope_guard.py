"""Garde-fou de périmètre de l'assistant Pilot.

Pilot ne répond qu'aux questions sur l'activité d'Olist (ventes, commandes, livraison, satisfaction, vendeurs,
catégories, analyses, scénarios) et sur les données ou les agents du pipeline. Tout le reste est refusé
poliment, avec des exemples de questions utiles.

Classement en trois couches :
  1. Règles déterministes (ce module), appliquées AVANT tout appel au LLM et dans les deux modes :
       - tentative de détournement des consignes ("ignore tes instructions"...)  -> hors périmètre
       - vocabulaire métier explicite (vendeur, commande, livraison, KPI...)      -> dans le périmètre
       - sujet clairement étranger (météo, recette, code, politique...)          -> hors périmètre
       - salutation, remerciement, « que sais-tu faire ? »                       -> conversation
       - vocabulaire d'analyse générique, ou relance courte dans une conversation -> dans le périmètre
       - sinon                                                                   -> incertain
  2. Mode LLM : une question « incertaine » est confiée au LLM, dont la consigne système définit le
     périmètre ; s'il la juge hors sujet, il répond par le jeton HORS_PERIMETRE (OUT_OF_SCOPE_TOKEN).
  3. Mode règles (sans LLM) : une question « incertaine » est refusée, car le routeur par mots-clés ne
     saurait de toute façon pas y répondre correctement.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

IN_SCOPE = "dans_perimetre"
OUT_OF_SCOPE = "hors_perimetre"
SMALL_TALK = "conversation"
UNCERTAIN = "incertain"

OUT_OF_SCOPE_TOKEN = "HORS_PERIMETRE"


@dataclass(frozen=True)
class Verdict:
    label: str
    reason: str
    injection: bool = False


def normalize(text: str) -> str:
    """Minuscules, sans accents, apostrophes et espaces unifiés : « Délai » -> « delai »."""
    t = unicodedata.normalize("NFKD", str(text or ""))
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
    t = t.replace("’", "'")
    return re.sub(r"\s+", " ", t).strip()


def _stems(words) -> re.Pattern:
    """Mots ou débuts de mots, reconnus en début de mot (« vend » -> vendeur, vente, vendu)."""
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + ")")


# Vocabulaire propre à l'activité d'Olist : sa présence suffit à rester dans le périmètre
STRONG = _stems([
    "olist", "marketplace", "chiffre d'affaire", "vente", "vendu", "vendeur", "seller", "sales", "revenue",
    "revenu", "commande", "order", "achat", "acheteur", "client", "customer", "categori", "produit", "product",
    "article", "livr", "deliver", "retard", "late", "delai", "expedi", "fret", "freight", "logisti", "transporteur",
    "avis client", "review", "satisf", "mecontent", "etoile", "kpi", "indicateur", "panier", "fidelis", "recurren",
    "reachat", "retention", "churn", "marge", "paiement", "payment", "boleto", "versement", "mensualit",
    "black friday", "jour ferie", "jours feries", "dashboard", "tableau de bord", "bi analyst", "xai", "shap",
    "pipeline", "agent kpi", "brl", "reais", "annul", "commandes perdues", "cohorte", "rfm", "best-seller",
    "best seller", "pareto",
])
REVENUE_ABBR = re.compile(r"\bCA\b")   # « CA » en majuscules (sinon « ça va » serait pris pour le CA)

# Vocabulaire d'analyse générique : dans le périmètre s'il n'y a pas de sujet étranger
WEAK = _stems([
    "note", "avis", "tendance", "croissance", "evolution", "baisse", "hausse", "recul", "chute", "progress",
    "augment", "diminu", "alerte", "signal", "anomal", "analys", "recommand", "priorit", "que faire", "simul",
    "scenario", "et si", "risque", "prevision", "periode", "mois", "annee", "trimestre", "semaine", "compar",
    "classement", "top", "meilleur", "pire", "definition", "defini", "calcul", "donnee", "qualite", "fiab",
    "explique", "explication", "prix", "volume", "performance", "bilan", "synthese", "resume", "situation",
    "chiffre", "perimetre", "filtre", "region", "etat", "ville", "saison", "evenement", "concentration",
    "objectif", "cible", "budget", "cout", "rentab", "benefice", "strategi",
    # lieux : seuls, ils ne suffisent pas (« quel temps fait-il à São Paulo ? » reste hors périmètre)
    "sao paulo", "rio de janeiro", "minas gerais", "pernambuco", "bresil", "bresilien",
])

# Sujets étrangers à l'activité d'Olist
OFF_TOPIC = _stems([
    "meteo", "quel temps", "temps qu'il fait", "pluie", "recette", "cuisin", "blague", "humour", "poeme",
    "poesie", "chanson", "musique", "film", "serie tv", "netflix", "football", "match de", "sport", "politique",
    "election", "president", "gouvernement", "guerre", "religion", "dieu", "medecin", "maladie", "symptome",
    "medicament", "sante", "python", "javascript", "java ", "sql", "html", "css", "programm", "code source",
    "ecris un code", "ecris du code", "traduis", "traduire", "traduction de", "capitale", "geographie",
    "histoire de", "bitcoin", "crypto", "bourse", "horoscope", "astrolog", "voyage", "hotel", "vacances",
    "devoir", "dissertation", "lettre de motivation", "cv ", "rencontre", "amour", "philosoph", "equation",
    "integrale", "derivee", "jeu video", "minecraft", "chatgpt", "openai",
])

INJECTION = re.compile(
    r"\b(ignore|oublie|ignorer|oublier)\b.{0,30}\b(instruction|consigne|regle|prompt|directive)"
    r"|\b(system prompt|prompt systeme|jailbreak|developer mode|mode developpeur)\b"
    r"|\b(tu es maintenant|fais semblant|joue le role|act as|you are now|pretend)\b"
    r"|\bignore (all|previous|the above)\b"
)

SMALL_TALK_RE = re.compile(
    r"^(bonjour|bonsoir|salut|hello|hi|coucou|hey|merci( beaucoup)?|thanks|thank you|au revoir|bye|a bientot|"
    r"ok|d'accord|super|parfait|top merci|qui es[- ]tu|tu es qui|c'est quoi pilot|que (sais|peux)[- ]tu faire|"
    r"comment (ca|tu) (va|vas)|aide|help|comment t'utiliser)\b[ !.?,]*$"
)

ARITHMETIC = re.compile(r"^\s*(combien (font|fait)|calcule)?\s*[\d\s.,]+[+*/x×÷-][\d\s.,+*/x×÷-]+\??\s*$")
STATE_CODE = re.compile(r"\b(AC|AL|AM|AP|BA|CE|DF|ES|GO|MA|MG|MS|MT|PA|PB|PE|PI|PR|RJ|RN|RO|RR|RS|SC|SE|SP|TO)\b")
DATE_HINT = re.compile(r"\b(201[6-8]|20\d\d-\d\d|janv|fevr|mars|avril|mai|juin|juil|aout|sept|oct|nov|dec)")


def classify(question: str, history: list | None = None) -> Verdict:
    """Classe une question de l'utilisateur. Déterministe, sans appel réseau."""
    raw = str(question or "")
    q = normalize(raw)
    if not q:
        return Verdict(OUT_OF_SCOPE, "question vide")
    if INJECTION.search(q):
        return Verdict(OUT_OF_SCOPE, "tentative de modifier les consignes de l'assistant", injection=True)
    if ARITHMETIC.match(q):
        return Verdict(OUT_OF_SCOPE, "calcul sans lien avec les données d'Olist")
    if STRONG.search(q) or REVENUE_ABBR.search(raw):
        return Verdict(IN_SCOPE, "vocabulaire de l'activité Olist")
    if OFF_TOPIC.search(q + " "):
        return Verdict(OUT_OF_SCOPE, "sujet étranger à l'activité d'Olist")
    if SMALL_TALK_RE.match(q):
        return Verdict(SMALL_TALK, "salutation ou question sur l'assistant")
    if WEAK.search(q):
        return Verdict(IN_SCOPE, "vocabulaire d'analyse")
    if history and len(q) <= 80 and (STATE_CODE.search(raw) or DATE_HINT.search(q) or q.startswith(("et ", "pour ", "en "))):
        return Verdict(IN_SCOPE, "relance de la question précédente")
    return Verdict(UNCERTAIN, "aucun vocabulaire métier reconnu")


CAPABILITIES = (
    "- **Ventes** : chiffre d'affaires, commandes, panier moyen, catégories, croissance sur un an\n"
    "- **Livraison et satisfaction** : retards, délais, notes, causes des mauvaises notes\n"
    "- **Vendeurs** : vendeurs à risque, fiche d'un vendeur, concentration\n"
    "- **Analyses** : alertes du BI Analyst, pourquoi le CA a changé (XAI), scénarios « et si ? »"
)
EXAMPLE = "Par exemple : « Pourquoi le CA a-t-il baissé le dernier mois ? »"


def refusal_text(verdict: Verdict) -> str:
    if verdict.injection:
        head = ("Je ne peux pas modifier mes consignes : je reste **Pilot**, l'assistant BI d'Olist, "
                "et je réponds uniquement à partir des agents du pipeline.")
    elif verdict.label == UNCERTAIN:
        head = ("Je n'ai pas reconnu de question sur l'activité d'Olist. Je réponds uniquement sur les données "
                "de la marketplace, à partir des agents du pipeline.")
    else:
        head = ("Cette question sort de mon périmètre : je suis **Pilot**, l'assistant BI d'Olist, et je réponds "
                "uniquement sur l'activité de la marketplace.")
    return f"{head}\n\nJe peux vous aider sur :\n{CAPABILITIES}\n\n{EXAMPLE}"


def small_talk_text(question: str) -> str:
    q = normalize(question)
    if q.startswith(("merci", "thanks", "thank", "super", "parfait", "ok", "d'accord", "top")):
        return "Avec plaisir. Posez-moi une autre question sur l'activité d'Olist quand vous voulez."
    if q.startswith(("au revoir", "bye", "a bientot")):
        return "À bientôt !"
    return ("Bonjour, je suis **Pilot**, l'assistant BI d'Olist. Je réponds avec le travail des agents du "
            f"pipeline (KPI, BI Analyst, XAI), sur le périmètre filtré à l'écran.\n\nJe peux vous aider sur :\n"
            f"{CAPABILITIES}\n\n{EXAMPLE}")


def is_llm_refusal(text: str) -> bool:
    """Le LLM a jugé la question hors périmètre (il répond alors par le jeton convenu)."""
    return OUT_OF_SCOPE_TOKEN in str(text or "")[:200]


def guard_trace(verdict: Verdict, by: str = "règles") -> dict:
    return {"outil": "controle_perimetre", "agent": "Garde-fou",
            "parametres": {"verdict": verdict.label, "motif": verdict.reason, "decide_par": by}, "resultat": ""}
