"""Dashboard Streamlit : suivi en temps reel du consumer Spark BUSI.

Place dans le schema :
    8. lecture de data/output/ (fichiers ecrits par le consumer et le service ML) et affichage.

Le dashboard ne calcule rien lui-meme : il relit les fichiers de data/output toutes les
quelques secondes et les affiche. Il permet aussi d'envoyer ses propres images : elles sont
deposees dans data/destination, exactement comme le ferait le producer, et c'est le consumer
qui les predit.

Lancer depuis la racine du projet :
    streamlit run dashboard/app.py
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

import altair as alt          # librairie de graphiques (incluse avec Streamlit)
import pandas as pd
import streamlit as st
from PIL import Image         # lecture / conversion des images envoyees

# chemins calcules a partir de l'emplacement de ce fichier
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "output"         # resultats du consumer et du service ML
IMAGES_DIR = ROOT / "data" / "destination"    # images du flux (pour les afficher)
STAGING_DIR = ROOT / "data" / "staging"       # dossier tampon pour envoyer une image

LABELS = ["benign", "malignant", "normal"]
# une couleur fixe par classe (meme couleur dans tous les graphiques)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
COLOR_SCALE = alt.Scale(domain=LABELS, range=COLORS)

# titre de l'onglet du navigateur ; "wide" = la page utilise toute la largeur de l'ecran
st.set_page_config(page_title="BUSI Streaming", layout="wide")


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    """Lit un CSV de data/output. Renvoie un tableau vide si le fichier n'existe pas encore."""
    path = OUTPUT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    try:
        # on_bad_lines : ignore une derniere ligne en cours d'ecriture
        return pd.read_csv(path, on_bad_lines="skip", **kwargs)
    except (pd.errors.EmptyDataError, PermissionError):
        return pd.DataFrame()


def read_metrics() -> dict:
    """Lit les metriques du dernier modele (ecrites par le service ML). {} si pas encore de modele."""
    path = OUTPUT_DIR / "model_metrics.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return {}


def send_to_pipeline(uploaded_file) -> str:
    """Convertit l'image en PNG et la depose dans data/destination (le consumer la detecte).

    Ecriture dans staging puis rename atomique, comme le producer.
    Le nom commence par "upload_" : le consumer n'en deduit aucune classe,
    et le service ML ne l'utilise donc pas pour l'entrainement.
    """
    # nom du fichier nettoye : on remplace espaces, accents, etc. par "_" et on limite la longueur
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(uploaded_file.name).stem)[:40]
    # la date a la microseconde rend le nom unique (on peut envoyer deux fois la meme image)
    name = f"upload_{datetime.now():%Y%m%d-%H%M%S-%f}_{stem}.png"
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STAGING_DIR / name
    # le consumer ne lit que les .png : un JPG est converti en PNG ici
    Image.open(uploaded_file).convert("RGB").save(tmp, format="PNG")
    os.replace(tmp, IMAGES_DIR / name)
    return name


def confusion_chart(matrix: pd.DataFrame) -> alt.Chart:
    """Matrice de confusion : une case par couple (vraie classe, classe predite).
    La diagonale = bonnes predictions ; le reste = erreurs. Plus la case est foncee, plus il y a d'images.

    matrix : colonnes actual, predicted, n."""
    base = alt.Chart(matrix).encode(
        x=alt.X("predicted:N", title="Classe predite", sort=LABELS),
        y=alt.Y("actual:N", title="Vraie classe", sort=LABELS),
    )
    # les cases colorees (du bleu clair au bleu fonce selon le nombre d'images)
    cells = base.mark_rect(stroke="white", strokeWidth=2, cornerRadius=4).encode(
        color=alt.Color("n:Q", scale=alt.Scale(range=["#cde2fb", "#104281"]), legend=None),
        tooltip=[
            alt.Tooltip("actual:N", title="Vraie classe"),
            alt.Tooltip("predicted:N", title="Predite"),
            alt.Tooltip("n:Q", title="Images"),
        ],
    )
    # le nombre ecrit dans chaque case : en blanc sur les cases foncees, en noir sur les claires
    threshold = max(int(matrix["n"].max()), 1) / 2
    text = base.mark_text(fontSize=16, fontWeight="bold").encode(
        text="n:Q",
        color=alt.condition(alt.datum.n > threshold, alt.value("white"), alt.value("#0b0b0b")),
    )
    return (cells + text).properties(height=280)


# session_state : memoire propre a chaque onglet du navigateur, conservee entre les rafraichissements.
# On y garde la liste des images envoyees par l'utilisateur pour afficher leurs resultats.
if "uploads" not in st.session_state:
    st.session_state.uploads = []  # noms des images envoyees par l'utilisateur

# ---------- barre laterale : reglages ----------
with st.sidebar:
    st.header("Parametres")
    refresh = st.slider("Rafraichissement (s)", 1, 30, 3)                       # frequence de relecture
    gallery_size = st.slider("Images dans la galerie", 3, 24, 6, step=3)       # taille de la galerie


# st.fragment(run_every=...) : cette fonction est re-executee toute seule toutes les `refresh`
# secondes, sans recharger le reste de la page. C'est ce qui rend le dashboard "temps reel".
@st.fragment(run_every=refresh)
def sidebar_summary():
    """Resume du pipeline dans la barre laterale, rafraichi comme le reste."""
    images = read_csv("images.csv", parse_dates=["ingestionTime"])
    metrics = read_metrics()

    st.header("Resume")
    # etat du pipeline : vert si une image a ete traitee dans la derniere minute
    last = images["ingestionTime"].max() if not images.empty else None
    if last is None or pd.isna(last):
        st.caption("⚪ Pipeline en attente")
    else:
        age = max((datetime.now() - last).total_seconds(), 0)
        if age < 60:
            st.caption(f"🟢 Pipeline actif - derniere image il y a {age:.0f} s")
        else:
            st.caption(f"⚪ Aucune image depuis {age / 60:.0f} min")

    # nombre d'images, detaille par classe (les images envoyees n'ont pas de classe : label vide)
    st.metric("Images recues", len(images))
    if not images.empty:
        counts = images["label"].value_counts()
        lines = [f"- {label} : **{int(counts.get(label, 0))}**" for label in LABELS]
        lines.append(f"- envoyees par toi : **{int(images['label'].isna().sum())}**")
        st.markdown("\n".join(lines))

    # informations sur le dernier modele exporte par le service ML
    st.subheader("Modele")
    if not metrics:
        st.caption("Pas encore de modele")
        return
    st.metric(
        "Precision",
        f"{metrics['cvAccuracy']:.0%}",
        help="Validation croisee : chaque image est predite par un modele qui ne l'a pas vue.",
    )
    st.caption(f"Version {metrics['version']}  \nEntraine sur {metrics['nbImages']} images")
    # images reellement utilisees pour l'entrainement (differentes des images recues si --balance)
    train_counts = metrics.get("classCounts", {})
    st.markdown("\n".join(f"- {label} : **{train_counts.get(label, 0)}**" for label in LABELS))


with st.sidebar:
    st.divider()
    sidebar_summary()

st.title("BUSI - Spark Streaming")
st.caption("Echographies mammaires ingerees en temps reel par le consumer Spark")

# ---------- envoi d'images par l'utilisateur ----------
# st.form : les fichiers ne sont envoyes qu'au clic sur le bouton ;
# clear_on_submit vide la zone de depot apres l'envoi.
st.header("Predire mes images")
with st.form("upload", clear_on_submit=True):
    files = st.file_uploader(
        "Depose une ou plusieurs echographies (PNG ou JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    submitted = st.form_submit_button("Envoyer au pipeline", type="primary")
if submitted and files:
    for f in files:
        st.session_state.uploads.append(send_to_pipeline(f))
    st.success(f"{len(files)} image(s) envoyee(s). Le consumer va les analyser au prochain micro-batch.")


# partie principale de la page, rafraichie elle aussi toutes les `refresh` secondes
@st.fragment(run_every=refresh)
def live_dashboard():
    images = read_csv("images.csv", parse_dates=["ingestionTime"])
    metrics = read_metrics()

    # ---------- resultats des images envoyees ----------
    # pour chaque image envoyee, on cherche sa ligne dans images.csv :
    # absente = pas encore traitee par le consumer ; presente = on affiche la prediction
    if st.session_state.uploads:
        st.subheader("Resultats de mes images")
        # index par nom de fichier pour retrouver une image rapidement
        by_name = (
            images.drop_duplicates("fileName", keep="last").set_index("fileName")
            if not images.empty else pd.DataFrame()
        )
        grid = st.columns(3)   # grille de 3 colonnes
        for i, name in enumerate(reversed(st.session_state.uploads)):   # la plus recente d'abord
            with grid[i % 3]:
                path = IMAGES_DIR / name
                if path.exists():
                    st.image(str(path), width="stretch")
                if name not in by_name.index:
                    st.caption("⏳ En cours d'analyse par le consumer...")
                    continue
                row = by_name.loc[name]
                if isinstance(row.get("predictedLabel"), str):
                    st.markdown(f"Prediction : **{row['predictedLabel']}** ({row['confidence']:.0%} de confiance)")
                else:
                    # le consumer a traite l'image avant que le service ML ait exporte un modele
                    st.caption("Analysee, mais aucun modele n'est encore disponible.")

    st.divider()

    if images.empty:
        st.info("En attente de donnees... Lance le consumer puis le producer.")
        return

    left, right = st.columns(2)

    # ---------- images par classe ----------
    with left:
        st.subheader("Images par classe")
        # deux barres par classe : images recues par le consumer, et images utilisees pour
        # entrainer le modele (moins nombreuses si le service ML equilibre les classes)
        # reindex : les 3 classes apparaissent toujours, meme a 0
        received = images["label"].value_counts().reindex(LABELS, fill_value=0)
        train_counts = metrics.get("classCounts", {}) if metrics else {}
        by_label = pd.DataFrame(
            [(label, "Recues", int(received[label])) for label in LABELS]
            + [(label, "Entrainement", int(train_counts.get(label, 0))) for label in LABELS if train_counts],
            columns=["label", "source", "nbImages"],
        )
        bar = (
            alt.Chart(by_label)
            .mark_bar(cornerRadiusEnd=4, size=14)
            .encode(
                x=alt.X("nbImages:Q", title="Nombre d'images", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("label:N", title=None, sort=LABELS),
                # yOffset : les deux barres d'une classe sont l'une sous l'autre
                yOffset=alt.YOffset("source:N", sort=["Recues", "Entrainement"]),
                color=alt.Color("label:N", scale=COLOR_SCALE, legend=None),
                # barres d'entrainement plus claires
                opacity=alt.Opacity("source:N", scale=alt.Scale(domain=["Recues", "Entrainement"], range=[1, 0.45]),
                                    legend=alt.Legend(title=None, orient="bottom")),
                tooltip=[alt.Tooltip("label:N", title="Classe"), alt.Tooltip("source:N", title="Images"),
                         alt.Tooltip("nbImages:Q", title="Nombre")],
            )
        )
        # le nombre ecrit au bout de chaque barre
        labels = bar.mark_text(align="left", dx=6).encode(text="nbImages:Q", color=alt.value("gray"))
        st.altair_chart((bar + labels).properties(height=280), width="stretch")

    # ---------- matrice de confusion ----------
    with right:
        st.subheader("Erreurs du modele")
        if not metrics:
            st.info("Pas encore de modele. Lance `python ml_service/train_model.py`.")
        else:
            # le JSON contient la matrice sous forme de liste de listes [[..],[..],[..]] :
            # on la transforme en tableau a 3 colonnes (vraie classe, classe predite, nombre)
            cm = metrics["confusionMatrix"]
            cm_df = pd.DataFrame(
                [(a, p, cm[i][j]) for i, a in enumerate(metrics["labels"]) for j, p in enumerate(metrics["labels"])],
                columns=["actual", "predicted", "n"],
            )
            st.altair_chart(confusion_chart(cm_df), width="stretch")

    # ---------- galerie ----------
    # les dernieres images traitees, avec leur vraie classe et la prediction du modele
    st.subheader("Dernieres images recues")
    latest = images.sort_values("ingestionTime", ascending=False).head(gallery_size)
    grid = st.columns(3)
    for i, row in enumerate(latest.itertuples()):
        path = IMAGES_DIR / row.fileName
        with grid[i % 3]:
            if path.exists():
                st.image(str(path), width="stretch")
            # label vide (NaN) = image envoyee par l'utilisateur, sans vraie classe connue
            actual = row.label if isinstance(row.label, str) else None
            caption = f"**{actual or 'image envoyee'}** - {row.fileName}"
            predicted = getattr(row, "predictedLabel", None)
            if isinstance(predicted, str):
                # bonne ou mauvaise prediction (impossible a dire pour une image envoyee)
                verdict = "" if actual is None else ("✅ " if predicted == actual else "❌ ")
                caption += f"  \n{verdict}predit : **{predicted}** ({row.confidence:.0%})"
            st.caption(caption)


live_dashboard()
