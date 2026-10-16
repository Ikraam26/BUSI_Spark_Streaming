"""Service ML : lit les caracteristiques ecrites par le consumer, entraine un
classifieur et l'exporte en ONNX pour que le consumer Scala le charge.

Place dans le schema :
    5. lecture de data/output/images.csv (ecrit par le consumer)
    6. export du modele dans model/busi_model.onnx (charge par le consumer)

Fonctionnement : le script tourne en boucle. Toutes les --interval secondes il relit
images.csv ; des qu'il y a assez d'images (au moins --min-per-class par classe), il
entraine un premier modele, puis il re-entraine a chaque fois que --retrain-every
nouvelles images sont arrivees. Le modele s'ameliore donc au fil du flux.
Le service s'arrete tout seul quand aucune nouvelle image n'arrive pendant --idle-timeout secondes.

Lancer depuis la racine du projet :
    python ml_service/train_model.py           # surveille images.csv et re-entraine
    python ml_service/train_model.py --once    # un seul entrainement
"""
import argparse
import itertools
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort                       # execute un modele ONNX (ici : pour verifier l'export)
import pandas as pd
from skl2onnx import convert_sklearn            # convertit un modele scikit-learn en ONNX
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# chemins calcules a partir de l'emplacement de ce fichier : le script marche quel que soit
# le dossier depuis lequel on le lance
ROOT = Path(__file__).resolve().parent.parent
IMAGES_CSV = ROOT / "data" / "output" / "images.csv"            # entree : ecrit par le consumer
METRICS_JSON = ROOT / "data" / "output" / "model_metrics.json"  # precision, matrice de confusion...
HISTORY_CSV = ROOT / "data" / "output" / "model_history.csv"    # une ligne par entrainement
MODEL_FILE = ROOT / "model" / "busi_model.onnx"                 # sortie : lu par le consumer

# l'ordre de cette liste definit le numero de chaque classe : benign=0, malignant=1, normal=2
LABELS = ["benign", "malignant", "normal"]
# memes noms de colonnes que ImageConsumer.featureColumns, sauf width et height :
# la taille d'une image ne dit rien de la lesion, et dans la base BUSI elle varie selon la classe,
# le modele apprendrait donc un biais de la base plutot que l'aspect de l'echographie
FEATURES = (
    ["meanIntensity", "stdIntensity", "entropy", "gradient"]
    + [f"h{i:02d}" for i in range(16)]
    + ["glcmContrast", "glcmHomogeneity", "glcmEnergy", "glcmCorrelation"]
    + [f"lbp{i}" for i in range(10)]
    + ["centerMean", "centerStd", "centerRatio"]
)
# opset supporte par onnxruntime Java 1.19 (cf. build.sbt)
# (l'opset est la "version" des operations ONNX ; une version trop recente ne serait pas lisible en Scala)
TARGET_OPSET = {"": 17, "ai.onnx.ml": 3}


def load_dataset() -> pd.DataFrame:
    """Lit images.csv et ne garde que les lignes utilisables pour l'entrainement."""
    if not IMAGES_CSV.exists():
        return pd.DataFrame()
    try:
        # on_bad_lines="skip" : ignore une derniere ligne que le consumer serait en train d'ecrire
        df = pd.read_csv(IMAGES_CSV, on_bad_lines="skip")
    except (pd.errors.EmptyDataError, PermissionError):
        return pd.DataFrame()
    # on ne garde que les images dont on connait la vraie classe (pas les images envoyees
    # depuis Streamlit) et dont toutes les caracteristiques sont presentes
    df = df[df["label"].isin(LABELS)].dropna(subset=FEATURES)
    # une image re-envoyee par le producer ne compte qu'une fois
    return df.drop_duplicates(subset="fileName", keep="last")


def balance(df: pd.DataFrame) -> pd.DataFrame:
    """Garde le meme nombre d'images dans chaque classe : celui de la plus petite classe
    (ex. 133 benign, 133 malignant, 133 normal), tirees au hasard (random_state fixe = reproductible)."""
    n = int(df["label"].value_counts().min())
    return df.groupby("label", group_keys=False).sample(n=n, random_state=42)


def write_atomic(path: Path, data: bytes) -> None:
    """Ecrit dans un fichier temporaire puis le renomme : le lecteur (consumer ou Streamlit)
    voit soit l'ancienne version complete, soit la nouvelle, jamais un fichier a moitie ecrit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    for _ in range(10):  # Windows : le fichier peut etre lu au meme instant
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.2)
    raise PermissionError(f"Impossible de remplacer {path}")


def train(df: pd.DataFrame) -> dict:
    """Entraine le modele, l'evalue, l'exporte en ONNX et ecrit ses metriques."""
    # X : une ligne par image, une colonne par caracteristique (float32 = type attendu par ONNX)
    X = df[FEATURES].to_numpy(dtype=np.float32)
    # y : la classe de chaque image, sous forme de numero (benign -> 0, malignant -> 1, normal -> 2)
    y = df["label"].map(LABELS.index).to_numpy(dtype=np.int64)

    # foret aleatoire : 200 arbres de decision qui votent.
    #  - pas de limite de profondeur : sur la base complete (780 images), des arbres complets
    #    donnent ~72 % contre ~66 % avec max_depth=6 ; le vote des 200 arbres limite le "par coeur"
    #  - class_weight="balanced" compense les classes moins representees
    #  - random_state=42 rend les resultats reproductibles
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)

    # evaluation honnete : validation croisee (chaque image predite par un modele qui ne l'a pas vue)
    # Les images sont coupees en `folds` groupes ; on entraine sur tous les groupes sauf un,
    # on predit le groupe restant, et on recommence pour chaque groupe.
    # "Stratified" : chaque groupe garde la meme proportion de benign/malignant/normal.
    # Le nombre de groupes ne peut pas depasser le nombre d'images de la plus petite classe.
    folds = int(min(5, np.bincount(y, minlength=len(LABELS)).min()))
    cv_pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=42))
    cv_accuracy = float((cv_pred == y).mean())   # part des images bien classees

    # modele final entraine sur toutes les images
    clf.fit(X, y)

    # la version est simplement la date et l'heure de l'entrainement
    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    # conversion scikit-learn -> ONNX
    #  - "input" : nom de l'entree, un tableau [n images, 37 caracteristiques] (None = n quelconque)
    #  - zipmap False : les probabilites sortent en tableau simple, plus facile a lire en Scala
    onx = convert_sklearn(
        clf,
        initial_types=[("input", FloatTensorType([None, len(FEATURES)]))],
        options={id(clf): {"zipmap": False}},  # probabilites en tenseur [n, nbClasses]
        target_opset=TARGET_OPSET,
    )
    # metadonnees enregistrees dans le fichier .onnx : le consumer y lit l'ordre des colonnes,
    # le nom des classes et la version. Ainsi le Scala n'a rien a "deviner".
    for key, value in {"labels": ",".join(LABELS), "features": ",".join(FEATURES), "version": version}.items():
        prop = onx.metadata_props.add()
        prop.key, prop.value = key, value
    model_bytes = onx.SerializeToString()

    # verification : l'ONNX donne les memes probabilites que scikit-learn
    # (si la conversion avait un probleme, on n'exporte pas un modele faux)
    sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    onnx_proba = sess.run(None, {"input": X})[1]   # sortie 1 = probabilites
    if not np.allclose(onnx_proba, clf.predict_proba(X), atol=1e-4):
        raise RuntimeError("Les predictions ONNX different de scikit-learn, export annule")

    # export du modele : le consumer le detectera a son prochain micro-batch
    write_atomic(MODEL_FILE, model_bytes)

    # metriques lues par le dashboard Streamlit
    metrics = {
        "version": version,
        "trainedAt": datetime.now().isoformat(timespec="seconds"),
        "nbImages": int(len(df)),
        "classCounts": {label: int((y == i).sum()) for i, label in enumerate(LABELS)},
        "cvFolds": folds,
        "cvAccuracy": round(cv_accuracy, 4),
        "labels": LABELS,
        # matrice de confusion : ligne = vraie classe, colonne = classe predite
        # ex. confusionMatrix[0][2] = nombre d'images benign predites normal
        "confusionMatrix": confusion_matrix(y, cv_pred, labels=range(len(LABELS))).tolist(),
        # importance de chaque caracteristique dans les decisions de la foret (la somme vaut 1)
        "featureImportances": {f: round(float(v), 4) for f, v in zip(FEATURES, clf.feature_importances_)},
    }
    write_atomic(METRICS_JSON, json.dumps(metrics, indent=2).encode("utf-8"))

    # historique : une ligne par entrainement (permet de suivre l'evolution de la precision)
    history_line = f"{version},{metrics['trainedAt']},{len(df)},{cv_accuracy:.4f}\n"
    if not HISTORY_CSV.exists():
        history_line = "version,trainedAt,nbImages,cvAccuracy\n" + history_line
    with open(HISTORY_CSV, "a", encoding="utf-8") as f:
        f.write(history_line)

    print(f"[ML] modele {version} exporte : {len(df)} images, accuracy CV = {cv_accuracy:.1%} ({folds} folds)")
    return metrics


def can_train(df: pd.DataFrame, min_per_class: int) -> bool:
    """Il faut au moins `min_per_class` images de chaque classe : sinon le modele ne peut pas
    apprendre a reconnaitre une classe qu'il n'a jamais vue, et la validation croisee echoue."""
    if df.empty:
        return False
    counts = df["label"].value_counts()
    return all(counts.get(label, 0) >= min_per_class for label in LABELS)


def main() -> None:
    # options de la ligne de commande (ex. python ml_service/train_model.py --interval 5)
    parser = argparse.ArgumentParser(description="Service ML BUSI (export ONNX)")
    parser.add_argument("--once", action="store_true", help="un seul entrainement puis arret")
    parser.add_argument("--interval", type=int, default=10, help="secondes entre deux verifications")
    parser.add_argument("--retrain-every", type=int, default=6, help="nouvelles images avant re-entrainement")
    parser.add_argument("--min-per-class", type=int, default=2, help="images minimum par classe")
    parser.add_argument("--balance", action="store_true", help="meme nombre d'images par classe (celui de la plus petite)")
    parser.add_argument("--idle-timeout", type=int, default=600,
                        help="arret apres ce nombre de secondes sans nouvelle image")
    args = parser.parse_args()

    # nombre d'images utilisees au dernier entrainement (0 = jamais entraine)
    trained_on = 0
    # nombre d'images vues a la derniere verification, et heure de la derniere nouvelle image
    seen = 0
    last_new_image = time.time()
    print(f"[ML] lecture de {IMAGES_CSV.relative_to(ROOT)}, export vers {MODEL_FILE.relative_to(ROOT)}")
    # une verification par tour ; count() numerote les tours 0, 1, 2... et c'est le "break"
    # plus bas qui arrete la boucle
    for _ in itertools.count():
        df = load_dataset()
        if len(df) != seen:
            seen = len(df)
            last_new_image = time.time()
        # on s'arrete apres un seul tour (--once) ou si plus aucune image n'arrive depuis --idle-timeout s
        running = not args.once and time.time() - last_new_image < args.idle_timeout

        if not can_train(df, args.min_per_class):
            print(f"[ML] {len(df)} image(s), il en faut {args.min_per_class} par classe")
        # premier entrainement, assez de nouvelles images depuis le dernier,
        # ou dernier tour avec des images pas encore utilisees
        elif (trained_on == 0 or len(df) - trained_on >= args.retrain_every
              or (not running and len(df) > trained_on)):
            train(balance(df) if args.balance else df)
            trained_on = len(df)

        if not running:
            break
        time.sleep(args.interval)
    if not args.once:
        print(f"[ML] aucune nouvelle image depuis {args.idle_timeout} s, arret du service")


if __name__ == "__main__":
    main()
