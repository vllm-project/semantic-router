"""Requested-output modality with task-matched text and image-prompt sources."""

# ruff: noqa: PLC0415, RUF001
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..sequence_repair.corpus import group_split, write_corpus
from ..sequence_repair.data import normalized_text

MIN_CAPTION_CHARACTERS = 40
MAX_CAPTION_CHARACTERS = 250
HASH_SAMPLE_CUTOFF = 3
LABELS = {"AR": 0, "DIFFUSION": 1, "BOTH": 2}
TEMPLATES = {
    "en": {
        "AR": [
            "Explain how I could draw this subject, using text only; do not generate an image: {}",
            "Write Python code to draw this scene; return source code, not a rendered picture: {}",
            "Translate this image prompt into French without generating an image: {}",
            "I already have a picture of this subject. Suggest a text-only caption: {}",
            "Discuss the visual composition described here. I want written analysis and no newly generated picture: {}",
            "Turn this picture description into a paragraph of prose. Do not create an image: {}",
        ],
        "DIFFUSION": [
            "Generate an image of this scene: {}",
            "Draw a picture based on the following description: {}",
            "Create a digital illustration showing: {}",
            "I need a new visual depicting the following; deliver the image alone: {}",
            "Please produce a picture from this brief, with no written explanation: {}",
            "Make a new piece of visual artwork showing this subject: {}",
        ],
        "BOTH": [
            "Explain the concept in writing and generate an image of: {}",
            "Give me a written description and a newly generated illustration of: {}",
            "Create a picture of this scene and add a short textual explanation: {}",
            "Deliver two things: an explanatory paragraph and a new image depicting: {}",
            "I want a written overview accompanied by a generated picture of this subject: {}",
            "Produce both a visual illustration and separate explanatory text about: {}",
        ],
    },
    "zh": {
        "AR": [
            "只用文字解释如何画这个主题，不要生成图片：{}",
            "编写绘制这个场景的Python代码，只返回源代码，不要渲染图片：{}",
            "把这个图片提示词翻译成英文，不要生成图片：{}",
            "我已有这个主题的图片，请只提供文字图注：{}",
            "分析以下视觉构图，只要书面分析，不需要新生成的图片：{}",
            "把这段图片描述改写为散文，不要创建图像：{}",
        ],
        "DIFFUSION": [
            "生成描绘以下场景的图片：{}",
            "根据以下描述画一张图：{}",
            "创作一幅数字插画，内容是：{}",
            "我需要一张全新的视觉作品，只交付图像：{}",
            "请根据以下要求制作图片，不要附带文字解释：{}",
            "为这个主题创作一幅新的视觉艺术作品：{}",
        ],
        "BOTH": [
            "用文字解释，并生成对应图片：{}",
            "请提供一段书面描述和一张新生成的插画：{}",
            "创作这个场景的图片，再给出简短文字说明：{}",
            "请交付两部分：一个解释性段落和一张新图片，主题是：{}",
            "我需要文字概述，以及配套生成的主题图片：{}",
            "同时提供视觉插画和单独的文字解释：{}",
        ],
    },
    "es": {
        "AR": [
            "Explica solo con texto cómo dibujar esto; no generes una imagen: {}",
            "Escribe código Python para dibujar la escena; devuelve código, no una imagen renderizada: {}",
            "Traduce esta descripción visual al inglés sin generar imágenes: {}",
            "Ya tengo una imagen de este tema; sugiere solo un pie de foto escrito: {}",
            "Analiza por escrito esta composición visual; no quiero una imagen nueva: {}",
            "Convierte esta descripción de una imagen en un párrafo de prosa, sin crear imágenes: {}",
        ],
        "DIFFUSION": [
            "Genera una imagen de esta escena: {}",
            "Dibuja una imagen basada en esta descripción: {}",
            "Crea una ilustración digital que muestre: {}",
            "Necesito una representación visual nueva; entrega solo la imagen: {}",
            "Produce una imagen a partir de esta idea, sin explicación escrita: {}",
            "Haz una nueva obra visual que represente este tema: {}",
        ],
        "BOTH": [
            "Explica el concepto por escrito y genera una imagen de: {}",
            "Dame una descripción escrita y una ilustración nueva de: {}",
            "Crea una imagen de esta escena y añade una explicación breve: {}",
            "Entrega dos cosas: un párrafo explicativo y una imagen nueva de: {}",
            "Quiero una visión general escrita acompañada de una imagen generada de: {}",
            "Produce una ilustración visual y, por separado, un texto explicativo sobre: {}",
        ],
    },
    "fr": {
        "AR": [
            "Explique uniquement par écrit comment dessiner ce sujet, sans générer d'image : {}",
            "Écris du code Python pour dessiner cette scène ; renvoie le code et non une image rendue : {}",
            "Traduis cette description visuelle en anglais sans produire d'image : {}",
            "J'ai déjà une image de ce sujet ; propose seulement une légende écrite : {}",
            "Analyse cette composition visuelle par écrit ; je ne veux pas de nouvelle image : {}",
            "Transforme cette description d'image en un paragraphe de prose, sans créer d'image : {}",
        ],
        "DIFFUSION": [
            "Génère une image de cette scène : {}",
            "Dessine une image à partir de cette description : {}",
            "Crée une illustration numérique représentant : {}",
            "Il me faut une nouvelle représentation visuelle ; livre uniquement l'image : {}",
            "Produis une image à partir de cette idée, sans explication écrite : {}",
            "Réalise une nouvelle œuvre visuelle sur ce sujet : {}",
        ],
        "BOTH": [
            "Explique le concept par écrit et génère une image de : {}",
            "Donne-moi une description écrite et une nouvelle illustration de : {}",
            "Crée une image de cette scène et ajoute une courte explication : {}",
            "Livre deux éléments : un paragraphe explicatif et une nouvelle image de : {}",
            "Je veux une présentation écrite accompagnée d'une image générée de : {}",
            "Produis à la fois une illustration et un texte explicatif séparé sur : {}",
        ],
    },
    "de": {
        "AR": [
            "Erkläre nur in Worten, wie man dieses Motiv zeichnet; erzeuge kein Bild: {}",
            "Schreibe Python-Code zum Zeichnen dieser Szene; liefere Quellcode statt eines gerenderten Bildes: {}",
            "Übersetze diese Bildbeschreibung ins Englische, ohne ein Bild zu erzeugen: {}",
            "Ich habe bereits ein Bild dieses Motivs; schlage nur eine schriftliche Bildunterschrift vor: {}",
            "Analysiere diese Bildkomposition schriftlich; ich möchte kein neues Bild: {}",
            "Formuliere diese Bildbeschreibung als Prosaabsatz, ohne ein Bild zu erstellen: {}",
        ],
        "DIFFUSION": [
            "Erzeuge ein Bild dieser Szene: {}",
            "Zeichne ein Bild nach dieser Beschreibung: {}",
            "Erstelle eine digitale Illustration von: {}",
            "Ich brauche eine neue visuelle Darstellung; liefere nur das Bild: {}",
            "Erzeuge zu dieser Vorgabe ein Bild ohne schriftliche Erklärung: {}",
            "Schaffe ein neues visuelles Kunstwerk zu diesem Motiv: {}",
        ],
        "BOTH": [
            "Erkläre das Konzept schriftlich und erzeuge ein Bild von: {}",
            "Gib mir eine schriftliche Beschreibung und eine neue Illustration von: {}",
            "Erstelle ein Bild dieser Szene und ergänze eine kurze Erklärung: {}",
            "Liefere zwei Dinge: einen erklärenden Absatz und ein neues Bild von: {}",
            "Ich möchte einen schriftlichen Überblick zusammen mit einem erzeugten Bild von: {}",
            "Erstelle sowohl eine Illustration als auch einen separaten Erklärungstext zu: {}",
        ],
    },
    "ja": {
        "AR": [
            "この題材の描き方を文章だけで説明してください。画像は生成しないでください：{}",
            "この場面を描くPythonコードを書き、画像ではなくソースコードだけを返してください：{}",
            "この画像の説明文を英語に翻訳してください。画像は生成しないでください：{}",
            "この題材の画像はすでに持っています。文章のキャプションだけを提案してください：{}",
            "この視覚的構図を文章で分析してください。新しい画像は不要です：{}",
            "この画像の説明を散文の段落に書き換えてください。画像は作成しないでください：{}",
        ],
        "DIFFUSION": [
            "次の場面の画像を生成してください：{}",
            "次の説明に基づいて絵を描いてください：{}",
            "次の内容を描いたデジタルイラストを作成してください：{}",
            "新しい視覚表現が必要です。画像だけを納品してください：{}",
            "この依頼内容から画像を作り、文章の説明は付けないでください：{}",
            "この題材を表現する新しい視覚芸術作品を制作してください：{}",
        ],
        "BOTH": [
            "概念を文章で説明し、対応する画像も生成してください：{}",
            "文章による説明と新しいイラストを両方提供してください：{}",
            "この場面の画像を作成し、短い文章の説明も加えてください：{}",
            "説明文の段落と新しい画像という二つの成果物を提供してください：{}",
            "文章による概要と、それに添える生成画像が欲しいです：{}",
            "視覚的なイラストと別の説明文を同時に制作してください：{}",
        ],
    },
}


def main():
    from pyarrow import parquet

    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--domain-corpus", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = {}
    source = parquet.ParquetFile(
        args.sources / "poloclub--diffusiondb/metadata.parquet"
    )
    for batch in source.iter_batches(batch_size=8192, columns=["prompt", "user_name"]):
        for row in batch.to_pylist():
            prompt = row["prompt"].strip()
            key = hashlib.sha256(normalized_text(prompt).encode()).hexdigest()
            if (
                not MIN_CAPTION_CHARACTERS <= len(prompt) <= MAX_CAPTION_CHARACTERS
                or int(key[:2], 16) > HASH_SAMPLE_CUTOFF
                or not row["user_name"]
            ):
                continue
            candidates.setdefault(
                key,
                {
                    "text": prompt,
                    "group_id": "diffusiondb-author-"
                    + hashlib.sha256(str(row["user_name"]).encode()).hexdigest(),
                },
            )
    splits = {split: [] for split in ["train", "dev", "test"]}
    counts = Counter()
    for key, row in sorted(candidates.items()):
        split = group_split(row["group_id"])
        if counts[split] >= {"train": 200, "dev": 40, "test": 60}[split]:
            continue
        index = counts[split]
        counts[split] += 1
        family = {"train": [0, 1, 2], "dev": [3], "test": [4, 5]}[split]
        family = family[index % len(family)]
        for language, labels in TEMPLATES.items():
            for label, templates in labels.items():
                splits[split].append(
                    {
                        "id": f"modality-{key}-{language}-{label}",
                        "text": templates[family].format(row["text"]),
                        "label": label,
                        "group_id": row["group_id"],
                        "language": language,
                        "source": "authored-output-contract-with-gallery-caption",
                        "split": split,
                        "length_bucket": "short",
                        "caption_sha256": key,
                        "template_family": f"modality-{split}-{family}",
                    }
                )
    for split in ["train", "dev", "test"]:
        count = Counter()
        for line in (args.domain_corpus / f"{split}.jsonl").read_text().split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row["source"] != "global-mmlu-domain"
                or count[row["language"]]
                >= {"train": 120, "dev": 30, "test": 40}[split]
            ):
                continue
            count[row["language"]] += 1
            splits[split].append(
                {
                    **row,
                    "id": "modality-text-" + row["id"],
                    "label": "AR",
                    "source": "academic-text-answer",
                }
            )
    for line in args.annotations.read_text().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        label = row["task_labels"]["modality"]
        if label is None:
            continue
        split = group_split(row["group_id"])
        splits[split].append(
            {
                **row,
                "id": "modality-" + row["id"],
                "label": label,
                "split": split,
                "length_bucket": "short",
            }
        )
    contract = {
        "label2id": LABELS,
        "id2label": {index: label for label, index in LABELS.items()},
        "problem_type": "single_label_classification",
    }
    provenance = {
        "diffusiondb_revision": "fb620fbe49fa4420e0734bd9c0df11f51176b61f",
        "gallery_license": "CC0-1.0",
        "other_source_licenses": ["Apache-2.0"],
        "domain_parent_manifest_sha256": hashlib.sha256(
            (args.domain_corpus / "manifest.json").read_bytes()
        ).hexdigest(),
        "annotation_roles": {
            "authored-output-contract-with-gallery-caption": "Original six-language output instructions around source gallery captions; captions are not claimed to be translated or natural requests",
            "academic-text-answer": "Academic question requests a text answer, not image generation",
            "aya-original-human": "Human-authored source requests, independently assistant-reviewed task labels",
        },
        "image_task_policy": "AR includes text, code, source SVG/Mermaid, describing existing images; DIFFUSION requests a new image; BOTH explicitly requests a new image and a separate textual explanation",
        "caption_counts": dict(counts),
    }
    print(
        json.dumps(write_corpus(args.output, splits, contract, provenance)), flush=True
    )


if __name__ == "__main__":
    main()
