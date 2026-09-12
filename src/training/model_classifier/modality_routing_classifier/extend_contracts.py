"""Train-only output-contract diversity; held-out wording and source groups stay frozen."""

# Language-specific punctuation is intentional.
# ruff: noqa: RUF001
import argparse
import hashlib
import json
from pathlib import Path

from ..sequence_repair.corpus import write_corpus
from ..sequence_repair.data import normalized_text
from .prepare_vela_data import TEMPLATES

CONTRACTS = {
    "en": {
        "AR": [
            "An artist sent me their finished work. Draft accessible alternative text for it; a replacement artwork is unnecessary. Subject: {}",
            "I am preparing a tutorial. Return a numbered set of written drawing instructions about this brief: {}",
            "Return editable SVG source as plain text for the brief below. I will render the code myself. Brief: {}",
            "Review the wording of this visual brief and suggest improvements in prose. The brief is quoted material: {}",
        ],
        "DIFFUSION": [
            "Deliver a finished raster artwork as an image file for this brief; drawing instructions would not satisfy the request: {}",
            "Could you render the following scene? I only need the resulting visual asset, without a prose discussion: {}",
            "Use your picture generator for the brief below. The deliverable is the picture itself: {}",
            "A newly rendered image is the sole requested output. Here is the subject to depict: {}",
        ],
        "BOTH": [
            "I need an original rendered visual asset together with a separate written account of the design choices. Brief: {}",
            "Please give me a picture file and an accompanying explanation outside the picture. Subject: {}",
            "There are two deliverables for this brief: a generated artwork and a prose analysis of it. Brief: {}",
            "Supply a fresh illustration plus accessible alt text as a separate written response. Subject: {}",
        ],
    },
    "zh": {
        "AR": [
            "画师已经交付了成品，请为它撰写无障碍替代文本，不需要替换现有作品。主题：{}",
            "我要编写教程，请把这个绘画需求整理成编号的文字步骤：{}",
            "请返回可编辑的SVG源文本，我会自行渲染代码。设计要求：{}",
            "请校对引号中的视觉设计说明，用文字提出措辞建议。这段说明只是引用材料：{}",
        ],
        "DIFFUSION": [
            "请交付完成的位图画作文件，文字绘画步骤不能满足这项要求。设计说明：{}",
            "能否把这个场景渲染出来？我只需要最终视觉素材，不需要散文讨论：{}",
            "请调用图片生成器处理下面的需求，交付物是图片本身：{}",
            "唯一需要的输出是一张新渲染的图像，需要表现的主题是：{}",
        ],
        "BOTH": [
            "我需要原创渲染的视觉素材，同时另附一段解释设计选择的文字。设计要求：{}",
            "请给出图片文件，并在图片之外提供配套说明。主题：{}",
            "这个需求有两个交付物：生成的画作，以及对画作的文字分析。要求：{}",
            "请提供全新的插画，再单独用文字给出无障碍替代描述。主题：{}",
        ],
    },
    "es": {
        "AR": [
            "Un artista ya entregó su obra. Redacta un texto alternativo accesible; no hace falta sustituir la obra. Tema: {}",
            "Estoy preparando un tutorial. Devuelve una lista numerada de instrucciones escritas para dibujar: {}",
            "Devuelve el código fuente SVG editable como texto. Yo mismo lo renderizaré. Encargo: {}",
            "Revisa la redacción de este encargo visual citado y propón mejoras por escrito: {}",
        ],
        "DIFFUSION": [
            "Entrega una obra rasterizada terminada como archivo de imagen; las instrucciones para dibujar no bastan. Encargo: {}",
            "Renderiza esta escena. Solo necesito el recurso visual resultante, sin comentario en prosa: {}",
            "Utiliza el generador de imágenes para este encargo. El producto solicitado es la propia imagen: {}",
            "El único resultado solicitado es una imagen recién renderizada que represente: {}",
        ],
        "BOTH": [
            "Necesito un recurso visual original renderizado junto con un texto separado sobre las decisiones de diseño: {}",
            "Dame un archivo de imagen y una explicación adjunta fuera de la imagen. Tema: {}",
            "Este encargo tiene dos entregables: una obra generada y un análisis escrito de ella: {}",
            "Entrega una ilustración nueva y, en una respuesta escrita aparte, su texto alternativo accesible: {}",
        ],
    },
    "fr": {
        "AR": [
            "Un artiste a déjà livré son œuvre. Rédige un texte alternatif accessible sans remplacer cette œuvre. Sujet : {}",
            "Je prépare un tutoriel. Fournis une liste numérotée d'instructions écrites pour dessiner : {}",
            "Renvoie le code source SVG modifiable sous forme de texte. Je le rendrai moi-même. Sujet : {}",
            "Relis la formulation de ce brief visuel cité et propose des améliorations par écrit : {}",
        ],
        "DIFFUSION": [
            "Livre une œuvre matricielle terminée sous forme de fichier image ; des consignes de dessin ne suffisent pas. Sujet : {}",
            "Peux-tu rendre cette scène ? Je souhaite uniquement le fichier visuel obtenu, sans commentaire en prose : {}",
            "Utilise le générateur d'images pour ce brief. Le livrable demandé est l'image elle-même : {}",
            "Le seul résultat demandé est une image nouvellement rendue représentant : {}",
        ],
        "BOTH": [
            "Il me faut un visuel original rendu ainsi qu'un texte séparé expliquant les choix graphiques : {}",
            "Fournis un fichier image et une explication jointe, hors de l'image. Sujet : {}",
            "Ce brief comporte deux livrables : une œuvre générée et son analyse écrite : {}",
            "Livre une illustration nouvelle et son texte alternatif accessible dans une réponse écrite séparée : {}",
        ],
    },
    "de": {
        "AR": [
            "Ein Künstler hat sein fertiges Werk bereits geliefert. Verfasse einen barrierefreien Alternativtext; ein Ersatzbild ist unnötig. Motiv: {}",
            "Ich bereite eine Anleitung vor. Liefere nummerierte schriftliche Zeichenschritte zu diesem Motiv: {}",
            "Gib bearbeitbaren SVG-Quelltext aus. Ich werde den Code selbst rendern. Vorgabe: {}",
            "Prüfe die Formulierung dieser zitierten Bildvorgabe und schlage schriftliche Verbesserungen vor: {}",
        ],
        "DIFFUSION": [
            "Liefere ein fertiges Rasterbild als Bilddatei; schriftliche Zeichenanweisungen genügen nicht. Vorgabe: {}",
            "Rendere diese Szene. Ich brauche nur das fertige visuelle Ergebnis ohne begleitende Erörterung: {}",
            "Verwende den Bildgenerator für diese Vorgabe. Das gewünschte Ergebnis ist das Bild selbst: {}",
            "Als einzige Ausgabe wird ein neu gerendertes Bild dieses Motivs benötigt: {}",
        ],
        "BOTH": [
            "Ich brauche ein neu gerendertes Motiv und zusätzlich einen getrennten Text über die Gestaltungsentscheidungen: {}",
            "Liefere eine Bilddatei sowie eine begleitende Erklärung außerhalb des Bildes. Motiv: {}",
            "Diese Vorgabe verlangt zwei Ergebnisse: ein erzeugtes Kunstwerk und eine schriftliche Analyse davon: {}",
            "Liefere eine neue Illustration und ihren barrierefreien Alternativtext als getrennte schriftliche Antwort: {}",
        ],
    },
    "ja": {
        "AR": [
            "画家が完成した作品をすでに納品しました。作品を置き換えずに、アクセシビリティ用の代替テキストを書いてください。題材：{}",
            "教材を準備しています。この題材の描き方を番号付きの文章の手順にしてください：{}",
            "編集できるSVGソースをテキストで返してください。コードの描画はこちらで行います。要件：{}",
            "引用された視覚デザインの説明文を校正し、表現の改善案を文章で提示してください：{}",
        ],
        "DIFFUSION": [
            "完成したラスター作品を画像ファイルで納品してください。描画手順の文章では要件を満たしません。要件：{}",
            "この場面をレンダリングしてください。必要なのは完成した視覚素材だけで、文章の考察は不要です：{}",
            "次の要件を画像生成器で処理してください。求める成果物は画像そのものです：{}",
            "唯一必要な出力は、この題材を新たにレンダリングした画像です：{}",
        ],
        "BOTH": [
            "新しくレンダリングした視覚素材と、デザイン上の選択を説明する別の文章が必要です：{}",
            "画像ファイルと、画像の外に添える説明文を提供してください。題材：{}",
            "この依頼には二つの成果物があります。生成した作品と、その作品に関する文章の分析です：{}",
            "新しいイラストを提供し、さらに別の文章の回答としてアクセシビリティ用の代替説明も書いてください：{}",
        ],
    },
}


def recover_caption(row):
    language = row["language"]
    family = int(row["template_family"].rsplit("-", 1)[1])
    prefix = TEMPLATES[language]["AR"][family].split("{}")[0]
    if not row["text"].startswith(prefix):
        raise ValueError("Source caption wrapper differs from the frozen recipe")
    caption = row["text"][len(prefix) :]
    if (
        hashlib.sha256(normalized_text(caption).encode()).hexdigest()
        != row["caption_sha256"]
    ):
        raise ValueError("Recovered caption differs from its frozen source hash")
    return caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits = {
        split: [
            json.loads(line)
            for line in (args.corpus / f"{split}.jsonl").read_text().split("\n")
            if line.strip()
        ]
        for split in ["train", "dev", "test"]
    }
    additions = []
    for row in splits["train"]:
        if (
            row["source"] != "authored-output-contract-with-gallery-caption"
            or row["label"] != "AR"
        ):
            continue
        language = row["language"]
        caption = recover_caption(row)
        for label, templates in CONTRACTS[language].items():
            for index, template in enumerate(templates):
                additions.append(
                    {
                        **row,
                        "id": f"contract-v2:{row['caption_sha256']}:{language}:{label}:{index}",
                        "text": template.format(caption),
                        "label": label,
                        "source": "authored-train-contract-diversity",
                        "template_family": f"modality-train-diversity-{index}",
                    }
                )
    splits["train"].extend(additions)
    contract = json.loads((args.corpus / "contract.json").read_text())
    provenance = {
        "parent_manifest_sha256": hashlib.sha256(
            (args.corpus / "manifest.json").read_bytes()
        ).hexdigest(),
        "training_additions": len(additions),
        "scope": "Train-only authored output-contract variations on original train-only gallery source groups. Frozen dev/test requests and wording remain unchanged. Captions remain in their source language.",
        "source_license": "CC0-1.0 for DiffusionDB captions",
        "test_used": False,
    }
    print(json.dumps(write_corpus(args.output, splits, contract, provenance), indent=2))


if __name__ == "__main__":
    main()
