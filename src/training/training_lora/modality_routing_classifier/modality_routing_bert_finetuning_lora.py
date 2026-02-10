"""
Modality Routing Classification Fine-tuning with Enhanced LoRA Training
Uses PEFT (Parameter-Efficient Fine-Tuning) with LoRA adapters for efficient modality routing.

🎯 **PURPOSE**: Train a model to classify user prompt intent into response modality:
   - AR: Text-only response via autoregressive LLM (e.g., Llama, Qwen)
   - DIFFUSION: Image generation via diffusion model (e.g., Flux, SDXL)
   - BOTH: Hybrid response requiring both text explanation AND image generation

📚 **DATASETS USED** (publicly available):
   DIFFUSION class (image generation intent):
   - DiffusionDB (poloclub/diffusiondb): 1.8M unique Stable Diffusion prompts (ACL 2023 Best Paper HM)
     https://huggingface.co/datasets/poloclub/diffusiondb

   AR class (text-only intent):
   - OASST2 (OpenAssistant/oasst2): 135K instruction-following conversations
   - Alpaca (tatsu-lab/alpaca): 52K instruction-following examples (Stanford, 2023)
   - Dolly (databricks/databricks-dolly-15k): 15K instruction-following examples

   BOTH class (mixed modality intent):
   - Synthetic generation via vLLM endpoint (configurable)
   - Seed examples with diverse domain coverage

Usage:
    # Train with recommended parameters
    python modality_routing_bert_finetuning_lora.py --mode train --model mmbert-32k --epochs 8 --lora-rank 32 --max-samples 5000

    # Train with vLLM synthesis for BOTH class
    python modality_routing_bert_finetuning_lora.py --mode train --vllm-endpoint http://localhost:8000/v1 --synthesize-both 2000

    # Test inference with trained LoRA model
    python modality_routing_bert_finetuning_lora.py --mode test --model-path lora_modality_router_mmbert-32k_r32_model

    # Quick training test (for debugging)
    python modality_routing_bert_finetuning_lora.py --mode train --epochs 1 --max-samples 200

Supported models:
    - mmbert-32k: mmBERT-32K YaRN - 32K context, multilingual (RECOMMENDED)
    - mmbert-base: mmBERT base model (149M parameters, 1800+ languages, 8K context)
    - modernbert-base: ModernBERT base model (149M parameters, latest architecture)
    - bert-base-uncased: Standard BERT base model (110M parameters, most stable)
    - roberta-base: RoBERTa base model (125M parameters, better context understanding)

Key Features:
    - LoRA (Low-Rank Adaptation) for 3-class modality routing
    - 99%+ parameter reduction (only ~0.02% trainable parameters)
    - Uses real, peer-reviewed datasets for AR and DIFFUSION classes
    - Synthetic data generation via vLLM endpoint for BOTH class
    - Auto-merge functionality: Generates both LoRA adapters and Rust-compatible models
    - Multi-architecture support: Dynamic target_modules configuration for all models
    - Comprehensive evaluation metrics (accuracy, weighted F1, per-class precision/recall)
    - Production-ready: Robust error handling and validation throughout
"""

import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Import common LoRA utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_lora_utils import (
    clear_gpu_memory,
    create_lora_config,
    log_memory_usage,
    resolve_model_path,
    set_gpu_device,
    setup_logging,
)

# Setup logging
logger = setup_logging()

# ============================================================================
# Label Definitions
# ============================================================================

# 3-class modality routing labels
LABEL_AR = "AR"  # Autoregressive text-only response
LABEL_DIFFUSION = "DIFFUSION"  # Image generation via diffusion model
LABEL_BOTH = "BOTH"  # Hybrid: text explanation + image generation

MODALITY_LABELS = [LABEL_AR, LABEL_DIFFUSION, LABEL_BOTH]


def create_tokenizer_for_model(model_path: str, base_model_name: str = None):
    """
    Create tokenizer with model-specific configuration.

    Args:
        model_path: Path to load tokenizer from
        base_model_name: Optional base model name for configuration
    """
    model_identifier = base_model_name or model_path

    if "roberta" in model_identifier.lower():
        logger.info("Using RoBERTa tokenizer with add_prefix_space=True")
        return AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    else:
        return AutoTokenizer.from_pretrained(model_path)


# ============================================================================
# vLLM Synthetic Data Generation for BOTH class
# ============================================================================


class VLLMSynthesizer:
    """
    Generate synthetic 'BOTH' class prompts using a vLLM endpoint.

    The BOTH class (prompts requiring both text + image response) is the hardest
    to find natural data for. This synthesizer uses a vLLM-served LLM to generate
    diverse, realistic prompts that would benefit from both modalities.
    """

    # Seed examples covering diverse domains for few-shot prompting.
    # Includes both EXPLICIT requests ("show me", "illustrate") and IMPLICIT ones
    # where the visual need is implied by the topic (e.g., "How do I tile a bathroom?").
    SEED_BOTH_EXAMPLES = [
        # ---- EXPLICIT: Education / Science ----
        "Explain how photosynthesis works and show me a diagram of the process",
        "Teach me about the water cycle with illustrations for each stage",
        "Describe the structure of DNA and generate a visual representation",
        "Explain how a neural network works and draw the architecture",
        "What is plate tectonics? Show me how the plates move",
        "Explain the human digestive system and illustrate each organ",
        "How does an internal combustion engine work? Show me a cross-section",
        "Explain mitosis vs meiosis with a comparison diagram",
        "How does the electromagnetic spectrum work? Visualize the wavelengths",
        "Teach me the periodic table organization with a color-coded chart",
        # ---- EXPLICIT: Cooking / Food ----
        "How do I julienne vegetables? Show me the knife technique",
        "Explain the Maillard reaction and show what it looks like on steak",
        "What is the difference between types of pasta? Show me each shape",
        "How do I properly fold a dumpling? Show me the steps visually",
        "Explain how sourdough fermentation works and show the stages",
        "How do I temper chocolate? Illustrate the temperature curve",
        # ---- EXPLICIT: Design / Architecture ----
        "Explain the golden ratio in design and show examples of it in architecture",
        "What is brutalist architecture? Show me some iconic examples",
        "Explain color theory and generate a complementary color wheel",
        "Describe Art Deco style and create an example illustration",
        "What are the principles of responsive web design? Show layout examples",
        # ---- EXPLICIT: Medical / Health ----
        "Explain how vaccines work and illustrate the immune response",
        "What does a healthy vs unhealthy lung look like? Show a comparison",
        "Describe the stages of wound healing with visual diagrams",
        "How does the knee joint work? Show me the anatomy",
        "Explain blood types and create a compatibility chart",
        # ---- EXPLICIT: Engineering / Tech ----
        "Explain how a bridge suspension system works and show the forces",
        "What is a heat exchanger? Draw a diagram of the flow",
        "Explain OAuth 2.0 authorization flow and show a sequence diagram",
        "How does a CPU pipeline work? Illustrate the stages",
        "Describe how HTTPS encryption works and show the handshake process",
        "Explain microservices vs monolith architecture with a comparison diagram",
        "How does garbage collection work in Java? Visualize the heap generations",
        # ---- EXPLICIT: Geography / History ----
        "What did ancient Rome look like? Describe and show a reconstruction",
        "Explain how tectonic plates formed the Himalayas with diagrams",
        "Describe the architecture of Egyptian pyramids with cross-section views",
        # ---- EXPLICIT: Math / Abstract ----
        "Explain the Pythagorean theorem with a visual proof",
        "What is a Fourier transform? Show me the time vs frequency domain",
        "Explain how sorting algorithms work and visualize each step",
        "Describe fractals and generate a Mandelbrot set visualization",
        # ---- EXPLICIT: Everyday / Practical ----
        "How do I tie a bowline knot? Show me each step",
        "Explain how to read a topographic map with an example",
        "What is the proper form for a deadlift? Show the positioning",
        "How do I wire a 3-way light switch? Show the circuit diagram",
        "Explain how to read sheet music and show the notation",
        # ---- IMPLICIT: visual is needed but not explicitly asked ----
        # These are crucial for the model to learn -- the user doesn't say
        # "show me" but the response clearly benefits from images.
        "How do I tile a bathroom floor?",
        "What's the correct way to do a French braid?",
        "How do I replace a toilet fill valve?",
        "Walk me through building a raised garden bed",
        "How do I adjust the derailleur on my bike?",
        "What are the different types of wood joints?",
        "How does a sewing machine thread path work?",
        "Explain how to set up a freshwater aquarium",
        "How do I install crown molding in a room with corners?",
        "What's the proper way to sharpen a chef's knife on a whetstone?",
        "How do I perform CPR on an adult?",
        "Explain how to parallel park step by step",
        "How do I build a basic Arduino circuit with an LED?",
        "What does each yoga sun salutation pose look like?",
        "How do I apply a screen protector without bubbles?",
        "Walk me through assembling IKEA PAX wardrobe",
        "How do I check and replace brake pads on a car?",
        "Explain the different crochet stitches for beginners",
        "How do I properly set a formal dinner table?",
        "What is the correct batting stance in baseball?",
        "How do I fold a fitted sheet neatly?",
        "Explain the stages of butterfly metamorphosis",
        "How do I create a waterfall braid?",
        "What does a normal EKG reading look like vs abnormal?",
        "How do I build a campfire in different configurations?",
        "Explain the differences between serif and sans-serif typefaces",
        "How do I create a macramé plant hanger?",
        "What are the hand signals for directing traffic?",
        "How do I identify common poisonous mushrooms vs edible ones?",
        "Explain the orbit patterns of our solar system's planets",
    ]

    SYNTHESIS_SYSTEM_PROMPT = """You are a dataset generator for training an AI modality router.
Your task is to generate realistic user prompts that would REQUIRE BOTH a text explanation AND an image/diagram/visualization in the response.

These prompts should be diverse across domains and should sound like natural user queries.
Each prompt should clearly benefit from having both textual explanation AND visual content.

IMPORTANT RULES:
- Generate prompts that genuinely need BOTH text and image (not just text, not just image)
- Cover diverse domains: science, cooking, design, medicine, engineering, history, math, DIY, etc.
- Make prompts sound natural - as a real user would type them
- Vary the phrasing: "show me", "illustrate", "draw", "visualize", "with a diagram", "what does X look like", etc.
- Some prompts should implicitly need visuals without explicitly asking (e.g., "How do I tile a bathroom?" implies visual steps)
- Each prompt should be 1-2 sentences max
- Do NOT repeat domains or topics within a batch"""

    SYNTHESIS_USER_PROMPT = """Generate {count} diverse user prompts that would require BOTH text explanation AND image/visual content in the response.

Here are some examples for reference:
{examples}

Now generate {count} NEW and DIVERSE prompts (do not repeat the examples above).
Output ONLY the prompts, one per line, with no numbering or bullets."""

    def __init__(self, endpoint: str, model: str = None, api_key: str = None):
        """
        Initialize the vLLM synthesizer.

        Args:
            endpoint: vLLM OpenAI-compatible API endpoint (e.g., http://localhost:8000/v1)
            model: Model name served by vLLM (auto-detected if None)
            api_key: Optional API key for authentication
        """
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")

    def _get_model_name(self) -> str:
        """Auto-detect the model name from the vLLM endpoint."""
        if self.model:
            return self.model

        try:
            import requests

            resp = requests.get(
                f"{self.endpoint}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models:
                self.model = models[0]["id"]
                logger.info(f"Auto-detected vLLM model: {self.model}")
                return self.model
        except Exception as e:
            logger.warning(f"Failed to auto-detect model: {e}")

        # Fallback
        self.model = "default"
        return self.model

    def synthesize(
        self,
        count: int,
        batch_size: int = 50,
        temperature: float = 0.9,
        max_retries: int = 3,
    ) -> List[str]:
        """
        Synthesize 'BOTH' class prompts using the vLLM endpoint.

        Args:
            count: Total number of prompts to generate
            batch_size: Number of prompts per API call
            temperature: Sampling temperature (higher = more diverse)
            max_retries: Maximum retries per batch on failure

        Returns:
            List of synthesized prompts
        """
        import requests

        model_name = self._get_model_name()
        all_prompts = set()  # Use set for deduplication
        batches_needed = (count + batch_size - 1) // batch_size

        logger.info(
            f"Synthesizing {count} BOTH-class prompts via vLLM ({batches_needed} batches)..."
        )

        for batch_idx in range(batches_needed):
            remaining = count - len(all_prompts)
            if remaining <= 0:
                break

            current_batch_size = min(batch_size, remaining)

            # Sample diverse seed examples for few-shot context
            example_sample = random.sample(
                self.SEED_BOTH_EXAMPLES,
                min(8, len(self.SEED_BOTH_EXAMPLES)),
            )
            examples_text = "\n".join(f"- {ex}" for ex in example_sample)

            user_prompt = self.SYNTHESIS_USER_PROMPT.format(
                count=current_batch_size,
                examples=examples_text,
            )

            for retry in range(max_retries):
                try:
                    resp = requests.post(
                        f"{self.endpoint}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model_name,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": self.SYNTHESIS_SYSTEM_PROMPT,
                                },
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": temperature,
                            "max_tokens": 4096,
                            "top_p": 0.95,
                        },
                        timeout=120,
                    )
                    resp.raise_for_status()

                    content = resp.json()["choices"][0]["message"]["content"]
                    lines = [
                        line.strip()
                        for line in content.strip().split("\n")
                        if line.strip()
                        and len(line.strip()) > 15
                        and len(line.strip()) < 500
                    ]

                    # Clean up: remove numbering, bullets, quotes
                    cleaned = []
                    for line in lines:
                        # Remove common prefixes: "1.", "- ", "* ", '"'
                        line = line.lstrip("0123456789.-)*] ").strip()
                        line = line.strip('"').strip("'").strip()
                        if line and len(line) > 15:
                            cleaned.append(line)

                    all_prompts.update(cleaned)
                    logger.info(
                        f"  Batch {batch_idx + 1}/{batches_needed}: "
                        f"generated {len(cleaned)} prompts (total: {len(all_prompts)})"
                    )
                    break

                except Exception as e:
                    logger.warning(
                        f"  Batch {batch_idx + 1} retry {retry + 1}/{max_retries}: {e}"
                    )
                    if retry < max_retries - 1:
                        time.sleep(2**retry)  # Exponential backoff

        result = list(all_prompts)[:count]
        logger.info(f"Synthesized {len(result)} BOTH-class prompts")
        return result


# ============================================================================
# Dataset Loading
# ============================================================================


class ModalityRoutingDataset:
    """
    Dataset class for modality routing classification.

    Loads and combines data from multiple sources for 3-class classification:
    - AR (autoregressive/text-only): From conversational/instruction datasets
    - DIFFUSION (image generation): From text-to-image prompt datasets
    - BOTH (text + image): Synthetic + seed examples

    References:
    - DiffusionDB: "DiffusionDB: A Large-scale Prompt Gallery Dataset" (ACL 2023)
      https://poloclub.github.io/diffusiondb/
    - OASST: "OpenAssistant Conversations" (NeurIPS 2023)
    - Alpaca: "Stanford Alpaca" (Stanford, 2023)
    """

    def __init__(
        self,
        vllm_endpoint: str = None,
        vllm_model: str = None,
        vllm_api_key: str = None,
    ):
        """
        Initialize the dataset loader.

        Args:
            vllm_endpoint: Optional vLLM endpoint for synthetic BOTH class generation
            vllm_model: Optional model name for vLLM
            vllm_api_key: Optional API key for vLLM
        """
        self.label2id = {label: idx for idx, label in enumerate(MODALITY_LABELS)}
        self.id2label = {idx: label for idx, label in enumerate(MODALITY_LABELS)}

        self.vllm_synthesizer = None
        if vllm_endpoint:
            self.vllm_synthesizer = VLLMSynthesizer(
                endpoint=vllm_endpoint,
                model=vllm_model,
                api_key=vllm_api_key,
            )

    # ----------------------------------------------------------------
    # DIFFUSION class loaders
    # ----------------------------------------------------------------

    def _load_diffusion_prompts(
        self, max_samples: int, global_seen: set = None
    ) -> List[str]:
        """
        Load image generation prompts from multiple text-to-image prompt datasets.

        Sources (tried in order until target is met):
        1. Gustavosta/Stable-Diffusion-Prompts: ~80K curated SD prompts
        2. FredZhang7/stable-diffusion-prompts-2.47M: 2.47M prompts (streaming)
        3. succinctly/midjourney-prompts: Midjourney prompts
        4. Falah/image_generation_prompts_SDXL: SDXL prompts

        Args:
            max_samples: Maximum prompts to collect
            global_seen: Shared dedup set across all loaders (prevents cross-class leaks)
        """
        logger.info("Loading diffusion/image-generation prompts...")
        prompts = []
        seen = global_seen if global_seen is not None else set()

        # Datasets to try in order of quality/availability
        datasets_to_try = [
            {
                "name": "Gustavosta/Stable-Diffusion-Prompts",
                "split": "train",
                "text_field": "Prompt",
                "description": "Curated SD prompts (~80K)",
            },
            {
                "name": "FredZhang7/stable-diffusion-prompts-2.47M",
                "split": "train",
                "text_field": "text",
                "streaming": True,
                "description": "2.47M SD prompts (streaming)",
            },
            {
                "name": "succinctly/midjourney-prompts",
                "split": "train",
                "text_field": "text",
                "description": "Midjourney prompts",
            },
            {
                "name": "Falah/image_generation_prompts_SDXL",
                "split": "train",
                "text_field": "prompts",
                "description": "SDXL prompts",
            },
        ]

        for ds_config in datasets_to_try:
            if len(prompts) >= max_samples:
                break

            ds_name = ds_config["name"]
            text_field = ds_config["text_field"]
            use_streaming = ds_config.get("streaming", False)

            try:
                logger.info(f"  Loading '{ds_name}' ({ds_config['description']})...")

                if use_streaming:
                    dataset = load_dataset(
                        ds_name, split=ds_config["split"], streaming=True
                    )
                else:
                    dataset = load_dataset(ds_name, split=ds_config["split"])

                loaded_count = 0
                for item in dataset:
                    if len(prompts) >= max_samples:
                        break
                    prompt = item.get(text_field, "")
                    prompt_clean = prompt.strip() if prompt else ""
                    if (
                        prompt_clean
                        and len(prompt_clean) > 10
                        and len(prompt_clean) < 1000
                        and prompt_clean not in seen
                    ):
                        prompts.append(prompt_clean)
                        seen.add(prompt_clean)
                        loaded_count += 1

                logger.info(
                    f"  Loaded {loaded_count} prompts from '{ds_name}' "
                    f"(total: {len(prompts)})"
                )
            except Exception as e:
                logger.warning(f"  Failed to load '{ds_name}': {e}")
                continue

        logger.info(f"Diffusion prompts total: {len(prompts)} unique prompts")
        return prompts

    def _generate_fallback_diffusion_prompts(self, count: int) -> List[str]:
        """Generate fallback diffusion-style prompts from templates."""
        templates = [
            "A {style} painting of {subject} in {setting}",
            "{subject}, {style} style, {lighting} lighting, detailed",
            "Generate an image of {subject} with {style} aesthetic",
            "Create a {style} illustration of {subject}",
            "A photorealistic {subject} in {setting}, professional photography",
            "{subject}, digital art, trending on artstation, {style}",
            "An anime-style {subject} with {lighting} lighting",
            "A {style} portrait of {subject}, highly detailed",
            "Fantasy {subject} in a {setting}, {style} art",
            "A logo design featuring {subject}, minimal, clean",
            "A watercolor painting of {subject} in {setting}",
            "Concept art of {subject}, {style}, cinematic",
            "A vintage poster of {subject}, retro {style}",
            "Abstract {style} composition inspired by {subject}",
            "A 3D render of {subject}, {lighting} lighting, octane render",
        ]
        values = {
            "style": [
                "impressionist",
                "cyberpunk",
                "art deco",
                "minimalist",
                "surrealist",
                "steampunk",
                "baroque",
                "pop art",
                "ukiyo-e",
                "gothic",
            ],
            "subject": [
                "a majestic lion",
                "a futuristic city",
                "a serene forest",
                "a warrior princess",
                "a dragon",
                "an astronaut",
                "a coffee shop",
                "ocean waves",
                "a mountain landscape",
                "a samurai",
            ],
            "setting": [
                "a mystical forest",
                "outer space",
                "an ancient temple",
                "a neon-lit street",
                "a snowy mountain",
                "underwater",
                "a desert oasis",
                "a Victorian garden",
                "a futuristic lab",
                "a cozy library",
            ],
            "lighting": [
                "golden hour",
                "dramatic",
                "soft ambient",
                "neon",
                "moonlit",
                "studio",
                "cinematic",
                "volumetric",
            ],
        }

        prompts = set()
        max_attempts = count * 20
        attempts = 0
        while len(prompts) < count and attempts < max_attempts:
            template = random.choice(templates)
            result = template
            for key, vals in values.items():
                placeholder = "{" + key + "}"
                if placeholder in result:
                    result = result.replace(placeholder, random.choice(vals))
            if result not in prompts:
                prompts.add(result)
            attempts += 1

        return list(prompts)

    # ----------------------------------------------------------------
    # AR class loaders
    # ----------------------------------------------------------------

    def _load_oasst_prompts(self, max_samples: int) -> List[str]:
        """
        Load text-only conversational prompts from OpenAssistant (OASST2).

        OASST2 contains 135K instruction-following conversations in 35+ languages.
        We extract only the initial user prompts (prompter role).

        Source: https://huggingface.co/datasets/OpenAssistant/oasst2
        """
        logger.info("Loading OASST2 prompts (text-only conversational)...")
        prompts = []
        seen = set()
        try:
            dataset = load_dataset(
                "OpenAssistant/oasst2", split="train", streaming=True
            )
            for item in dataset:
                if len(prompts) >= max_samples:
                    break
                # Only take root-level prompter messages (initial user queries)
                role = item.get("role", "")
                parent_id = item.get("parent_id")
                text = item.get("text", "")

                if (
                    role == "prompter"
                    and parent_id is None
                    and text
                    and len(text) > 10
                    and len(text) < 1000
                    and text not in seen
                ):
                    # Filter out prompts that look like image generation requests
                    text_lower = text.lower()
                    image_keywords = [
                        "generate an image",
                        "create an image",
                        "draw me",
                        "draw a",
                        "make a picture",
                        "generate a photo",
                        "create a picture",
                        "design a logo",
                        "make an illustration",
                        "generate art",
                        "create art",
                        "paint me",
                    ]
                    if not any(kw in text_lower for kw in image_keywords):
                        prompts.append(text.strip())
                        seen.add(text)

            logger.info(f"Loaded {len(prompts)} prompts from OASST2")
        except Exception as e:
            logger.warning(f"Failed to load OASST2: {e}")
        return prompts

    def _load_alpaca_prompts(self, max_samples: int) -> List[str]:
        """
        Load text-only instructions from Stanford Alpaca.

        Alpaca contains 52K instruction-following examples.
        We filter for prompts that are clearly text-only tasks.

        Source: https://huggingface.co/datasets/tatsu-lab/alpaca
        """
        logger.info("Loading Alpaca prompts (text-only instructions)...")
        prompts = []
        seen = set()
        try:
            dataset = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
            for item in dataset:
                if len(prompts) >= max_samples:
                    break
                instruction = item.get("instruction", "")
                inp = item.get("input", "")

                # Combine instruction + input for context
                text = instruction.strip()
                if inp and inp.strip():
                    text = f"{instruction.strip()}\n{inp.strip()}"

                if text and len(text) > 10 and len(text) < 500 and text not in seen:
                    # Filter out image-related prompts
                    text_lower = text.lower()
                    image_keywords = [
                        "generate an image",
                        "draw",
                        "illustrate",
                        "paint",
                        "sketch",
                        "design a logo",
                        "create a picture",
                    ]
                    if not any(kw in text_lower for kw in image_keywords):
                        prompts.append(text)
                        seen.add(text)

            logger.info(f"Loaded {len(prompts)} prompts from Alpaca")
        except Exception as e:
            logger.warning(f"Failed to load Alpaca: {e}")
        return prompts

    def _load_dolly_prompts(self, max_samples: int) -> List[str]:
        """
        Load text-only instructions from Databricks Dolly.

        Dolly contains 15K instruction-following examples across multiple categories.
        We use categories that are clearly text-only: open_qa, closed_qa,
        information_extraction, summarization, classification.

        Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k
        """
        logger.info("Loading Dolly prompts (text-only instructions)...")
        prompts = []
        seen = set()

        # Text-only categories
        text_categories = {
            "open_qa",
            "closed_qa",
            "information_extraction",
            "summarization",
            "classification",
            "creative_writing",
            "brainstorming",
        }

        try:
            dataset = load_dataset(
                "databricks/databricks-dolly-15k", split="train", streaming=True
            )
            for item in dataset:
                if len(prompts) >= max_samples:
                    break

                category = item.get("category", "")
                instruction = item.get("instruction", "")

                if (
                    category in text_categories
                    and instruction
                    and len(instruction) > 10
                    and len(instruction) < 500
                    and instruction not in seen
                ):
                    prompts.append(instruction.strip())
                    seen.add(instruction)

            logger.info(f"Loaded {len(prompts)} prompts from Dolly")
        except Exception as e:
            logger.warning(f"Failed to load Dolly: {e}")
        return prompts

    def _load_lmsys_prompts(self, max_samples: int) -> List[str]:
        """
        Load text-only prompts from LMSYS-Chat-1M.

        Contains 1M real-world LLM conversations from Chatbot Arena.
        These are overwhelmingly text-only requests.

        Source: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
        Note: Requires agreement to dataset license on HuggingFace.
        """
        logger.info("Loading LMSYS-Chat-1M prompts (text-only real conversations)...")
        prompts = []
        seen = set()
        try:
            dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
            for item in dataset:
                if len(prompts) >= max_samples:
                    break

                conversation = item.get("conversation", [])
                if not conversation:
                    continue

                # Extract the first user message
                first_msg = conversation[0]
                if first_msg.get("role") != "human":
                    continue

                text = first_msg.get("content", "")
                if text and len(text) > 10 and len(text) < 1000 and text not in seen:
                    # Filter out image generation requests
                    text_lower = text.lower()
                    image_keywords = [
                        "generate an image",
                        "create an image",
                        "draw me",
                        "draw a",
                        "make a picture",
                        "generate a photo",
                    ]
                    if not any(kw in text_lower for kw in image_keywords):
                        prompts.append(text.strip())
                        seen.add(text)

            logger.info(f"Loaded {len(prompts)} prompts from LMSYS-Chat-1M")
        except Exception as e:
            logger.warning(
                f"Failed to load LMSYS-Chat-1M (may require license agreement): {e}"
            )
        return prompts

    def _load_ultrachat_prompts(self, max_samples: int) -> List[str]:
        """
        Load text-only prompts from UltraChat.

        UltraChat contains 1.5M multi-turn dialogues covering diverse topics.
        Great source for conversational text-only (AR) prompts.

        Source: https://huggingface.co/datasets/stingning/ultrachat
        """
        logger.info("Loading UltraChat prompts (text-only instructions)...")
        prompts = []
        seen = set()
        try:
            dataset = load_dataset("stingning/ultrachat", split="train", streaming=True)
            for item in dataset:
                if len(prompts) >= max_samples:
                    break

                data = item.get("data", [])
                if not data or not isinstance(data, list) or len(data) == 0:
                    continue

                # First element is the user's opening message
                text = data[0] if isinstance(data[0], str) else ""
                text = text.strip()
                if text and len(text) > 15 and len(text) < 500 and text not in seen:
                    # Filter out image/visual requests
                    text_lower = text.lower()
                    visual_keywords = [
                        "generate an image",
                        "create an image",
                        "draw ",
                        "make a picture",
                        "illustrate",
                        "paint ",
                        "stable diffusion",
                        "dall-e",
                        "midjourney",
                    ]
                    if not any(kw in text_lower for kw in visual_keywords):
                        prompts.append(text)
                        seen.add(text)

            logger.info(f"Loaded {len(prompts)} prompts from UltraChat")
        except Exception as e:
            logger.warning(f"Failed to load UltraChat: {e}")
        return prompts

    def _load_wildchat_prompts(
        self, max_samples: int, global_seen: set = None
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Mine WildChat (1M real ChatGPT conversations) for all three classes.

        Returns (ar_prompts, diffusion_prompts, both_prompts) tuple.
        Uses strict regex patterns to classify real user intent.

        Source: https://huggingface.co/datasets/allenai/WildChat
        """
        import re

        logger.info("Mining WildChat for modality-labeled real user prompts...")

        seen = global_seen if global_seen is not None else set()
        ar_prompts = []
        diffusion_prompts = []
        both_prompts = []

        # Strict diffusion patterns - must clearly request image creation
        diffusion_re = re.compile(
            r"(?:^|\b)(?:"
            r"(?:create|generate|make|produce)\s+(?:an?\s+)?(?:image|picture|photo|illustration|artwork|portrait|logo|icon|poster|banner|thumbnail|wallpaper)"
            r"|(?:can you|please|could you)\s+(?:create|generate|make|draw|paint)\s+(?:an?\s+)?(?:image|picture|illustration|artwork)"
            r"|(?:write|create|make|give)\s+(?:me\s+)?(?:an?\s+)?(?:stable\s+diffusion|dall-?e|midjourney)\s+prompt"
            r"|(?:text[\s-]?to[\s-]?image)\s+prompt"
            r"|make\s+(?:a\s+)?(?:picture|photo)\s+of"
            r")",
            re.IGNORECASE,
        )

        # Strict BOTH patterns - explicitly request text explanation + visual output
        both_re = re.compile(
            r"(?:"
            r"(?:explain|describe|write\s+about).*(?:and\s+(?:include|add|create|show|provide|generate)\s+(?:an?\s+)?(?:image|diagram|picture|illustration|visual|chart|figure|infographic))"
            r"|(?:explain|describe|write).*(?:with\s+(?:an?\s+)?(?:image|diagram|illustration|visual|chart|figure))"
            r"|(?:create|make|generate)\s+(?:an?\s+)?(?:infographic|poster|presentation)\s+(?:about|explaining|describing)"
            r")",
            re.IGNORECASE,
        )

        try:
            dataset = load_dataset("allenai/WildChat", split="train", streaming=True)

            scanned = 0
            target_per_class = max_samples
            max_scan = 300000  # Scan up to 300K conversations

            for item in dataset:
                if scanned >= max_scan:
                    break
                scanned += 1

                # English only, non-toxic
                if item.get("language") != "English" or item.get("toxic", False):
                    continue

                conv = item.get("conversation", [])
                if not conv:
                    continue

                # Get first user message
                first_msg = ""
                for msg in conv:
                    if msg.get("role") == "user":
                        first_msg = msg.get("content", "")
                        break

                if not first_msg or len(first_msg) < 15 or len(first_msg) > 500:
                    continue

                text = first_msg.strip()
                if text in seen:
                    continue

                is_diffusion = bool(diffusion_re.search(text))
                is_both = bool(both_re.search(text))

                if is_both and len(both_prompts) < target_per_class:
                    both_prompts.append(text)
                    seen.add(text)
                elif (
                    is_diffusion
                    and not is_both
                    and len(diffusion_prompts) < target_per_class
                ):
                    diffusion_prompts.append(text)
                    seen.add(text)
                elif (
                    not is_diffusion
                    and not is_both
                    and len(ar_prompts) < target_per_class
                ):
                    ar_prompts.append(text)
                    seen.add(text)

                # Early exit if all classes are full
                if (
                    len(ar_prompts) >= target_per_class
                    and len(diffusion_prompts) >= target_per_class
                    and len(both_prompts) >= target_per_class
                ):
                    break

            logger.info(
                f"WildChat mined ({scanned} scanned): "
                f"AR={len(ar_prompts)}, DIFFUSION={len(diffusion_prompts)}, "
                f"BOTH={len(both_prompts)}"
            )
        except Exception as e:
            logger.warning(f"Failed to mine WildChat: {e}")

        return ar_prompts, diffusion_prompts, both_prompts

    def _load_interleavedbench_prompts(
        self, max_samples: int, global_seen: set = None
    ) -> List[str]:
        """
        Load prompts from InterleavedBench that explicitly require both text+image output.

        InterleavedBench (EMNLP 2024) is the first benchmark for interleaved text-image
        generation. Tasks like report_generation, education_generation, marketing_generation,
        and storytelling_generation all require producing both text and images.

        Source: https://huggingface.co/mqliu/InterleavedBench
        """
        logger.info(
            "Loading InterleavedBench prompts (interleaved text+image tasks)..."
        )
        prompts = []
        seen = global_seen if global_seen is not None else set()

        # Task types that inherently require BOTH text and image output
        both_tasks = {
            "report_generation",
            "education_generation",
            "marketing_generation",
            "storytelling_generation",
            "wikihow_next_step",
        }

        try:
            from huggingface_hub import hf_hub_download
            import json

            path = hf_hub_download(
                "mqliu/InterleavedBench",
                "interleaved_bench.json",
                repo_type="model",
            )
            with open(path) as f:
                data = json.load(f)

            for item in data:
                if len(prompts) >= max_samples:
                    break

                task = item.get("task_name", "")
                if task not in both_tasks:
                    continue

                # Extract the human prompt from conversations
                convs = item.get("conversations", [])
                for msg in convs:
                    if msg.get("from") == "human":
                        text = msg.get("value", "")
                        # Clean up image placeholders
                        text = text.replace("<image>", "").replace("<BEGIN>", "")
                        text = " ".join(text.split())  # normalize whitespace
                        text = text.strip()

                        if (
                            text
                            and len(text) > 20
                            and len(text) < 1000
                            and text not in seen
                        ):
                            prompts.append(text)
                            seen.add(text)
                        break

            logger.info(f"Loaded {len(prompts)} prompts from InterleavedBench")
        except Exception as e:
            logger.warning(f"Failed to load InterleavedBench: {e}")

        return prompts

    def _load_parti_prompts(
        self, max_samples: int, global_seen: set = None
    ) -> List[str]:
        """
        Load Parti Prompts - Google's curated text-to-image evaluation prompts.

        1,632 prompts across 12 categories (Animals, Artifacts, People, etc.)
        with difficulty levels. High-quality, diverse image generation prompts.

        Source: https://huggingface.co/datasets/nateraw/parti-prompts
        """
        logger.info("Loading Parti Prompts (curated T2I evaluation prompts)...")
        prompts = []
        seen = global_seen if global_seen is not None else set()

        try:
            dataset = load_dataset("nateraw/parti-prompts", split="train")
            for item in dataset:
                if len(prompts) >= max_samples:
                    break
                text = item.get("Prompt", "").strip()
                if text and len(text) > 3 and len(text) < 500 and text not in seen:
                    prompts.append(text)
                    seen.add(text)

            logger.info(f"Loaded {len(prompts)} prompts from Parti Prompts")
        except Exception as e:
            logger.warning(f"Failed to load Parti Prompts: {e}")

        return prompts

    def _load_fal_prompts(self, max_samples: int, global_seen: set = None) -> List[str]:
        """
        Load fal/image-generation-prompts - structured T2I prompts with categories.

        800 prompts across 7 categories and 32 subcategories.
        High-quality, detailed image generation prompts.

        Source: https://huggingface.co/datasets/fal/image-generation-prompts
        """
        logger.info("Loading fal image-generation prompts...")
        prompts = []
        seen = global_seen if global_seen is not None else set()

        try:
            dataset = load_dataset("fal/image-generation-prompts", split="train")
            for item in dataset:
                if len(prompts) >= max_samples:
                    break
                text = item.get("prompt", "").strip()
                if text and len(text) > 10 and len(text) < 1500 and text not in seen:
                    prompts.append(text)
                    seen.add(text)

            logger.info(f"Loaded {len(prompts)} prompts from fal image-gen")
        except Exception as e:
            logger.warning(f"Failed to load fal prompts: {e}")

        return prompts

    def _generate_fallback_ar_prompts(self, count: int) -> List[str]:
        """Generate fallback text-only prompts from templates."""
        templates = [
            "What is {topic}?",
            "Explain {topic} in simple terms",
            "How does {concept} work?",
            "Write a summary of {topic}",
            "What are the pros and cons of {topic}?",
            "Compare {thing1} and {thing2}",
            "Write a Python function to {task}",
            "Help me debug this code: {code_desc}",
            "Translate this to {language}: {phrase}",
            "What is the capital of {country}?",
            "Write an email to {recipient} about {topic}",
            "Summarize the key points of {topic}",
            "What are best practices for {topic}?",
            "How do I {action}?",
            "Calculate {expression}",
            "Write a poem about {theme}",
            "Create a business plan for {business}",
            "What happened in {event}?",
            "List the top 10 {category}",
            "Explain the difference between {thing1} and {thing2}",
        ]
        values = {
            "topic": [
                "machine learning",
                "quantum computing",
                "climate change",
                "blockchain",
                "CRISPR gene editing",
                "supply chain management",
            ],
            "concept": [
                "TCP/IP",
                "encryption",
                "compilers",
                "garbage collection",
                "photosynthesis",
                "evolution",
            ],
            "thing1": ["Python", "React", "SQL", "REST", "microservices", "Docker"],
            "thing2": [
                "JavaScript",
                "Angular",
                "NoSQL",
                "GraphQL",
                "monoliths",
                "Kubernetes",
            ],
            "task": [
                "sort a list",
                "read a CSV file",
                "make an API call",
                "parse JSON",
                "connect to a database",
            ],
            "code_desc": [
                "a recursive function with a stack overflow",
                "async/await not working",
                "memory leak in a loop",
            ],
            "language": ["Spanish", "French", "Japanese", "German", "Chinese"],
            "phrase": ["Hello, how are you?", "Where is the library?", "Thank you"],
            "country": ["Japan", "Brazil", "France", "Australia", "Egypt", "Canada"],
            "recipient": ["my boss", "a client", "my team", "HR department"],
            "action": [
                "set up a CI/CD pipeline",
                "deploy to AWS",
                "configure nginx",
            ],
            "expression": ["integral of x^2", "derivative of sin(x)", "sqrt(144)"],
            "theme": ["the ocean", "autumn", "technology"],
            "business": [
                "a coffee shop",
                "a SaaS startup",
                "an online tutoring service",
            ],
            "event": [
                "the industrial revolution",
                "the moon landing",
                "the dot-com bubble",
            ],
            "category": [
                "programming languages",
                "design patterns",
                "machine learning frameworks",
            ],
        }

        prompts = set()
        max_attempts = count * 20
        attempts = 0
        while len(prompts) < count and attempts < max_attempts:
            template = random.choice(templates)
            result = template
            for key, vals in values.items():
                placeholder = "{" + key + "}"
                if placeholder in result:
                    result = result.replace(placeholder, random.choice(vals))
            if result not in prompts:
                prompts.add(result)
            attempts += 1

        return list(prompts)

    # ----------------------------------------------------------------
    # BOTH class loaders
    # ----------------------------------------------------------------

    def _get_both_seed_examples(self) -> List[str]:
        """Return seed examples for the BOTH class."""
        return list(VLLMSynthesizer.SEED_BOTH_EXAMPLES)

    def _generate_fallback_both_prompts(self, count: int) -> List[str]:
        """
        Generate fallback BOTH-class prompts from templates.

        Uses TWO categories of templates for diversity:
        1. EXPLICIT: Directly ask for visual content ("show me", "illustrate", "diagram")
        2. IMPLICIT: Topics that inherently need visuals without saying so
           (e.g., "How do I tile a floor?" clearly needs step-by-step images)

        This prevents the model from learning a shallow heuristic like
        "if 'show me' is present → BOTH" and instead forces it to learn
        that certain TOPICS inherently require visual responses.
        """
        # ---- EXPLICIT templates (user asks for visuals) ----
        explicit_templates = [
            "Explain {topic} and show me a diagram",
            "What does {thing} look like? Describe and illustrate it",
            "Teach me about {topic} with visual examples",
            "How does {process} work? Include a diagram of each step",
            "Describe {thing} and generate an image of it",
            "Show me the {technique} technique and explain each step visually",
            "What is {concept}? Draw a visual representation",
            "Explain the anatomy of {body_part} with labeled diagrams",
            "Compare {thing1} and {thing2} with side-by-side images",
            "Walk me through {process} step by step with illustrations",
            "Create an infographic explaining {topic}",
            "Describe and visualize the architecture of {system}",
            "How do I {diy_task}? Show me what each step looks like",
            "Explain {concept} and create a chart showing the relationships",
            "What are the stages of {process}? Illustrate each one",
            "Generate a diagram of {topic} and explain the key components",
            "I need to understand {concept} - can you explain with visuals?",
            "Give me a visual guide to {technique} with explanations",
            "Show me the difference between {thing1} and {thing2} and explain why",
            "Help me learn {topic} - I need both text and pictures",
        ]

        # ---- IMPLICIT templates (visuals needed but not explicitly asked) ----
        implicit_templates = [
            "How do I {diy_task}?",
            "Walk me through {technique} step by step",
            "I need to {diy_task}, what are the steps?",
            "Can you help me understand {spatial_concept}?",
            "What are the different types of {visual_category}?",
            "How do I identify {identification_target}?",
            "Teach me {technique} for beginners",
            "What does {anatomy_topic} look like in detail?",
            "How do I set up {setup_task}?",
            "What's the correct way to do {physical_skill}?",
            "Guide me through {process} from start to finish",
            "I want to learn {craft} - where do I start?",
            "Help me understand the layout of {spatial_thing}",
            "What do the warning signs of {medical_sign} look like?",
            "How is {constructed_thing} put together?",
        ]

        values = {
            "topic": [
                "the water cycle",
                "how vaccines work",
                "plate tectonics",
                "photosynthesis",
                "neural networks",
                "the solar system",
                "DNA replication",
                "color theory",
                "ocean currents",
                "how electricity reaches your home",
                "cloud formation",
                "the nitrogen cycle",
                "how earthquakes happen",
                "the rock cycle",
                "how stars form and die",
                "the immune system",
                "how transistors work",
                "the krebs cycle",
                "continental drift",
            ],
            "thing": [
                "a healthy coral reef",
                "the inside of a volcano",
                "a cross-section of the Earth",
                "the Milky Way galaxy",
                "a Tesla coil in action",
                "the interior of a cell",
                "a black hole accretion disk",
                "a submarine interior",
                "the layers of the atmosphere",
                "a glacial formation",
            ],
            "process": [
                "bread baking",
                "3D printing",
                "steel manufacturing",
                "wound healing",
                "cloud formation",
                "OAuth authentication",
                "beer brewing",
                "silk weaving",
                "concrete curing",
                "glass blowing",
                "cheese making",
                "bone fracture repair",
                "semiconductor fabrication",
                "paper recycling",
            ],
            "technique": [
                "julienne cutting",
                "French braid",
                "origami crane folding",
                "watercolor wet-on-wet",
                "soldering",
                "knot tying",
                "dovetail joinery",
                "pottery throwing",
                "cross-stitching",
                "cake decorating with fondant",
                "TIG welding",
                "leather tooling",
                "calligraphy",
                "macramé",
            ],
            "concept": [
                "the golden ratio",
                "Fourier transform",
                "supply and demand",
                "the electromagnetic spectrum",
                "recursion",
                "Ohm's law in circuits",
                "the doppler effect",
                "binary search trees",
                "gradient descent",
                "convolution in image processing",
                "Bayesian inference",
            ],
            "body_part": [
                "the human heart",
                "the knee joint",
                "the eye",
                "the brain",
                "the respiratory system",
                "the inner ear",
                "the spine",
                "the shoulder rotator cuff",
                "the liver",
            ],
            "thing1": [
                "serif fonts",
                "CPU",
                "oil painting",
                "bridge types",
                "AC motors",
                "hardwood",
                "LCD",
                "induction cooktop",
            ],
            "thing2": [
                "sans-serif fonts",
                "GPU",
                "watercolor painting",
                "arch types",
                "DC motors",
                "softwood",
                "OLED",
                "gas cooktop",
            ],
            "system": [
                "a microservices application",
                "a nuclear reactor",
                "a suspension bridge",
                "a spacecraft",
                "a wastewater treatment plant",
                "a wind turbine",
                "an ethernet network",
                "a hydroelectric dam",
            ],
            # ---- Values for IMPLICIT templates ----
            "diy_task": [
                "tile a bathroom floor",
                "build a raised garden bed",
                "change brake pads",
                "set up a home network",
                "install a ceiling fan",
                "build a bookshelf",
                "replace a toilet fill valve",
                "install a dimmer switch",
                "lay laminate flooring",
                "build a fence gate",
                "install a garbage disposal",
                "paint a room with clean edges",
                "hang heavy shelves on drywall",
                "replace a car battery",
                "assemble a PC from parts",
                "install a dishwasher",
                "patch a hole in drywall",
                "install a drip irrigation system",
            ],
            "spatial_concept": [
                "how a 4-stroke engine works",
                "the structure of a Gothic cathedral",
                "how a lock mechanism operates",
                "the layout of a circuit board",
                "how plumbing works in a house",
                "how a jet engine produces thrust",
                "the structure of a coral reef ecosystem",
            ],
            "visual_category": [
                "wood joints used in furniture",
                "roof truss designs",
                "guitar chord shapes",
                "suture patterns in surgery",
                "cloud formations and what they mean",
                "gem cuts and their names",
                "electrical outlet types around the world",
                "embroidery stitches",
            ],
            "identification_target": [
                "common poisonous mushrooms vs edible ones",
                "skin cancer warning signs",
                "types of spider webs",
                "different bird species in my backyard",
                "types of wood grain patterns",
                "different welding defects",
                "common plant diseases from leaf symptoms",
            ],
            "anatomy_topic": [
                "the layers of human skin",
                "the inside of a tooth",
                "the structure of a muscle fiber",
                "a cross-section of the eye",
                "the chambers of the heart",
            ],
            "setup_task": [
                "a freshwater aquarium",
                "a home recording studio",
                "a hydroponic growing system",
                "a sourdough starter",
                "a basic Arduino circuit",
                "a home darkroom for film",
            ],
            "physical_skill": [
                "a deadlift with proper form",
                "a handstand progression",
                "CPR on an adult",
                "the Heimlich maneuver",
                "parallel parking",
                "a proper golf swing",
            ],
            "craft": [
                "pottery on a wheel",
                "leather working",
                "stained glass making",
                "blacksmithing basics",
                "bookbinding by hand",
                "screen printing",
            ],
            "spatial_thing": [
                "a typical car engine bay",
                "the New York subway system",
                "a commercial airplane cockpit",
                "a standard basketball court",
            ],
            "medical_sign": [
                "a melanoma vs a benign mole",
                "an infected wound",
                "dehydration in a child",
                "a broken vs sprained ankle",
            ],
            "constructed_thing": [
                "a violin",
                "a mechanical watch",
                "a residential HVAC system",
                "a telescope",
                "a lithium-ion battery",
                "a diesel engine",
            ],
        }

        # Mix explicit and implicit templates ~50/50 for balanced learning
        all_templates = explicit_templates + implicit_templates

        prompts = set()
        max_attempts = count * 30  # More attempts since we have more templates
        attempts = 0
        while len(prompts) < count and attempts < max_attempts:
            template = random.choice(all_templates)
            result = template
            for key, vals in values.items():
                placeholder = "{" + key + "}"
                if placeholder in result:
                    result = result.replace(placeholder, random.choice(vals))
            if result not in prompts:
                prompts.add(result)
            attempts += 1

        return list(prompts)

    # ----------------------------------------------------------------
    # Main dataset assembly
    # ----------------------------------------------------------------

    def load_datasets(
        self,
        max_samples: int = 6000,
        synthesize_both: int = 0,
    ) -> Tuple[List[str], List[int]]:
        """
        Load and combine datasets for 3-class modality routing classification.

        Strategy:
        - Target balanced classes (~1/3 each)
        - DIFFUSION: DiffusionDB (real user prompts to Stable Diffusion)
        - AR: OASST2 + Alpaca + Dolly (text-only conversations/instructions)
        - BOTH: Seed examples + vLLM synthesis + template fallbacks

        Args:
            max_samples: Maximum total samples across all classes
            synthesize_both: Number of BOTH-class samples to synthesize via vLLM

        Returns:
            Tuple of (texts, labels) where labels are 0 (AR), 1 (DIFFUSION), 2 (BOTH)
        """
        logger.info(
            f"Loading modality routing datasets (target: {max_samples} samples)..."
        )

        samples_per_class = max_samples // 3
        source_counts = {}

        # Global dedup set: prevents the same prompt from appearing in multiple classes
        global_seen = set()

        # ==================================
        # Mine WildChat for all 3 classes first (real user prompts)
        # ==================================
        wc_max = samples_per_class // 4  # Use WildChat for ~25% of each class
        wc_ar, wc_diffusion, wc_both = self._load_wildchat_prompts(
            wc_max, global_seen=global_seen
        )
        source_counts["WildChat_AR"] = len(wc_ar)
        source_counts["WildChat_DIFF"] = len(wc_diffusion)
        source_counts["WildChat_BOTH"] = len(wc_both)

        # ==================================
        # DIFFUSION class
        # ==================================
        diffusion_texts = []

        # Source 1: WildChat real image-gen requests
        diffusion_texts.extend(wc_diffusion)

        # Source 2: Multiple diffusion prompt datasets (SD, Midjourney, etc.)
        diffusion_prompts = self._load_diffusion_prompts(
            samples_per_class - len(diffusion_texts), global_seen=global_seen
        )
        diffusion_texts.extend(diffusion_prompts)
        source_counts["Diffusion_SD"] = len(diffusion_prompts)

        # Source 3: Parti Prompts (Google curated T2I eval)
        if len(diffusion_texts) < samples_per_class:
            parti_prompts = self._load_parti_prompts(
                samples_per_class - len(diffusion_texts), global_seen=global_seen
            )
            diffusion_texts.extend(parti_prompts)
            source_counts["Parti"] = len(parti_prompts)

        # Source 4: fal image-generation-prompts
        if len(diffusion_texts) < samples_per_class:
            fal_prompts = self._load_fal_prompts(
                samples_per_class - len(diffusion_texts), global_seen=global_seen
            )
            diffusion_texts.extend(fal_prompts)
            source_counts["fal"] = len(fal_prompts)

        # Fallback: Template-based generation
        if len(diffusion_texts) < samples_per_class:
            fallback_count = samples_per_class - len(diffusion_texts)
            fallback_prompts = self._generate_fallback_diffusion_prompts(fallback_count)
            prev = len(diffusion_texts)
            for p in fallback_prompts:
                if p not in global_seen and len(diffusion_texts) < samples_per_class:
                    diffusion_texts.append(p)
                    global_seen.add(p)
            source_counts["Diffusion_fallback"] = len(diffusion_texts) - prev

        diffusion_texts = diffusion_texts[:samples_per_class]
        logger.info(f"DIFFUSION class: {len(diffusion_texts)} samples")

        # ==================================
        # AR class
        # ==================================
        ar_texts = []
        ar_per_source = samples_per_class // 4  # Split among 4 sources

        # Source 0: WildChat real text-only conversations
        for p in wc_ar:
            ar_texts.append(p)

        # Source 1: OASST2
        prev_count = len(ar_texts)
        oasst_prompts = self._load_oasst_prompts(ar_per_source)
        for p in oasst_prompts:
            if p not in global_seen:
                ar_texts.append(p)
                global_seen.add(p)
        source_counts["OASST2"] = len(ar_texts) - prev_count

        # Source 2: Alpaca
        prev_count = len(ar_texts)
        alpaca_prompts = self._load_alpaca_prompts(ar_per_source)
        for p in alpaca_prompts:
            if p not in global_seen:
                ar_texts.append(p)
                global_seen.add(p)
        source_counts["Alpaca"] = len(ar_texts) - prev_count

        # Source 3: Dolly
        prev_count = len(ar_texts)
        dolly_prompts = self._load_dolly_prompts(ar_per_source)
        for p in dolly_prompts:
            if p not in global_seen:
                ar_texts.append(p)
                global_seen.add(p)
        source_counts["Dolly"] = len(ar_texts) - prev_count

        # Source 4: UltraChat (large instruction dataset)
        if len(ar_texts) < samples_per_class:
            prev_count = len(ar_texts)
            ultrachat_prompts = self._load_ultrachat_prompts(
                samples_per_class - len(ar_texts)
            )
            for p in ultrachat_prompts:
                if p not in global_seen and len(ar_texts) < samples_per_class:
                    ar_texts.append(p)
                    global_seen.add(p)
            source_counts["UltraChat"] = len(ar_texts) - prev_count

        # Source 5: LMSYS (if still short)
        if len(ar_texts) < samples_per_class:
            prev_count = len(ar_texts)
            lmsys_prompts = self._load_lmsys_prompts(samples_per_class - len(ar_texts))
            for p in lmsys_prompts:
                if p not in global_seen:
                    ar_texts.append(p)
                    global_seen.add(p)
            source_counts["LMSYS"] = len(ar_texts) - prev_count

        # Fallback: Template-based generation
        if len(ar_texts) < samples_per_class:
            fallback_count = samples_per_class - len(ar_texts)
            fallback_prompts = self._generate_fallback_ar_prompts(fallback_count)
            prev_count = len(ar_texts)
            for p in fallback_prompts:
                if p not in global_seen and len(ar_texts) < samples_per_class:
                    ar_texts.append(p)
                    global_seen.add(p)
            source_counts["AR_fallback"] = len(ar_texts) - prev_count

        ar_texts = ar_texts[:samples_per_class]
        logger.info(f"AR class: {len(ar_texts)} samples")

        # ==================================
        # BOTH class
        # ==================================
        both_texts = []
        both_real_count = 0  # Track non-template samples for quality ratio

        # Source 0: WildChat real multimodal requests
        for p in wc_both:
            both_texts.append(p)
        both_real_count = len(both_texts)

        # Source 1: InterleavedBench (gold-standard text+image tasks)
        interleaved_prompts = self._load_interleavedbench_prompts(
            samples_per_class, global_seen=global_seen
        )
        # Prompts are already deduped against global_seen inside the loader
        both_texts.extend(interleaved_prompts)
        both_real_count += len(interleaved_prompts)
        source_counts["InterleavedBench"] = len(interleaved_prompts)

        # Source 2: Seed examples (curated)
        seed_added = 0
        seed_examples = self._get_both_seed_examples()
        for p in seed_examples:
            if p not in global_seen:
                both_texts.append(p)
                global_seen.add(p)
                seed_added += 1
        both_real_count += seed_added
        source_counts["BOTH_seeds"] = seed_added

        # vLLM synthesis (if endpoint configured)
        if self.vllm_synthesizer and synthesize_both > 0:
            synth_count = min(synthesize_both, samples_per_class - len(both_texts))
            if synth_count > 0:
                synthesized = self.vllm_synthesizer.synthesize(synth_count)
                prev_count = len(both_texts)
                for p in synthesized:
                    if p not in global_seen and len(both_texts) < samples_per_class:
                        both_texts.append(p)
                        global_seen.add(p)
                synth_added = len(both_texts) - prev_count
                both_real_count += synth_added
                source_counts["BOTH_vllm_synth"] = synth_added

        # Template-based fallback to reach target
        if len(both_texts) < samples_per_class:
            fallback_count = samples_per_class - len(both_texts)
            fallback_prompts = self._generate_fallback_both_prompts(fallback_count)
            prev_count = len(both_texts)
            for p in fallback_prompts:
                if p not in global_seen and len(both_texts) < samples_per_class:
                    both_texts.append(p)
                    global_seen.add(p)
            source_counts["BOTH_fallback"] = len(both_texts) - prev_count

        both_texts = both_texts[:samples_per_class]
        logger.info(f"BOTH class: {len(both_texts)} samples")

        # ---- QUALITY VALIDATION ----
        template_ratio = 1.0 - (both_real_count / max(len(both_texts), 1))
        if template_ratio > 0.5:
            logger.warning(
                f"⚠ BOTH class is {template_ratio * 100:.0f}% template-generated "
                f"({both_real_count} real / {len(both_texts)} total). "
                f"Model may overfit on template patterns. "
                f"Strongly recommend using --vllm-endpoint to synthesize diverse data "
                f"(e.g., --vllm-endpoint http://localhost:8000/v1 --synthesize-both 2000)"
            )

        # ==================================
        # Combine all classes
        # ==================================
        texts = []
        labels = []

        for text in ar_texts:
            texts.append(text)
            labels.append(self.label2id[LABEL_AR])

        for text in diffusion_texts:
            texts.append(text)
            labels.append(self.label2id[LABEL_DIFFUSION])

        for text in both_texts:
            texts.append(text)
            labels.append(self.label2id[LABEL_BOTH])

        # Shuffle
        combined = list(zip(texts, labels))
        random.seed(42)
        random.shuffle(combined)
        texts, labels = zip(*combined)
        texts, labels = list(texts), list(labels)

        # ---- CLASS BALANCE VALIDATION ----
        class_counts = {}
        for label_id in labels:
            label_name = self.id2label[label_id]
            class_counts[label_name] = class_counts.get(label_name, 0) + 1

        logger.info(f"Total dataset: {len(texts)} samples")
        for label_name, count in sorted(class_counts.items()):
            pct = count / len(texts) * 100
            logger.info(f"  {label_name}: {count} ({pct:.1f}%)")

        # Check for severe imbalance (any class <20% or >50%)
        for label_name, count in class_counts.items():
            pct = count / len(texts) * 100
            if pct < 20:
                logger.warning(
                    f"⚠ Class '{label_name}' is severely underrepresented "
                    f"({pct:.1f}%). Consider increasing data for this class."
                )
            elif pct > 50:
                logger.warning(
                    f"⚠ Class '{label_name}' dominates the dataset "
                    f"({pct:.1f}%). Consider balancing."
                )

        sources_str = ", ".join(f"{k}={v}" for k, v in source_counts.items())
        logger.info(f"  Sources: {sources_str}")

        return texts, labels

    def prepare_datasets(
        self,
        max_samples: int = 6000,
        synthesize_both: int = 0,
    ):
        """
        Prepare train/validation/test datasets for modality routing.

        Args:
            max_samples: Maximum total samples
            synthesize_both: Number of BOTH-class samples to synthesize via vLLM

        Returns:
            Dictionary with 'train', 'validation', 'test' splits
        """
        texts, labels = self.load_datasets(max_samples, synthesize_both)

        # Split: 70% train, 15% validation, 15% test
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels
        )

        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts,
            temp_labels,
            test_size=0.5,
            random_state=42,
            stratify=temp_labels,
        )

        logger.info("Dataset splits:")
        logger.info(f"  Train: {len(train_texts)}")
        logger.info(f"  Validation: {len(val_texts)}")
        logger.info(f"  Test: {len(test_texts)}")

        return {
            "train": (train_texts, train_labels),
            "validation": (val_texts, val_labels),
            "test": (test_texts, test_labels),
        }


def create_modality_routing_dataset(
    max_samples: int = 6000,
    synthesize_both: int = 0,
    vllm_endpoint: str = None,
    vllm_model: str = None,
    vllm_api_key: str = None,
):
    """
    Create the modality routing dataset.

    Args:
        max_samples: Maximum total samples
        synthesize_both: Number of BOTH-class prompts to synthesize via vLLM
        vllm_endpoint: vLLM endpoint URL for synthesis
        vllm_model: Model name for vLLM
        vllm_api_key: API key for vLLM
    """
    dataset_loader = ModalityRoutingDataset(
        vllm_endpoint=vllm_endpoint,
        vllm_model=vllm_model,
        vllm_api_key=vllm_api_key,
    )
    datasets = dataset_loader.prepare_datasets(max_samples, synthesize_both)

    train_texts, train_labels = datasets["train"]
    val_texts, val_labels = datasets["validation"]

    # Convert to the format expected by our training
    sample_data = []
    for text, label in zip(train_texts + val_texts, train_labels + val_labels):
        sample_data.append({"text": text, "label": label})

    logger.info(f"Created dataset with {len(sample_data)} samples")
    logger.info(f"Label mapping: {dataset_loader.label2id}")

    return sample_data, dataset_loader.label2id, dataset_loader.id2label


# ============================================================================
# LoRA Rank Recommendation
# ============================================================================


def recommend_lora_rank(
    num_train_samples: int,
    num_classes: int = 3,
    user_rank: int = None,
) -> int:
    """
    Recommend LoRA rank based on training data volume to avoid under/overfitting.

    Heuristic: aim for a param-to-sample ratio between 10:1 and 100:1.
    For mmBERT (22 layers, 4 modules, hidden_size=768):
      trainable_params ≈ 22 * 4 * 2 * 768 * rank = 135,168 * rank

    Guidelines:
        ≤2K samples  → rank 8  (~1.1M params, ~550:1 ratio — tight, regularize hard)
        ≤6K samples  → rank 16 (~2.2M params, ~370:1 ratio — good)
        ≤20K samples → rank 32 (~4.3M params, ~215:1 ratio — good)
        ≤50K samples → rank 48 (~6.5M params, ~130:1 ratio — good)
        >50K samples → rank 64 (~8.6M params, ≤170:1 ratio — good)

    If user explicitly sets a rank, we respect it but warn if it looks off.

    Args:
        num_train_samples: Number of training samples
        num_classes: Number of output classes
        user_rank: User-specified rank (None = auto-select)

    Returns:
        Recommended LoRA rank
    """
    # Auto-scale thresholds
    if num_train_samples <= 2000:
        auto_rank = 8
    elif num_train_samples <= 6000:
        auto_rank = 16
    elif num_train_samples <= 20000:
        auto_rank = 32
    elif num_train_samples <= 50000:
        auto_rank = 48
    else:
        auto_rank = 64

    if user_rank is not None:
        # User explicitly set rank — respect it but warn on mismatch
        approx_params = 135_168 * user_rank  # Rough estimate for mmBERT
        ratio = approx_params / max(num_train_samples, 1)

        if ratio > 800:
            logger.warning(
                f"⚠ LoRA rank {user_rank} yields ~{approx_params / 1e6:.1f}M params "
                f"for {num_train_samples} train samples (ratio {ratio:.0f}:1). "
                f"High overfitting risk! Recommended rank: {auto_rank}. "
                f"Proceeding with rank={user_rank} — consider increasing weight_decay "
                f"and using early stopping."
            )
        elif ratio < 20:
            logger.warning(
                f"⚠ LoRA rank {user_rank} yields ~{approx_params / 1e6:.1f}M params "
                f"for {num_train_samples} train samples (ratio {ratio:.0f}:1). "
                f"Model may underfit. Recommended rank: {auto_rank}."
            )
        else:
            logger.info(
                f"LoRA rank {user_rank}: ~{approx_params / 1e6:.1f}M params, "
                f"{num_train_samples} samples, ratio {ratio:.0f}:1 — OK"
            )
        return user_rank
    else:
        approx_params = 135_168 * auto_rank
        ratio = approx_params / max(num_train_samples, 1)
        logger.info(
            f"Auto-selected LoRA rank: {auto_rank} "
            f"(~{approx_params / 1e6:.1f}M params for {num_train_samples} samples, "
            f"ratio {ratio:.0f}:1)"
        )
        return auto_rank


# ============================================================================
# Focal Loss for Class Imbalance
# ============================================================================


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in multi-class classification.

    From "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017).

    Focal loss down-weights well-classified (easy) examples and focuses
    training on hard, misclassified examples. This is much better than
    naive class weighting because:
    1. It handles QUALITY imbalance (BOTH class has lower-quality template data)
    2. It reduces the contribution of easy AR/DIFFUSION samples that dominate
    3. It forces the model to focus on hard boundary cases (AR vs BOTH)

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Per-class weighting tensor (like class_weights)
        gamma: Focusing parameter. gamma=0 is standard CE. gamma=2 is typical.
               Higher gamma = more focus on hard examples.
        reduction: 'mean' or 'sum'
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ============================================================================
# Model and Training
# ============================================================================


class ModalityRoutingLoRATrainer(Trainer):
    """
    Enhanced Trainer for modality routing classification with LoRA.

    Uses Focal Loss instead of naive class-weighted CrossEntropy for better
    handling of class imbalance and data quality differences.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self._focal_loss = None  # Lazily initialized on first forward pass

    def _get_loss_fn(self, device: torch.device) -> FocalLoss:
        """Get or create the Focal Loss function."""
        if self._focal_loss is None:
            alpha = (
                self.class_weights.to(device)
                if self.class_weights is not None
                else None
            )
            self._focal_loss = FocalLoss(
                alpha=alpha,
                gamma=self.focal_gamma,
                reduction="mean",
            )
            logger.info(
                f"Using Focal Loss (gamma={self.focal_gamma}, "
                f"alpha={self.class_weights.tolist() if self.class_weights is not None else 'None'})"
            )
        return self._focal_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute Focal Loss for modality routing classification."""
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if labels is not None:
            focal_loss = self._get_loss_fn(outputs.logits.device)
            loss = focal_loss(
                outputs.logits.view(-1, self.model.config.num_labels),
                labels.view(-1),
            )
        else:
            loss = None

        return (loss, outputs) if return_outputs else loss


def create_lora_modality_routing_model(
    model_name: str, num_labels: int, lora_config: dict
):
    """Create LoRA-enhanced modality routing classification model."""
    logger.info(f"Creating LoRA modality routing model with base: {model_name}")

    # Load tokenizer with model-specific configuration
    tokenizer = create_tokenizer_for_model(model_name, model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model for 3-class classification
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=torch.float32,  # Use FP32 for stable training
    )

    # Create LoRA configuration for sequence classification
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"],
        bias="none",
    )

    # Apply LoRA to the model
    lora_model = get_peft_model(base_model, peft_config)
    lora_model.print_trainable_parameters()

    return lora_model, tokenizer


def tokenize_modality_data(data, tokenizer, max_length=256):
    """Tokenize modality routing data."""
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]

    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )

    return Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }
    )


def compute_modality_metrics(eval_pred):
    """Compute modality routing metrics (3-class)."""
    predictions, labels = eval_pred
    predictions = torch.argmax(torch.tensor(predictions), dim=1)

    accuracy = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average="weighted")
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, labels=[0, 1, 2]
    )

    metrics = {
        "accuracy": accuracy,
        "f1": f1_weighted,
    }

    # Per-class metrics
    for idx, label_name in enumerate(MODALITY_LABELS):
        if idx < len(precision):
            metrics[f"precision_{label_name}"] = precision[idx]
            metrics[f"recall_{label_name}"] = recall[idx]
            metrics[f"f1_{label_name}"] = f1_per_class[idx]

    return metrics


def main(
    model_name: str = "mmbert-32k",
    lora_rank: int = None,
    lora_alpha: int = None,
    lora_dropout: float = 0.1,
    num_epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_samples: int = 6000,
    output_dir: str = None,
    gpu_id: int = None,
    vllm_endpoint: str = None,
    vllm_model: str = None,
    vllm_api_key: str = None,
    synthesize_both: int = 0,
    use_class_weights: bool = True,
):
    """
    Main training function for LoRA modality routing classification.

    Args:
        model_name: Base model name
        lora_rank: LoRA rank (higher = more capacity)
        lora_alpha: LoRA alpha scaling (typically 2x rank)
        lora_dropout: LoRA dropout rate
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        max_samples: Maximum dataset size
        output_dir: Output directory for model
        gpu_id: Specific GPU ID to use
        vllm_endpoint: vLLM endpoint for BOTH-class synthesis
        vllm_model: Model name for vLLM
        vllm_api_key: API key for vLLM
        synthesize_both: Number of BOTH-class prompts to synthesize
        use_class_weights: Whether to use class weights for imbalanced data
    """
    logger.info("=" * 70)
    logger.info("Starting Modality Routing LoRA Classification Training")
    logger.info(f"  Classes: {MODALITY_LABELS}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Max samples: {max_samples}")
    logger.info(f"  Synthesize BOTH: {synthesize_both}")
    logger.info("=" * 70)

    # Device configuration and memory management
    if gpu_id is not None:
        device, _ = set_gpu_device(gpu_id=gpu_id, auto_select=False)
    else:
        device, _ = set_gpu_device(gpu_id=None, auto_select=True)

    clear_gpu_memory()
    log_memory_usage("Pre-training")

    # Get actual model path
    model_path = resolve_model_path(model_name)
    logger.info(f"Using model: {model_name} -> {model_path}")

    # Create dataset FIRST (we need the size to recommend LoRA rank)
    sample_data, label_to_id, id_to_label = create_modality_routing_dataset(
        max_samples=max_samples,
        synthesize_both=synthesize_both,
        vllm_endpoint=vllm_endpoint,
        vllm_model=vllm_model,
        vllm_api_key=vllm_api_key,
    )

    # Split data
    train_size = int(0.8 * len(sample_data))
    train_data = sample_data[:train_size]
    val_data = sample_data[train_size:]

    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    logger.info(f"Classes: {len(label_to_id)} -> {label_to_id}")

    # ---- LoRA rank: auto-scale based on data volume ----
    effective_rank = recommend_lora_rank(
        num_train_samples=len(train_data),
        num_classes=len(label_to_id),
        user_rank=lora_rank,  # None = auto-select
    )
    # Keep alpha = 2 * rank (standard ratio) unless user explicitly set it
    if lora_alpha is not None:
        effective_alpha = lora_alpha
    else:
        effective_alpha = 2 * effective_rank
    logger.info(
        f"LoRA config: rank={effective_rank}, alpha={effective_alpha}, "
        f"dropout={lora_dropout}"
    )

    # Create LoRA configuration with dynamic target_modules
    try:
        lora_config = create_lora_config(
            model_name, effective_rank, effective_alpha, lora_dropout
        )
    except Exception as e:
        logger.error(f"Failed to create LoRA config: {e}")
        raise

    # ---- Class weights + balance analysis ----
    class_weights = None
    focal_gamma = 2.0  # Default focal loss gamma

    if use_class_weights:
        train_labels = [item["label"] for item in train_data]
        label_counts = {}
        for label in train_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        total = len(train_labels)
        n_classes = len(label_to_id)
        weights = []
        for i in range(n_classes):
            count = label_counts.get(i, 1)
            # Inverse frequency weighting with sqrt dampening
            # sqrt dampening prevents extreme weights when imbalance is severe
            raw_weight = total / (n_classes * count)
            weights.append(max(0.5, min(raw_weight**0.5, 3.0)))  # Clamp [0.5, 3.0]

        class_weights = torch.tensor(weights, dtype=torch.float32)

        # Compute imbalance ratio
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        imbalance_ratio = max_count / max(min_count, 1)

        logger.info(f"Class distribution in training data:")
        for i in range(n_classes):
            label_name = (
                MODALITY_LABELS[i] if i < len(MODALITY_LABELS) else f"class_{i}"
            )
            count = label_counts.get(i, 0)
            pct = count / total * 100
            logger.info(
                f"  {label_name}: {count} ({pct:.1f}%), weight={weights[i]:.3f}"
            )
        logger.info(f"Imbalance ratio: {imbalance_ratio:.1f}:1")

        # Adapt focal gamma based on imbalance severity
        if imbalance_ratio > 3.0:
            focal_gamma = 3.0
            logger.warning(
                f"⚠ Severe class imbalance ({imbalance_ratio:.1f}:1). "
                f"Increasing focal gamma to {focal_gamma} for stronger hard-example focus."
            )
        elif imbalance_ratio > 1.5:
            focal_gamma = 2.0
            logger.info(
                f"Moderate imbalance ({imbalance_ratio:.1f}:1). Using focal gamma={focal_gamma}"
            )
        else:
            focal_gamma = 1.5
            logger.info(
                f"Balanced data ({imbalance_ratio:.1f}:1). Using focal gamma={focal_gamma}"
            )

    # ---- Oversample minority class if severely imbalanced ----
    if use_class_weights:
        train_labels_list = [item["label"] for item in train_data]
        label_counts_train = {}
        for label in train_labels_list:
            label_counts_train[label] = label_counts_train.get(label, 0) + 1

        max_class_count = max(label_counts_train.values())
        min_class_count = min(label_counts_train.values())
        imbalance = max_class_count / max(min_class_count, 1)

        if imbalance > 2.0:
            logger.info(
                f"Oversampling minority classes to reduce {imbalance:.1f}:1 imbalance..."
            )
            # Group by class
            class_buckets: Dict[int, List] = {}
            for item in train_data:
                label = item["label"]
                if label not in class_buckets:
                    class_buckets[label] = []
                class_buckets[label].append(item)

            # Oversample each class to match the majority class count
            oversampled_train = []
            for label, items in class_buckets.items():
                if len(items) < max_class_count:
                    # Repeat + random sample to fill gap
                    repeats = max_class_count // len(items)
                    remainder = max_class_count % len(items)
                    oversampled = items * repeats + random.sample(items, remainder)
                    oversampled_train.extend(oversampled)
                    logger.info(
                        f"  {MODALITY_LABELS[label]}: {len(items)} → {len(oversampled)} "
                        f"(oversampled {repeats}x + {remainder})"
                    )
                else:
                    oversampled_train.extend(items)
                    logger.info(
                        f"  {MODALITY_LABELS[label]}: {len(items)} (majority, no change)"
                    )

            random.shuffle(oversampled_train)
            train_data = oversampled_train
            logger.info(f"Oversampled training set: {len(train_data)} samples")

    # Create LoRA model
    model, tokenizer = create_lora_modality_routing_model(
        model_path, len(label_to_id), lora_config
    )

    # Prepare datasets
    train_dataset = tokenize_modality_data(train_data, tokenizer)
    val_dataset = tokenize_modality_data(val_data, tokenizer)

    # Setup output directory
    if output_dir is None:
        output_dir = f"lora_modality_router_{model_name}_r{effective_rank}_model"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Model will be saved to: {output_dir}")

    # ---- Adaptive regularization ----
    # Higher weight_decay when param-to-sample ratio is high (overfitting risk)
    approx_params = 135_168 * effective_rank
    ratio = approx_params / max(len(train_data), 1)
    if ratio > 500:
        weight_decay = 0.15  # Aggressive regularization
        logger.info(
            f"High param-to-sample ratio ({ratio:.0f}:1) → weight_decay={weight_decay}"
        )
    elif ratio > 200:
        weight_decay = 0.10
        logger.info(
            f"Moderate param-to-sample ratio ({ratio:.0f}:1) → weight_decay={weight_decay}"
        )
    else:
        weight_decay = 0.05
        logger.info(
            f"Good param-to-sample ratio ({ratio:.0f}:1) → weight_decay={weight_decay}"
        )

    # Training arguments - optimized for 3-class LoRA classification
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        max_grad_norm=1.0,  # Gradient clipping
        lr_scheduler_type="cosine",  # Cosine LR schedule for stable convergence
        warmup_ratio=0.06,  # PEFT recommended warmup
        weight_decay=weight_decay,  # Adaptive regularization
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=3,
        report_to=[],
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=False,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=2,  # Effective larger batch for stability
    )

    # Create trainer with Focal Loss
    trainer = ModalityRoutingLoRATrainer(
        class_weights=class_weights,
        focal_gamma=focal_gamma,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_modality_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save the LoRA adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label mapping
    label_mapping_data = {
        "label_to_idx": label_to_id,
        "idx_to_label": {str(v): k for k, v in label_to_id.items()},
    }
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping_data, f, indent=2)

    # Save modality_mapping.json for Go/Rust compatibility
    with open(os.path.join(output_dir, "modality_mapping.json"), "w") as f:
        json.dump(label_mapping_data, f, indent=2)
    logger.info("Created modality_mapping.json for Go/Rust compatibility")

    # Save LoRA config
    with open(os.path.join(output_dir, "lora_config.json"), "w") as f:
        json.dump(lora_config, f, indent=2)

    # Final evaluation
    logger.info("Final evaluation on validation set...")
    eval_results = trainer.evaluate()
    logger.info("=" * 50)
    logger.info("Validation Results:")
    logger.info(f"  Accuracy:  {eval_results['eval_accuracy']:.4f}")
    logger.info(f"  F1 (weighted): {eval_results['eval_f1']:.4f}")
    for label_name in MODALITY_LABELS:
        p_key = f"eval_precision_{label_name}"
        r_key = f"eval_recall_{label_name}"
        f_key = f"eval_f1_{label_name}"
        if p_key in eval_results:
            logger.info(
                f"  {label_name}: P={eval_results[p_key]:.4f} "
                f"R={eval_results[r_key]:.4f} "
                f"F1={eval_results[f_key]:.4f}"
            )
    logger.info("=" * 50)

    # Save evaluation results
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(
            {
                k: float(v)
                for k, v in eval_results.items()
                if isinstance(v, (int, float))
            },
            f,
            indent=2,
        )

    logger.info(f"LoRA modality routing model saved to: {output_dir}")
    logger.info(f"Base model: {model_path} (adapters kept separate)")


def merge_lora_adapter_to_full_model(
    lora_adapter_path: str, output_path: str, base_model_path: str
):
    """
    Merge LoRA adapter with base model to create a complete model for Rust inference.
    """
    logger.info(f"Loading base model: {base_model_path}")

    # Load label mapping to get correct number of labels
    with open(os.path.join(lora_adapter_path, "label_mapping.json"), "r") as f:
        mapping_data = json.load(f)

    if "idx_to_label" in mapping_data:
        num_labels = len(mapping_data["idx_to_label"])
    elif "label_to_idx" in mapping_data:
        num_labels = len(mapping_data["label_to_idx"])
    else:
        num_labels = len(MODALITY_LABELS)

    # Load base model with correct number of labels
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path, num_labels=num_labels, torch_dtype=torch.float32
    )

    # Load tokenizer
    tokenizer = create_tokenizer_for_model(base_model_path, base_model_path)

    logger.info(f"Loading LoRA adapter from: {lora_adapter_path}")

    # Load LoRA model
    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    logger.info("Merging LoRA adapter with base model...")

    # Merge and unload LoRA
    merged_model = lora_model.merge_and_unload()

    logger.info(f"Saving merged model to: {output_path}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Save merged model
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Fix config.json to include correct id2label mapping
    config_path = os.path.join(output_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        if "idx_to_label" in mapping_data:
            config["id2label"] = mapping_data["idx_to_label"]
        if "label_to_idx" in mapping_data:
            config["label2id"] = mapping_data["label_to_idx"]

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Updated config.json with correct label mappings")

    # Copy important files from LoRA adapter
    for file_name in [
        "label_mapping.json",
        "lora_config.json",
        "modality_mapping.json",
        "eval_results.json",
    ]:
        src_file = Path(lora_adapter_path) / file_name
        if src_file.exists():
            shutil.copy(src_file, Path(output_path) / file_name)

    logger.info("LoRA adapter merged successfully!")


def demo_inference(
    model_path: str = "lora_modality_router_mmbert-32k_r32_model",
    model_name: str = "mmbert-32k",
):
    """Demonstrate inference with trained LoRA modality routing model."""
    logger.info(f"Loading LoRA modality routing model from: {model_path}")

    try:
        # Load label mapping
        with open(os.path.join(model_path, "label_mapping.json"), "r") as f:
            mapping_data = json.load(f)
        id_to_label = mapping_data.get("idx_to_label", {})
        num_labels = len(id_to_label)

        # Check if this is a LoRA adapter or a merged model
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            logger.info("Detected LoRA adapter model, loading with PEFT...")
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                peft_config.base_model_name_or_path,
                num_labels=num_labels,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            tokenizer = create_tokenizer_for_model(
                model_path, peft_config.base_model_name_or_path
            )
        else:
            logger.info("Detected merged model, loading directly...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=num_labels
            )
            tokenizer = create_tokenizer_for_model(model_path)

        model.eval()

        # Test examples covering all 3 classes
        test_examples = [
            # Should be AR (text-only)
            ("What is the capital of France?", "AR"),
            ("Write a Python function to sort a list", "AR"),
            ("Explain the theory of relativity", "AR"),
            ("Summarize the key points of machine learning", "AR"),
            ("What are the pros and cons of microservices?", "AR"),
            ("Translate 'hello world' to Japanese", "AR"),
            ("Write a haiku about autumn", "AR"),
            # Should be DIFFUSION (image generation)
            ("A photorealistic sunset over mountains, golden hour", "DIFFUSION"),
            ("Generate an anime-style warrior princess", "DIFFUSION"),
            ("A cyberpunk city at night, neon lights, rain", "DIFFUSION"),
            ("Professional headshot portrait, studio lighting", "DIFFUSION"),
            ("A cute cat sitting on a windowsill, oil painting style", "DIFFUSION"),
            ("Logo design for a coffee shop called Bean There", "DIFFUSION"),
            ("Fantasy landscape with floating islands and waterfalls", "DIFFUSION"),
            # Should be BOTH (text + image)
            ("Explain how photosynthesis works and show me a diagram", "BOTH"),
            ("How do I julienne vegetables? Show me the technique", "BOTH"),
            ("What is the golden ratio? Illustrate it in architecture", "BOTH"),
            ("Explain OAuth 2.0 flow with a sequence diagram", "BOTH"),
            ("Describe the anatomy of the human heart with labeled diagrams", "BOTH"),
            ("How does a CPU pipeline work? Visualize the stages", "BOTH"),
        ]

        logger.info("Running modality routing inference...")
        print("\n" + "=" * 80)
        print("MODALITY ROUTING CLASSIFICATION RESULTS")
        print("=" * 80)

        correct = 0
        total = len(test_examples)

        for example_text, expected_label in test_examples:
            inputs = tokenizer(
                example_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )

            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_id = predictions.argmax().item()
                confidence = predictions[0][predicted_class_id].item()

            predicted_label = id_to_label.get(str(predicted_class_id), "UNKNOWN")
            is_correct = predicted_label == expected_label
            if is_correct:
                correct += 1

            status = "CORRECT" if is_correct else "WRONG"

            print(f"\n  Input: {example_text}")
            print(f"  Expected: {expected_label}")
            print(
                f"  Predicted: {predicted_label} (confidence: {confidence:.4f}) [{status}]"
            )

            # Show all class probabilities
            probs = predictions[0].tolist()
            prob_str = ", ".join(
                f"{id_to_label.get(str(i), '?')}={p:.3f}" for i, p in enumerate(probs)
            )
            print(f"  Probabilities: [{prob_str}]")
            print("-" * 70)

        print(f"\nAccuracy: {correct}/{total} ({correct / total * 100:.1f}%)")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Modality Routing LoRA Classification (AR / DIFFUSION / BOTH)"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="Training or inference mode",
    )
    parser.add_argument(
        "--model",
        choices=[
            "mmbert-32k",  # mmBERT-32K YaRN - 32K context, multilingual (RECOMMENDED)
            "mmbert-base",  # mmBERT - Multilingual ModernBERT (1800+ languages, 8K context)
            "modernbert-base",  # ModernBERT base model - latest architecture
            "bert-base-uncased",  # BERT base model - most stable
            "roberta-base",  # RoBERTa base model - good for classification
        ],
        default="mmbert-32k",
        help="Base model for fine-tuning (default: mmbert-32k)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="LoRA rank. If not set, auto-scales based on dataset size "
        "(≤6K→16, ≤20K→32, ≤50K→48, >50K→64). Set explicitly to override.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha scaling (default: 2x rank). Set explicitly to override.",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (lower = more stable)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=6000,
        help="Maximum total samples across all 3 classes (recommended: 6000+)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for saving the model",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="lora_modality_router_mmbert-32k_r32_model",
        help="Path to saved model for inference",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Specific GPU ID to use. Auto-selects if not specified",
    )

    # vLLM synthesis options
    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        default=None,
        help="vLLM OpenAI-compatible endpoint URL (e.g., http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--vllm-model",
        type=str,
        default=None,
        help="Model name served by vLLM (auto-detected if not specified)",
    )
    parser.add_argument(
        "--vllm-api-key",
        type=str,
        default=None,
        help="API key for vLLM endpoint (default: EMPTY)",
    )
    parser.add_argument(
        "--synthesize-both",
        type=int,
        default=0,
        help="Number of BOTH-class prompts to synthesize via vLLM (0 = use templates only)",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting (not recommended for imbalanced data)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        main(
            model_name=args.model,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            gpu_id=args.gpu_id,
            vllm_endpoint=args.vllm_endpoint,
            vllm_model=args.vllm_model,
            vllm_api_key=args.vllm_api_key,
            synthesize_both=args.synthesize_both,
            use_class_weights=not args.no_class_weights,
        )
    elif args.mode == "test":
        demo_inference(args.model_path, args.model)
