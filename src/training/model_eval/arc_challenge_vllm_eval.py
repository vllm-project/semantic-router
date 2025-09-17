# ARC Challenge evaluation script for vLLM OpenAI API endpoint
# Based on mmlu_pro_vllm_eval.py and ARC dataset format
# Usage example:
# python arc_challenge_vllm_eval.py --endpoint http://localhost:8000/v1 --models gemma3:27b,phi4:latest

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

# Constants
ANSWER_PATTERN = re.compile(r"(?:answer(?:\\sis)?:?\\s*)(A|B|C|D)", re.IGNORECASE)
TIMEOUT_SECONDS = 120
MAX_RETRIES = 1  # No retries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ARC Challenge benchmark against a vLLM OpenAI endpoint"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM OpenAI API endpoint URL",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="List of model names to evaluate. If not provided, will be fetched from the API.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of questions to sample. If not provided, all questions will be used.",
    )
    parser.add_argument(
        "--api-key", type=str, default="", help="API key for vLLM endpoint"
    )
    parser.add_argument(
        "--use-cot", action="store_true", help="Use Chain-of-Thought prompting"
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=1,
        help="Number of concurrent requests to make",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,  # Make it sufficient for the model to answer the question
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for text generation"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def get_available_models(endpoint: str, api_key: str = "") -> List[str]:
    """Get the list of available models from the vLLM OpenAI API endpoint."""
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )
    try:
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception as e:
        print(f"Error communicating with vLLM endpoint: {e}")
        return []


def load_arc_challenge_dataset(
    samples: Optional[int] = None, seed: int = 42
) -> pd.DataFrame:
    """Load the ARC Challenge dataset"""
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    df = pd.DataFrame(dataset)

    if samples:
        random.seed(seed)
        np.random.seed(seed)
        if len(df) > samples:
            df = df.sample(samples, random_state=seed)

    return df


def format_cot_prompt_arc(
    question: str, choices: Dict[str, List[str]], use_cot: bool = False
) -> str:
    """Format the prompt for the model with or without Chain-of-Thought."""
    formatted_options = ""

    for label, text in zip(choices["label"], choices["text"]):
        formatted_options += f"{label}) {text}\n"

    if use_cot:
        prompt = f"Question: {question}\n\nOptions:\n{formatted_options}\n\nPlease solve this step-by-step, then provide your final answer in the format 'Answer: [letter]'."
    else:
        prompt = f"Question: {question}\n\nOptions:\n{formatted_options}\n\nPlease choose the correct answer from the options above. Provide your answer in the format 'Answer: [letter]'."

    return prompt


def extract_answer_arc(response: str) -> Optional[str]:
    """Extract the answer letter from the model's response."""
    # Try to find the answer using regex pattern
    match = ANSWER_PATTERN.search(response)
    if match:
        return match.group(1).upper()

    # fallback: last occurrence of A/B/C/D
    for char in reversed(response):
        if char.upper() in "ABCD":
            return char.upper()

    return None


def call_model_with_retry(
    client: OpenAI, model: str, prompt: str, max_tokens: int, temperature: float
) -> Tuple[str, bool]:
    """Call the model with retry logic for handling timeouts and errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content, True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:  # Exponential backoff
                delay = 2**attempt
                print(
                    f"Error calling model (attempt {attempt+1}/{MAX_RETRIES}), retrying in {delay}s: {e}"
                )
                time.sleep(delay)
            else:
                print(f"Failed to call model after {MAX_RETRIES} attempts: {e}")
                return "ERROR", False


def process_question_arc(
    client: OpenAI,
    model: str,
    question_data: Dict[str, Any],
    use_cot: bool,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Process a single question and return the results."""
    question = question_data["question"]
    choices = question_data["choices"]
    correct_answer = question_data["answerKey"]

    prompt = format_cot_prompt_arc(question, choices, use_cot)

    # append the prompt and correct answer to a file
    with open("arc_challenge_vllm_eval.txt", "a") as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Correct answer: {correct_answer}\n\n")

    start_time = time.time()
    response_text, success = call_model_with_retry(
        client, model, prompt, max_tokens, temperature
    )
    end_time = time.time()

    predicted_answer = extract_answer_arc(response_text) if success else None
    is_correct = (predicted_answer == correct_answer) if predicted_answer else False
    print(f"Predicted answer: {predicted_answer}, Correct answer: {correct_answer}")

    return {
        "id": question_data["id"],
        "question": question,
        "choices": choices,
        "correct_answer": correct_answer,
        "model_response": response_text,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "response_time": end_time - start_time,
        "success": success,
    }


def evaluate_model_arc(
    df: pd.DataFrame,
    model: str,
    endpoint: str,
    api_key: str,
    use_cot: bool,
    concurrent_requests: int,
    max_tokens: int,
    temperature: float,
) -> pd.DataFrame:
    """Evaluate a model on the ARC Challenge dataset."""
    client = OpenAI(base_url=endpoint, api_key=api_key if api_key else "dummy")
    print(f"Using model: {model}, endpoint: {endpoint}, api_key: {api_key}")
    results = []

    # Convert DataFrame rows to dictionaries for processing
    questions_data = df.to_dict("records")

    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = []
        for question_data in questions_data:
            future = executor.submit(
                process_question_arc,
                client,
                model,
                question_data,
                use_cot,
                max_tokens,
                temperature,
            )
            futures.append(future)

        for future in tqdm(futures, total=len(futures), desc=f"Evaluating {model}"):
            result = future.result()
            results.append(result)

    results_df = pd.DataFrame(results)
    return results_df


def analyze_results_arc(results_df: pd.DataFrame) -> Dict[str, float]:
    """Analyze the results and compute statistics."""
    # Skip failed requests in the analysis
    valid_results = results_df[results_df["success"]]

    # Overall accuracy
    overall_accuracy = (
        valid_results["is_correct"].mean() if not valid_results.empty else 0.0
    )

    # Compute average response time
    avg_response_time = (
        valid_results["response_time"].mean() if not valid_results.empty else 0.0
    )

    return {
        "overall_accuracy": overall_accuracy,
        "avg_response_time": avg_response_time,
        "total_questions": len(results_df),
        "successful_queries": len(valid_results),
        "failed_queries": len(results_df) - len(valid_results),
    }


def save_results_arc(
    results_df: pd.DataFrame,
    analysis: Dict[str, Any],
    model: str,
    output_dir: str,
    use_cot: bool,
):
    """Save the results and analysis to files."""
    model_name = model.replace("/", "_")
    cot_suffix = "cot" if use_cot else "direct"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, f"{model_name}_{cot_suffix}")
    os.makedirs(model_dir, exist_ok=True)

    # Save detailed results
    results_df.to_csv(os.path.join(model_dir, "detailed_results.csv"), index=False)

    # Save analysis
    with open(os.path.join(model_dir, "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    # Save summary
    summary = {
        "model": model,
        "approach": "Chain-of-Thought" if use_cot else "Direct",
        "overall_accuracy": analysis["overall_accuracy"],
        "total_questions": analysis["total_questions"],
        "successful_queries": analysis["successful_queries"],
        "failed_queries": analysis["failed_queries"],
        "avg_response_time": analysis["avg_response_time"],
    }

    with open(os.path.join(model_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print(f"Model: {model}")
    print(f"Approach: {'Chain-of-Thought' if use_cot else 'Direct'}")
    print(f"Overall Accuracy: {analysis['overall_accuracy']:.4f}")
    print(f"Total Questions: {analysis['total_questions']}")
    print(f"Successful Queries: {analysis['successful_queries']}")
    print(f"Failed Queries: {analysis['failed_queries']}")
    print(f"Average Response Time: {analysis['avg_response_time']:.2f}s")
    print("=" * 50 + "\n")


def main():
    args = parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Get available models if not specified
    if not args.models:
        print("Fetching available models from vLLM endpoint...")
        models = get_available_models(args.endpoint, args.api_key)
        if not models:
            print(
                "Could not retrieve models from the endpoint. Please specify models using --models."
            )
            return
        args.models = models

    if args.models and len(args.models) == 1 and "," in args.models[0]:
        args.models = args.models[0].split(",")

    print(f"Models to evaluate: {args.models}")

    # Load dataset
    print("Loading ARC Challenge dataset...")
    df = load_arc_challenge_dataset(samples=args.samples, seed=args.seed)

    print(f"Dataset loaded: {len(df)} questions")

    # Evaluate each model
    for model in args.models:
        print(f"\nEvaluating model: {model}")
        results_df = evaluate_model_arc(
            df=df,
            model=model,
            endpoint=args.endpoint,
            api_key=args.api_key,
            use_cot=args.use_cot,
            concurrent_requests=args.concurrent_requests,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        # Analyze and save results
        analysis = analyze_results_arc(results_df)
        save_results_arc(
            results_df=results_df,
            analysis=analysis,
            model=model,
            output_dir=args.output_dir,
            use_cot=args.use_cot,
        )


if __name__ == "__main__":
    main()
