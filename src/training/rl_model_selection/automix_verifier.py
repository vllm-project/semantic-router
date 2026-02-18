#!/usr/bin/env python3
"""
AutoMix Self-Verification Server
Based on arXiv:2310.12963 - AutoMix: Automatically Mixing Language Models (NeurIPS 2024)

This server implements self-verification via entailment checking:
1. Given a context (question) and answer
2. Use LLM to check if the answer is entailed by/consistent with the context
3. Return confidence score

The paper uses few-shot prompting for verification:
- Sample k times with temperature
- Compute p(correct) = count(verified) / k

Usage:
    python automix_verifier.py --port 8889 --model "microsoft/phi-2" --samples 5
"""

import argparse
import json
import logging
import statistics
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Tuple

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.error("transformers not installed")


class AutoMixVerifier:
    """
    AutoMix self-verification via entailment checking.

    The paper frames verification as entailment:
    "Given the context and the generated answer, is the answer correct?"

    We sample k times and compute p(correct) from the responses.
    """

    VERIFICATION_PROMPT = """You are a verification assistant. Given a question and an answer, determine if the answer is correct and complete.

Question: {question}
Answer: {answer}

Is this answer correct? Respond with ONLY 'YES' or 'NO'.
Response:"""

    FEW_SHOT_PROMPT = """You are a verification assistant. Your task is to verify if an answer is correct.

Example 1:
Question: What is the capital of France?
Answer: Paris
Is this answer correct? YES

Example 2:
Question: What is 2 + 2?
Answer: 5
Is this answer correct? NO

Example 3:
Question: Who wrote Romeo and Juliet?
Answer: William Shakespeare
Is this answer correct? YES

Now verify the following:
Question: {question}
Answer: {answer}
Is this answer correct?"""

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        num_samples: int = 5,
        temperature: float = 0.7,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.num_samples = num_samples
        self.temperature = temperature
        self.device = device if torch.cuda.is_available() else "cpu"

        if HAS_TRANSFORMERS:
            logger.info(f"Loading verifier model: {model_name} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Verifier model loaded successfully")
        else:
            self.tokenizer = None
            self.model = None

    def verify(
        self, question: str, answer: str, context: str = ""
    ) -> Tuple[float, List[str]]:
        """
        Verify an answer using few-shot entailment checking.

        Args:
            question: The original question/query
            answer: The generated answer to verify
            context: Optional additional context

        Returns:
            (confidence, sample_responses)
            confidence: float in [0, 1] representing p(correct)
            sample_responses: list of individual sample responses
        """
        if not self.model:
            # Fallback: random confidence
            import random

            return random.uniform(0.5, 0.9), ["NO_MODEL"]

        # Build prompt
        full_question = f"{context}\n{question}" if context else question
        prompt = self.FEW_SHOT_PROMPT.format(question=full_question, answer=answer)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Sample k times
        verified_count = 0
        sample_responses = []

        for i in range(self.num_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=True if i > 0 else False,  # First sample is greedy
                    temperature=self.temperature if i > 0 else 1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt) :].strip().upper()
            sample_responses.append(response)

            # Check if verified
            if response.startswith("YES") or response.startswith("Y"):
                verified_count += 1
            elif "YES" in response and "NO" not in response:
                verified_count += 1

        # Compute confidence
        confidence = verified_count / self.num_samples

        logger.info(
            f"Verification: {verified_count}/{self.num_samples} = {confidence:.2f}"
        )
        return confidence, sample_responses

    def should_escalate(self, confidence: float, threshold: float = 0.7) -> bool:
        """
        Determine if we should escalate to a larger model.

        Per AutoMix paper: if p(correct) < threshold, escalate.
        """
        return confidence < threshold


@dataclass
class VerificationResult:
    """Result of verification."""

    confidence: float
    should_escalate: bool
    samples: List[str]
    verified_count: int
    total_samples: int


class AutoMixHandler(BaseHTTPRequestHandler):
    """HTTP handler for AutoMix verification requests."""

    verifier: AutoMixVerifier = None  # Set by main
    threshold: float = 0.7  # Default threshold

    def do_POST(self):
        if self.path == "/verify":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            try:
                data = json.loads(body)
                question = data.get("question", "")
                answer = data.get("answer", "")
                context = data.get("context", "")
                threshold = data.get("threshold", self.threshold)

                if not question or not answer:
                    self._send_error(400, "Missing 'question' or 'answer' field")
                    return

                # Verify the answer
                confidence, samples = self.verifier.verify(question, answer, context)
                should_escalate = self.verifier.should_escalate(confidence, threshold)

                verified_count = sum(
                    1
                    for s in samples
                    if "YES" in s.upper() or s.upper().startswith("Y")
                )

                response = {
                    "confidence": confidence,
                    "should_escalate": should_escalate,
                    "verified_count": verified_count,
                    "total_samples": len(samples),
                    "samples": samples,
                    "threshold": threshold,
                }

                self._send_json(200, response)

            except json.JSONDecodeError:
                self._send_error(400, "Invalid JSON")
            except Exception as e:
                logger.exception("Error processing request")
                self._send_error(500, str(e))
        else:
            self._send_error(404, "Not found")

    def do_GET(self):
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "model": self.verifier.model_name,
                    "num_samples": self.verifier.num_samples,
                    "threshold": self.threshold,
                },
            )
        else:
            self._send_error(404, "Not found")

    def _send_json(self, status: int, data: dict):
        response = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response))
        self.end_headers()
        self.wfile.write(response)

    def _send_error(self, status: int, message: str):
        self._send_json(status, {"error": message})

    def log_message(self, format, *args):
        logger.info(f"{self.client_address[0]} - {format % args}")


def main():
    parser = argparse.ArgumentParser(description="AutoMix Self-Verification Server")
    parser.add_argument("--port", type=int, default=8889, help="Server port")
    parser.add_argument(
        "--model", type=str, default="microsoft/phi-2", help="Verifier model"
    )
    parser.add_argument("--samples", type=int, default=5, help="Number of samples (k)")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="Escalation threshold"
    )
    parser.add_argument("--test", action="store_true", help="Run test verification")
    args = parser.parse_args()

    # Initialize verifier
    verifier = AutoMixVerifier(
        model_name=args.model, num_samples=args.samples, temperature=args.temperature
    )

    if args.test:
        # Test verification
        test_cases = [
            ("What is 2+2?", "4", True),
            ("What is the capital of France?", "London", False),
            ("Who wrote Hamlet?", "William Shakespeare", True),
        ]
        for question, answer, expected in test_cases:
            confidence, samples = verifier.verify(question, answer)
            escalate = verifier.should_escalate(confidence)
            print(f"\nQ: {question}")
            print(f"A: {answer}")
            print(f"Confidence: {confidence:.2f}, Escalate: {escalate}")
            print(f"Expected correct: {expected}")
        return

    # Start server
    AutoMixHandler.verifier = verifier
    AutoMixHandler.threshold = args.threshold
    server = HTTPServer(("0.0.0.0", args.port), AutoMixHandler)
    logger.info(f"AutoMix verifier listening on port {args.port}")
    logger.info(
        f"Using model: {args.model}, samples: {args.samples}, threshold: {args.threshold}"
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
