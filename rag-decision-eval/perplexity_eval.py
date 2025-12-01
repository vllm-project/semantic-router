"""
For each question, query the model for the top 5 log probs of the first 100 tokens of the response.
Evaluate the avg perplexity score of the response.
If the avg perplexity score is greater than the threshold return true.
"""

import openai
import logging  # Assumed for observability, replace with your logger
import math
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score

class ChatCompletionClient:
    def __init__(self, endpoint: str):
        self.client = openai.OpenAI(base_url=endpoint)

    def query_model_for_logprobs(self, openai_request: dict, matched_model: str):
        logging.info("Querying '%s' for log probs", matched_model)
        
        try:
            response = self.client.chat.completions.create(**openai_request)
        except Exception as e:
            raise ValueError(f"Error calling chat completions: {e}")

        if not response.choices:
            raise ValueError("No choices returned")

        logprobs = response.choices[0].logprobs
        logging.debug("Full logprobs: %s", logprobs)
        return logprobs


def calculate_perplexity(logprobs):
    total_log_probs = 0
    token_length = 0
    for token_log_probs in logprobs.content:
        prob = token_log_probs.logprob
        total_log_probs += prob
        token_length += 1
    
    if token_length == 0:
        return 0
    
    perplexity = math.exp(-(total_log_probs / token_length))
    logging.info("Perplexity Score: '%s'", perplexity)
    return perplexity

def create_log_prob_request(question, model_name):
    openai_request = {}
    openai_request["messages"] = []
    new_message = {"role": "user", "content": question}
    openai_request["messages"].append(new_message)
    openai_request["model"] = model_name
    openai_request["logprobs"] = True
    openai_request["top_logprobs"] = 5
    openai_request["max_completion_tokens"] = 100
    return openai_request

def calc_eval_metrics(df):
    # Calculate metrics
    accuracy = accuracy_score(df['rag_decision'], df['perplexity_decision'])
    recall = recall_score(df['rag_decision'], df['perplexity_decision'])
    f1 = f1_score(df['rag_decision'], df['perplexity_decision'])

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


def main():
    data_path = "rag-decision-eval/rag_decision_data.csv"
    ENDPOINT = "http://" + "127.0.0.1:8000" + "/v1"
    client = ChatCompletionClient(ENDPOINT)
    df = pd.read_csv(data_path)
    model_name = "llama-3.2-3b"
    threshold = 1.3
    predicted = []

    for _,row in df.iterrows():
        question = row["question"]
        label = row["rag_decision"]
        request = create_log_prob_request(question, model_name)
        log_probs = client.query_model_for_logprobs(request, model_name)
        perplexity = calculate_perplexity(log_probs)
        if perplexity > threshold:
            inferred_decision = 1
        else:
            inferred_decision = 0
        predicted.append(inferred_decision)
    
    df["perplexity_decision"] = predicted
    calc_eval_metrics(df)

if __name__ == "__main__":
    main()







