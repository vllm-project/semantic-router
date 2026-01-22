from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Paths
ckpt_dir = Path("preference_model_qwen3")  # your saved --output-dir
sharegpt_like_convo = [
    {
        "role": "user",
        "content": "I need help booking a flight from SFO to JFK next week.",
    },
    {"role": "assistant", "content": "Sure, what dates and times do you prefer?"},
]

# Label space for inference (include your catch-all)
label_space = [
    "travel_booking",
    "weather",
    "code_generation",
    "general_inquiry",
]

# Load tokenizer/model
tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    ckpt_dir,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",  # or .to("cuda") if a single GPU
)
model.config.use_cache = False

# Build prompt text (mirrors training prompt style)
label_clause = ", ".join(sorted(set(label_space)))
conversation_text = "\n".join(
    [
        (
            "User: " + turn["content"]
            if turn["role"] == "user"
            else "Assistant: " + turn["content"]
        )
        for turn in sharegpt_like_convo
    ]
)
user_content = (
    f"Valid labels: {label_clause}\n\n"
    f"Conversation:\n{conversation_text}\n\n"
    "Answer with the single label that best matches the conversation."
)

messages = [
    {
        "role": "system",
        "content": "You are a routing controller that reads a conversation and outputs the best preference label for downstream model routing. If none of the labels apply, respond with 'general_inquiry'\n",
    },
    {"role": "user", "content": user_content},
]

prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

# Generate
inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=8,
        temperature=0.0,
        do_sample=False,
    )

# Decode just the newly generated tokens
gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
pred_label = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
print("Predicted label:", pred_label)
