from huggingface_hub import login, upload_folder


login()
upload_folder(
    folder_path="preference_model_qwen3",
    repo_id="ppppqp/vLLM-SR-Preference-V1",
    repo_type="model",
)
