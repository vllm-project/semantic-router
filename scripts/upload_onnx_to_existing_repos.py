#!/usr/bin/env python3
"""
Upload ONNX classifier models to existing HuggingFace repos under onnx/ directory.
Also cleans up any accidentally created separate ONNX repos.
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, delete_repo


def delete_onnx_repos(api: HfApi, org: str = "llm-semantic-router"):
    """Delete the accidentally created separate ONNX repos."""
    repos_to_delete = [
        "mmbert32k-intent-classifier-onnx",
        "mmbert32k-jailbreak-detector-onnx", 
        "mmbert32k-pii-detector-onnx",
    ]
    
    for repo_name in repos_to_delete:
        full_name = f"{org}/{repo_name}"
        try:
            print(f"Deleting repo: {full_name}")
            delete_repo(full_name, repo_type="model")
            print(f"  ✓ Deleted {full_name}")
        except Exception as e:
            print(f"  Could not delete {full_name}: {e}")


def upload_to_existing_repo(
    model_path: str, 
    repo_name: str, 
    org: str = "llm-semantic-router"
):
    """Upload ONNX model files to existing repo under onnx/ directory."""
    
    print(f"\n{'='*60}")
    print(f"Uploading: {model_path}")
    print(f"To: {org}/{repo_name}/onnx/")
    print(f"{'='*60}")
    
    api = HfApi()
    model_dir = Path(model_path)
    full_repo_name = f"{org}/{repo_name}"
    
    # Files to upload (exclude tokenizer files since they're already in parent repo)
    files_to_upload = [
        "model.onnx",
        "config.json",
    ]
    
    # Also upload any mapping files
    for f in model_dir.iterdir():
        if f.name.endswith("_mapping.json") or f.name == "label_mapping.json":
            files_to_upload.append(f.name)
    
    print(f"  Files to upload: {files_to_upload}")
    
    for filename in files_to_upload:
        file_path = model_dir / filename
        if not file_path.exists():
            print(f"  Skipping {filename} (not found)")
            continue
            
        print(f"  Uploading {filename}...")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=f"onnx/{filename}",
            repo_id=full_repo_name,
            repo_type="model",
            commit_message=f"Add ONNX model: {filename}",
        )
    
    print(f"  ✓ Upload complete!")
    print(f"  URL: https://huggingface.co/{full_repo_name}/tree/main/onnx")
    
    return full_repo_name


def main():
    parser = argparse.ArgumentParser(description="Upload ONNX models to existing HF repos")
    parser.add_argument("--model", choices=["intent", "jailbreak", "pii", "all"], 
                        default="all", help="Which model to upload")
    parser.add_argument("--org", default="llm-semantic-router", help="HuggingFace organization")
    parser.add_argument("--base-dir", default=".", help="Base directory containing ONNX models")
    parser.add_argument("--cleanup", action="store_true", help="Delete accidentally created separate ONNX repos")
    args = parser.parse_args()
    
    api = HfApi()
    
    # Cleanup separate ONNX repos if requested
    if args.cleanup:
        delete_onnx_repos(api, args.org)
    
    base_dir = Path(args.base_dir)
    
    # Map local ONNX dirs to existing repo names
    models = {
        "intent": {
            "local_path": "mmbert32k-intent-classifier-onnx",
            "repo_name": "mmbert32k-intent-classifier-merged",  # existing repo
        },
        "jailbreak": {
            "local_path": "mmbert32k-jailbreak-detector-onnx",
            "repo_name": "mmbert32k-jailbreak-detector-merged",  # existing repo
        },
        "pii": {
            "local_path": "mmbert32k-pii-detector-onnx",
            "repo_name": "mmbert32k-pii-detector-merged",  # existing repo
        }
    }
    
    to_upload = [args.model] if args.model != "all" else ["intent", "jailbreak", "pii"]
    
    uploaded = []
    for model_name in to_upload:
        model_info = models[model_name]
        local_path = base_dir / model_info["local_path"]
        
        if not local_path.exists():
            print(f"⚠ Skipping {model_name}: {local_path} not found")
            continue
        
        try:
            repo = upload_to_existing_repo(
                str(local_path), 
                model_info["repo_name"], 
                args.org
            )
            uploaded.append(repo)
        except Exception as e:
            print(f"⚠ Error uploading {model_name}: {e}")
    
    print("\n" + "="*60)
    print("Upload complete!")
    print("="*60)
    print("\nUploaded to:")
    for repo in uploaded:
        print(f"  - https://huggingface.co/{repo}/tree/main/onnx")


if __name__ == "__main__":
    main()
