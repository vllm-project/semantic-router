precommit-install:
	pip install pre-commit

precommit-check:
	@FILES=$$(find . -type f \( -name "*.go" -o -name "*.rs" -o -name "*.py" -o -name "*.js" -o -name "*.md" -o -name "*.yaml" -o -name "*.yml" \) \
		! -path "./target/*" \
		! -path "./candle-binding/target/*" \
		! -path "./.git/*" \
		! -path "./node_modules/*" \
		! -path "./vendor/*" \
		! -path "./__pycache__/*" \
		! -path "./site/*" \
		! -name "*.pb.go" \
		| tr '\n' ' '); \
	if [ -n "$$FILES" ]; then \
		echo "Running pre-commit on files: $$FILES"; \
		pre-commit run --files $$FILES; \
	else \
		echo "No Go, Rust, JavaScript, Markdown, Yaml, or Python files found to check"; \
	fi

precommit-local:
	docker pull ghcr.io/vllm/semantic-router/precommit:latest
	docker run --rm -v $$(pwd):/data ghcr.io/vllm-project/semantic-router/precommit:latest pre-commit run --all-files
