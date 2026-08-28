# ======== mom-eval.mk ========
# = MoM first-class evaluation gates =
# ====================================

MOM_EVAL_PYTHON ?= $(if $(wildcard $(CURDIR)/.venv-agent/bin/python),$(CURDIR)/.venv-agent/bin/python,python3)
MOM_EVAL_MANIFEST ?= config/recipes/built-in/latest/mom-v1/mom-evaluation.yaml
MOM_EVAL_ENTRYPOINT ?= vllm-sr/mom-v1-blend
MOM_EVAL_OUTPUT_DIR ?= $(CURDIR)/.agent-harness/mom-eval

##@ MoM Evaluation

mom-eval-validate: ## Validate MoM evaluation manifests and schemas
	@$(LOG_TARGET)
	@$(MOM_EVAL_PYTHON) tools/agent/scripts/mom_evaluation_validate.py --all-mom-recipes --check-model-cards

mom-eval-smoke: ## Run smoke MoM evaluation for one entrypoint (set MOM_EVAL_ENTRYPOINT)
	@$(LOG_TARGET)
	@$(MOM_EVAL_PYTHON) bench/mom_eval/run_mom_eval.py \
		--manifest "$(MOM_EVAL_MANIFEST)" \
		--entrypoint "$(MOM_EVAL_ENTRYPOINT)" \
		--run-mode smoke \
		--synthesize \
		--output-dir "$(MOM_EVAL_OUTPUT_DIR)/$(MOM_EVAL_ENTRYPOINT)"

mom-eval-rc: ## Run release-candidate evaluation with regression gate
	@$(LOG_TARGET)
	@$(MOM_EVAL_PYTHON) bench/mom_eval/run_mom_eval.py \
		--manifest "$(MOM_EVAL_MANIFEST)" \
		--entrypoint "$(MOM_EVAL_ENTRYPOINT)" \
		--run-mode release-candidate \
		--output-dir "$(MOM_EVAL_OUTPUT_DIR)/rc/$(MOM_EVAL_ENTRYPOINT)"

mom-eval-publish: ## Publish scorecard artifacts for one entrypoint
	@$(LOG_TARGET)
	@slug=$$(echo "$(MOM_EVAL_ENTRYPOINT)" | awk -F/ '{print $$NF}'); \
	$(MOM_EVAL_PYTHON) bench/mom_eval/run_mom_eval.py \
		--manifest "$(MOM_EVAL_MANIFEST)" \
		--entrypoint "$(MOM_EVAL_ENTRYPOINT)" \
		--run-mode formal \
		--synthesize \
		--output-dir "$(CURDIR)/config/evaluation/scorecards/mom-v1/$$slug/1.0.0"

mom-eval-reference: ## Generate reference scorecards for all MoM V1 entrypoints
	@$(LOG_TARGET)
	@$(MOM_EVAL_PYTHON) bench/mom_eval/generate_reference_scorecards.py

.PHONY: mom-eval-validate mom-eval-smoke mom-eval-rc mom-eval-publish mom-eval-reference
