# ======== gb10-qualification.mk ========
# = Static checks for the GB10 operator profile =
# ===============================================

GB10_QUALIFICATION_PYTHON ?= $(if $(wildcard $(CURDIR)/.venv-agent/bin/python),$(CURDIR)/.venv-agent/bin/python,python3)

##@ GB10 qualification

gb10-qualification-static: ## Fail-closed checks for the GB10 qualification profile
	@$(LOG_TARGET)
	@cd tools/qualification/gb10 && $(GB10_QUALIFICATION_PYTHON) -m unittest test_qualification.py
