# ========================== docs.mk ==========================
# = Everything For Docs,include API Docs and Docs Website     =
# ========================== docs.mk ==========================

# Documentation targets
docs-install:
	@$(LOG_TARGET)
	cd website && npm install

docs-dev: docs-install
	@$(LOG_TARGET)
	cd website && npm start

docs-build: docs-install
	@$(LOG_TARGET)
	cd website && npm run build

docs-serve: docs-build
	@$(LOG_TARGET)
	cd website && npm run serve

docs-clean:
	@$(LOG_TARGET)
	cd website && npm run clear
