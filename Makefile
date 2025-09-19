# Semantic Router Root Makefile Define.
# It is refer tools/make/*.mk as the sub-makefiles.

_run:
	@$(MAKE) --warn-undefined-variables \
		-f tools/make/common.mk \
		-f tools/make/envs.mk \
		-f tools/make/envoy.mk \
		-f tools/make/golang.mk \
		-f tools/make/rust.mk \
		-f tools/make/build-run-test.mk \
		-f tools/make/docs.mk \
		-f tools/make/linter.mk \
		-f tools/make/milvus.mk \
		-f tools/make/models.mk \
		-f tools/make/pre-commit.mk \
		$(MAKECMDGOALS)

.PHONY: _run

$(if $(MAKECMDGOALS),$(MAKECMDGOALS): %: _run)
