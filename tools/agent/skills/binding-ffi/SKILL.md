---
name: binding-ffi
category: fragment
description: Native bindings and FFI layers used by router-side classifiers or signal evaluation.
---

# Binding FFI

## Trigger

- The primary skill adds or changes native model/runtime behavior

## Required Surfaces

- `native_binding`

## Conditional Surfaces

- `signal_runtime`
- `local_e2e`

## Stop Conditions

- Native code cannot be compiled or validated in the current environment

## Must Read

- [docs/agent/playbooks/rust-bindings.md](../../../../docs/agent/playbooks/rust-bindings.md)

## Standard Commands

- `make test-binding-minimal`
- `make test-binding-lora`

## Acceptance

- Binding interface and router call sites stay aligned
