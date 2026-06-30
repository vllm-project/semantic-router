"""MAB algorithm comparison harness.

Pure-Python synthetic benchmarking for multi-armed bandit algorithms used in
Router Learning. Complementary to bench/agentic_routing_experiment.py
(fixture-based engineering eval); this harness adds academic MAB metrics
(cumulative regret, optimal-arm rate, non-stationary recovery time) which
do not exist elsewhere in the codebase.

Standard-library only on the runner path; matplotlib is required for plotting.

See README.md for usage and the comparison protocol.
"""
