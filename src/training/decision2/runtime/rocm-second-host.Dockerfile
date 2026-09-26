FROM decision-final-validation:35088a

# The FLA tree and causal-conv wheel are copied from the exact qualified
# Decision 2.0 runtime on the first research host. Their hashes are recorded
# with each run; this image is used only for the second research host.
COPY decision-fla /opt/decision-fla
COPY causal/causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl /tmp/
RUN python3 -m pip install --no-deps --no-cache-dir /tmp/causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl \
    && python3 -c "import causal_conv1d; print(causal_conv1d.__version__)"
ENV PYTHONPATH=/opt/decision-fla
WORKDIR /work
