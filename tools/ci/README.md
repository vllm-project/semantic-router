# CI tooling

## Docker image catalog

`docker-image-catalog.tsv` is the single mapping from each CI image name to its
Docker build context, Dockerfile, and target platforms.
`tools/ci/image_artifacts.py` loads that catalog into `DEFINITIONS`.
`.github/workflows/build-artifacts.yml` is the shared read-only producer and
resolves each image with `image_artifacts.py definition`.
`.github/workflows/docker-publish.yml` promotes the sealed artifact and does not
rebuild it.

When adding or renaming an image mapping, edit the TSV catalog rather than
copying a mapping into the artifact builder or a workflow. The workflow policy
validator checks that every catalog entry has a real context and Dockerfile,
that the catalog inventory matches the CI image inventory, and that the shared
producer consumes the catalog outputs.

To check every catalog Dockerfile without running a build, create the
`catalog-validation` Buildx builder and run one check per catalog platform:

```bash
docker buildx create --name catalog-validation --driver docker-container --use
while IFS=$'\t' read -r image context dockerfile platforms; do
  [[ -z "${image}" || "${image}" == \#* ]] && continue
  IFS=',' read -ra targets <<< "${platforms}"
  for platform in "${targets[@]}"; do
    docker buildx build --builder catalog-validation --check \
      --platform "${platform}" --file "${dockerfile}" "${context}" || exit $?
  done
done < tools/ci/docker-image-catalog.tsv
```
