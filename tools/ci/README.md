# CI tooling

## Docker image catalog

`docker-image-catalog.tsv` is the single mapping from each CI image name to its
Docker build context, Dockerfile, and target platforms. Both
`.github/workflows/docker-validate.yml` and `.github/workflows/docker-publish.yml`
resolve image definitions with `docker_image_catalog.py`.

When adding or renaming an image mapping, edit the TSV catalog rather than
copying a `case` block into either workflow. The workflow policy validator checks
that every catalog entry has a real context and Dockerfile and that the catalog
inventory remains aligned with the images selected by CI.

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
