"""Deterministic transport archives for managed Recipe directories."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

import yaml

RECIPE_FILES = (
    "metadata.yaml",
    "config.yaml",
    "probes.yaml",
    "recipe.dsl",
    "README.md",
)
ARCHIVE_FILES = tuple(sorted(RECIPE_FILES))
RECIPE_DIGEST_FILES = RECIPE_FILES
FILE_SIZE_LIMITS = {
    "metadata.yaml": 256 << 10,
    "config.yaml": 16 << 20,
    "probes.yaml": 8 << 20,
    "recipe.dsl": 8 << 20,
    "README.md": 2 << 20,
}
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
METADATA_SCHEMA_VERSION = "vllm-sr/recipe-metadata/v1"
MAX_HTTPS_URL_LENGTH = 2048
MAX_METADATA_AUTHORS = 32
MAX_AUTHOR_NAME_LENGTH = 128
MAX_METADATA_TAGS = 32
MAX_METADATA_TAG_LENGTH = 64
_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_CREDENTIAL_KEY_PATTERN = re.compile(
    r"(?:api_?key|access_?key|client_?secret|password|secret|token|private_?key)$"
)
_PURE_CREDENTIAL_ENV_REFERENCE_PATTERN = re.compile(r"\$\{[A-Z_][A-Z0-9_]*\}")
_CREDENTIAL_COLLECTION_KEYS = frozenset({"api_keys"})


class RecipePackageError(ValueError):
    """Raised when a directory cannot be represented as a Recipe package."""


class _UniqueKeyLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise RecipePackageError(
                "Managed Recipe YAML must not contain anchors, aliases, or merge keys; "
                "explicit YAML tags are also forbidden"
            )
        event = self.peek_event()
        if getattr(event, "anchor", None) is not None or getattr(event, "tag", None):
            raise RecipePackageError(
                "Managed Recipe YAML must not contain anchors, aliases, or merge keys; "
                "explicit YAML tags are also forbidden"
            )
        return super().compose_node(parent, index)


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        if key_node.value == "<<" or key_node.tag == "tag:yaml.org,2002:merge":
            raise RecipePackageError(
                "Managed Recipe YAML must not contain anchors, aliases, or merge keys; "
                "explicit YAML tags are also forbidden"
            )
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise RecipePackageError(
                "YAML mapping keys must be scalar values"
            ) from error
        if duplicate:
            raise RecipePackageError(f"Duplicate YAML mapping key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True)
class PackageResult:
    path: Path
    archive_sha256: str
    recipe_digest: str

    def as_dict(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "sha256": self.archive_sha256,
            "recipe_digest": self.recipe_digest,
        }


def _read_exact_recipe_files(recipe_dir: Path) -> dict[str, bytes]:
    if recipe_dir.is_symlink():
        raise RecipePackageError(
            f"Recipe directory must not be a symbolic link: {recipe_dir}"
        )
    if not recipe_dir.is_dir():
        raise RecipePackageError(f"Recipe directory does not exist: {recipe_dir}")

    entries = {entry.name: entry for entry in recipe_dir.iterdir()}
    runtime_state = entries.pop(".vllm-sr", None)
    if runtime_state is not None and (
        runtime_state.is_symlink() or not runtime_state.is_dir()
    ):
        raise RecipePackageError(
            "The optional .vllm-sr runtime state entry must be a real directory"
        )
    expected = set(RECIPE_FILES)
    missing = sorted(expected - set(entries))
    extra = sorted(set(entries) - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise RecipePackageError(
            "Recipe source must contain the five managed files and only optional "
            ".vllm-sr runtime state: " + "; ".join(details)
        )

    files: dict[str, bytes] = {}
    for filename in RECIPE_FILES:
        path = entries[filename]
        if path.is_symlink():
            raise RecipePackageError(
                f"Recipe package files must not be symbolic links: {filename}"
            )
        limit = FILE_SIZE_LIMITS[filename]
        data = _read_bounded_regular_file(path, filename, limit)
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RecipePackageError(
                f"Recipe package file must be valid UTF-8: {filename}"
            ) from error
        files[filename] = data
    return files


def _load_yaml_mapping(data: bytes, filename: str) -> dict[str, object]:
    try:
        document = yaml.load(data.decode("utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise RecipePackageError(f"Invalid {filename} syntax") from error
    if not isinstance(document, dict) or not all(
        isinstance(key, str) for key in document
    ):
        raise RecipePackageError(f"{filename} must contain one YAML mapping")
    return document


def _read_bounded_regular_file(path: Path, filename: str, limit: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise RecipePackageError(
            f"Recipe package entry cannot be inspected safely: {filename}"
        ) from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_size > limit
    ):
        if before.st_size > limit:
            raise RecipePackageError(
                f"Recipe package file {filename} exceeds the {limit}-byte limit"
            )
        raise RecipePackageError(
            f"Recipe package entry must be a direct regular file: {filename}"
        )

    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size > limit
        ):
            raise RecipePackageError(
                f"Recipe package entry changed while being read: {filename}"
            )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            data = handle.read(limit + 1)
    except RecipePackageError:
        raise
    except OSError as error:
        raise RecipePackageError(
            f"Recipe package entry cannot be opened safely: {filename}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(data) > limit:
        raise RecipePackageError(
            f"Recipe package file {filename} exceeds the {limit}-byte limit"
        )
    return data


def _require_bounded_string(
    document: dict[str, object], key: str, *, maximum: int
) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise RecipePackageError(
            f"metadata.yaml field {key} must be a non-empty string of at most "
            f"{maximum} characters"
        )
    return value


def _validate_https_url(value: object, field: str) -> None:
    if not isinstance(value, str) or len(value) > MAX_HTTPS_URL_LENGTH:
        raise RecipePackageError(f"metadata.yaml field {field} must be an HTTPS URL")
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname
    except ValueError as error:
        raise RecipePackageError(
            f"metadata.yaml field {field} must be a valid HTTPS URL"
        ) from error
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or "?" in value
        or "#" in value
    ):
        raise RecipePackageError(
            f"metadata.yaml field {field} must be an HTTPS URL without "
            "userinfo, query, or fragment"
        )


def _validate_metadata_field_set(metadata: dict[str, object]) -> None:
    expected_fields = {
        "schema_version",
        "id",
        "name",
        "version",
        "description",
        "authors",
        "license",
        "tags",
        "links",
    }
    if set(metadata) != expected_fields:
        missing = sorted(expected_fields - set(metadata))
        extra = sorted(set(metadata) - expected_fields)
        detail = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if extra:
            detail.append(f"unknown {', '.join(extra)}")
        raise RecipePackageError(
            "metadata.yaml field set is invalid: " + "; ".join(detail)
        )
    if metadata["schema_version"] != METADATA_SCHEMA_VERSION:
        raise RecipePackageError(
            f"metadata.yaml schema_version must be {METADATA_SCHEMA_VERSION}"
        )


def _validate_metadata_authors(authors: object) -> None:
    if not isinstance(authors, list) or not 1 <= len(authors) <= MAX_METADATA_AUTHORS:
        raise RecipePackageError("metadata.yaml authors must contain 1 to 32 entries")
    unique_authors: set[str] = set()
    for index, author in enumerate(authors):
        if (
            not isinstance(author, dict)
            or not set(author).issubset({"name", "email", "url"})
            or "name" not in author
        ):
            raise RecipePackageError(f"metadata.yaml authors[{index}] is invalid")
        name = author["name"]
        if (
            not isinstance(name, str)
            or not name.strip()
            or len(name) > MAX_AUTHOR_NAME_LENGTH
        ):
            raise RecipePackageError(f"metadata.yaml authors[{index}].name is invalid")
        email = author.get("email")
        if email is not None and (
            not isinstance(email, str) or not _EMAIL_PATTERN.fullmatch(email)
        ):
            raise RecipePackageError(f"metadata.yaml authors[{index}].email is invalid")
        if "url" in author:
            _validate_https_url(author["url"], f"authors[{index}].url")
        canonical = json.dumps(author, sort_keys=True, separators=(",", ":"))
        if canonical in unique_authors:
            raise RecipePackageError("metadata.yaml authors entries must be unique")
        unique_authors.add(canonical)


def _validate_metadata_tags(tags: object) -> None:
    if not isinstance(tags, list) or not 1 <= len(tags) <= MAX_METADATA_TAGS:
        raise RecipePackageError("metadata.yaml tags must contain 1 to 32 entries")
    if len(set(tags)) != len(tags):
        raise RecipePackageError("metadata.yaml tags must be unique")
    for tag in tags:
        if (
            not isinstance(tag, str)
            or len(tag) > MAX_METADATA_TAG_LENGTH
            or not _ID_PATTERN.fullmatch(tag)
        ):
            raise RecipePackageError("metadata.yaml tags contain an invalid value")


def _validate_metadata_links(links: object) -> None:
    if (
        not isinstance(links, dict)
        or "source" not in links
        or not set(links).issubset({"homepage", "source", "documentation"})
    ):
        raise RecipePackageError("metadata.yaml links is invalid")
    for key, value in links.items():
        _validate_https_url(value, f"links.{key}")


def _validate_metadata(data: bytes) -> tuple[str, str]:
    metadata = _load_yaml_mapping(data, "metadata.yaml")
    _validate_metadata_field_set(metadata)
    recipe_id = _require_bounded_string(metadata, "id", maximum=64)
    if not _ID_PATTERN.fullmatch(recipe_id):
        raise RecipePackageError("metadata.yaml id has invalid syntax")
    version = _require_bounded_string(metadata, "version", maximum=128)
    if not _SEMVER_PATTERN.fullmatch(version):
        raise RecipePackageError(
            "metadata.yaml version must be semantic version syntax"
        )
    _require_bounded_string(metadata, "name", maximum=128)
    _require_bounded_string(metadata, "description", maximum=2048)
    _require_bounded_string(metadata, "license", maximum=128)

    _validate_metadata_authors(metadata["authors"])
    _validate_metadata_tags(metadata["tags"])
    _validate_metadata_links(metadata["links"])
    return recipe_id, version


def recipe_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for filename in RECIPE_DIGEST_FILES:
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[filename])
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _credential_reference_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if _PURE_CREDENTIAL_ENV_REFERENCE_PATTERN.fullmatch(value) is None:
        return None
    return value[2:-1]


def _allowed_environment_name(
    name: str, predicate: Callable[[str], bool] | None
) -> bool:
    return predicate is None or predicate(name)


def _is_literal_credential(
    key: str,
    parent_key: str,
    value: object,
    environment_name_is_allowed: Callable[[str], bool] | None,
) -> bool:
    if value in (None, ""):
        return False
    normalized = key.strip().lower().replace("-", "_")
    if normalized.endswith("_env"):
        return not (
            isinstance(value, str)
            and _allowed_environment_name(value, environment_name_is_allowed)
        )
    if normalized in _CREDENTIAL_COLLECTION_KEYS:
        return not isinstance(value, list)
    credential_field = _CREDENTIAL_KEY_PATTERN.search(normalized) is not None
    header_credential = normalized in {
        "authorization",
        "proxy_authorization",
        "cookie",
        "set_cookie",
        "x_api_key",
        "xapikey",
    } and parent_key in {"headers", "extra_headers"}
    if not credential_field and not header_credential:
        return False
    reference_name = _credential_reference_name(value)
    return reference_name is None or not _allowed_environment_name(
        reference_name, environment_name_is_allowed
    )


def _credential_collection_item_is_allowed(
    value: object, environment_name_is_allowed: Callable[[str], bool] | None
) -> bool:
    if value in (None, ""):
        return True
    reference_name = _credential_reference_name(value)
    return reference_name is not None and _allowed_environment_name(
        reference_name, environment_name_is_allowed
    )


def _collect_literal_credential_paths(
    value: object,
    path: str,
    parent_key: str,
    found: set[str],
    environment_name_is_allowed: Callable[[str], bool] | None,
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if _is_literal_credential(
                key_text, parent_key, child, environment_name_is_allowed
            ) or _contains_embedded_url_credential(child):
                found.add(child_path)
            normalized_key = key_text.strip().lower().replace("-", "_")
            _collect_literal_credential_paths(
                child,
                child_path,
                normalized_key,
                found,
                environment_name_is_allowed,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            if (
                parent_key in _CREDENTIAL_COLLECTION_KEYS
                and not _credential_collection_item_is_allowed(
                    child, environment_name_is_allowed
                )
            ):
                found.add(child_path)
            _collect_literal_credential_paths(
                child,
                child_path,
                parent_key,
                found,
                environment_name_is_allowed,
            )


def literal_credential_paths(
    data: bytes,
    *,
    environment_name_is_allowed: Callable[[str], bool] | None = None,
) -> tuple[str, ...]:
    """Return credential-like literal paths without exposing their values."""

    try:
        document = yaml.safe_load(data.decode("utf-8"))
    except yaml.YAMLError:
        return ()
    found: set[str] = set()
    _collect_literal_credential_paths(
        document,
        "",
        "",
        found,
        environment_name_is_allowed,
    )
    return tuple(sorted(found))


def _contains_embedded_url_credential(value: object) -> bool:
    if not isinstance(value, str) or "://" not in value:
        return False
    try:
        parsed = urlparse(value)
        if not parsed.scheme or not parsed.netloc:
            return False
        if parsed.username is not None:
            return True
        return any(
            _credential_query_key(key)
            for key, _value in parse_qsl(parsed.query, keep_blank_values=True)
        )
    except ValueError:
        return False


def _credential_query_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(
        marker in normalized
        for marker in (
            "key",
            "token",
            "secret",
            "password",
            "credential",
            "auth",
            "signature",
        )
    )


def _local_process_mcp_path(data: bytes) -> str | None:
    """Return the enabled local-process MCP path without exposing its values."""

    try:
        document = yaml.safe_load(data.decode("utf-8"))
    except yaml.YAMLError:
        return None
    if not isinstance(document, dict):
        return None

    node: object = document
    path = "global.model_catalog.modules.classifier.mcp"
    for component in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(component)
    if not isinstance(node, dict) or node.get("enabled") is not True:
        return None

    raw_transport = node.get("transport_type")
    transport = raw_transport.strip().lower() if isinstance(raw_transport, str) else ""
    command = node.get("command")
    command_present = isinstance(command, str) and bool(command.strip())
    url = node.get("url")
    url_present = isinstance(url, str) and bool(url.strip())
    if not transport:
        transport = (
            "streamable-http" if url_present and not command_present else "stdio"
        )

    args = node.get("args")
    env = node.get("env")
    if (
        transport == "stdio"
        or command_present
        or (isinstance(args, (list, dict)) and len(args) != 0)
        or (isinstance(env, (list, dict)) and len(env) != 0)
    ):
        return path
    return None


def _resolve_output_path(
    recipe_dir: Path, recipe_id: str, version: str, output: Path | None
) -> Path:
    filename = f"{recipe_id}-{version}.vllm-sr-recipe.zip"
    if output is None:
        result = recipe_dir.parent / filename
    else:
        candidate = output.expanduser()
        if candidate.is_symlink():
            raise RecipePackageError(
                f"Output path must not be a symbolic link: {candidate}"
            )
        if candidate.is_dir() or candidate.suffix.lower() != ".zip":
            candidate.mkdir(parents=True, exist_ok=True)
            if candidate.is_symlink() or not candidate.is_dir():
                raise RecipePackageError(
                    f"Output directory must be a regular directory: {candidate}"
                )
            result = candidate / filename
        else:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            if candidate.parent.is_symlink() or not candidate.parent.is_dir():
                raise RecipePackageError(
                    f"Output parent must be a regular directory: {candidate.parent}"
                )
            result = candidate
    result = result.absolute()
    if result.is_symlink():
        raise RecipePackageError(
            f"Output archive must not be a symbolic link: {result}"
        )
    try:
        result.resolve().relative_to(recipe_dir.resolve(strict=True))
    except ValueError:
        pass
    else:
        raise RecipePackageError(
            "Output archive must be outside the exact five-file root"
        )
    return result


def _write_deterministic_archive(output_path: Path, files: dict[str, bytes]) -> None:
    if output_path.is_symlink():
        raise RecipePackageError(
            f"Output archive must not be a symbolic link: {output_path}"
        )
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, 0o644)
        handle = os.fdopen(fd, "w+b")
        fd = -1
        with handle:
            with zipfile.ZipFile(
                handle,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as archive:
                for filename in ARCHIVE_FILES:
                    info = zipfile.ZipInfo(filename, FIXED_ZIP_TIMESTAMP)
                    info.create_system = 3
                    info.create_version = 20
                    info.extract_version = 20
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = (stat.S_IFREG | 0o644) << 16
                    archive.writestr(info, files[filename], compresslevel=9)
            handle.flush()
            os.fsync(handle.fileno())
        if output_path.is_symlink():
            raise RecipePackageError(
                f"Output archive became a symbolic link: {output_path}"
            )
        os.replace(temp_path, output_path)
        try:
            parent_fd = os.open(
                output_path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
        except OSError:
            parent_fd = -1
        if parent_fd >= 0:
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        with suppress(FileNotFoundError):
            temp_path.unlink()


def pack_recipe(recipe_dir: Path, output: Path | None = None) -> PackageResult:
    recipe_dir = recipe_dir.expanduser().absolute()
    files = _read_exact_recipe_files(recipe_dir)
    recipe_id, version = _validate_metadata(files["metadata.yaml"])
    _load_yaml_mapping(files["config.yaml"], "config.yaml")
    _load_yaml_mapping(files["probes.yaml"], "probes.yaml")
    credential_paths = literal_credential_paths(files["config.yaml"])
    if credential_paths:
        raise RecipePackageError(
            "config.yaml contains a literal credential-like value at "
            f"{credential_paths[0]}; use api_key_env or a pure environment reference"
        )
    local_process_mcp_path = _local_process_mcp_path(files["config.yaml"])
    if local_process_mcp_path is not None:
        raise RecipePackageError(
            "config.yaml must not enable a local-process MCP transport at "
            f"{local_process_mcp_path}"
        )
    output_path = _resolve_output_path(recipe_dir, recipe_id, version, output)
    _write_deterministic_archive(output_path, files)
    archive_digest = f"sha256:{hashlib.sha256(output_path.read_bytes()).hexdigest()}"
    return PackageResult(
        path=output_path,
        archive_sha256=archive_digest,
        recipe_digest=recipe_digest(files),
    )
