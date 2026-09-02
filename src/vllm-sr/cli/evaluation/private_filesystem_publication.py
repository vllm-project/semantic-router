"""Verified cross-directory publication for private evaluation artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from cli.evaluation.artifact_store_error import StoreError
from cli.evaluation.private_filesystem_descriptor import (
    read_descriptor,
    same_inode,
)
from cli.evaluation.private_filesystem_mutation import (
    PrivateFilesystemMutationPrimitives,
)


class PrivateFilesystemPublicationPrimitives(PrivateFilesystemMutationPrimitives):
    """Publish immutable files after verifying source and destination identity."""

    @classmethod
    def _read_optional_file_at(
        cls,
        parent: int,
        name: str,
        description: str,
    ) -> tuple[bytes, os.stat_result] | None:
        try:
            descriptor = cls._open_file_at(
                parent,
                name,
                description=description,
            )
        except FileNotFoundError:
            return None
        try:
            return read_descriptor(descriptor, description)
        finally:
            os.close(descriptor)

    @staticmethod
    def _require_named_inode(
        parent: int,
        name: str,
        expected: os.stat_result,
        description: str,
    ) -> None:
        current = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if not same_inode(expected, current):
            raise StoreError(f"{description} changed")

    @classmethod
    def _remove_equal_publication_source(
        cls,
        source_parent: int,
        source_name: str,
        source_metadata: os.stat_result,
        target_parent: int,
        target_name: str,
        target_snapshot: tuple[bytes, os.stat_result],
        expected_data: bytes,
    ) -> None:
        target_data, target_metadata = target_snapshot
        if target_data != expected_data:
            raise StoreError("private publication conflicts with its destination")
        cls._require_named_inode(
            target_parent,
            target_name,
            target_metadata,
            "private publication destination",
        )
        cls._require_named_inode(
            source_parent,
            source_name,
            source_metadata,
            "private publication source",
        )
        os.fsync(target_parent)
        os.unlink(source_name, dir_fd=source_parent)

    @classmethod
    def _move_and_verify_publication(
        cls,
        source_parent: int,
        source_name: str,
        source_metadata: os.stat_result,
        target_parent: int,
        target_name: str,
        target: Path,
        expected_data: bytes,
    ) -> None:
        os.replace(
            source_name,
            target_name,
            src_dir_fd=source_parent,
            dst_dir_fd=target_parent,
        )
        published = cls._open_file_at(
            target_parent,
            target_name,
            description=f"published private file {target}",
        )
        try:
            published_data, published_metadata = read_descriptor(
                published,
                f"published private file {target}",
            )
            if (
                not same_inode(source_metadata, published_metadata)
                or published_data != expected_data
            ):
                raise StoreError("private publication changed its file identity")
            os.fsync(published)
        finally:
            os.close(published)

    def replace_private_file(
        self,
        source: Path,
        target: Path,
        *,
        expected_data: bytes,
    ) -> bool:
        source = self.within_root(source)
        target = self.within_root(target)
        source_name = self._name(source)
        target_name = self._name(target)
        with (
            self._directory_descriptor(source.parent) as source_parent,
            self._directory_descriptor(target.parent) as target_parent,
        ):
            try:
                source_descriptor = self._open_file_at(
                    source_parent,
                    source_name,
                    description=f"private source file {source}",
                )
            except FileNotFoundError as exc:
                raise StoreError("private publication source is unavailable") from exc
            try:
                source_data, source_metadata = read_descriptor(
                    source_descriptor,
                    f"private source file {source}",
                )
                if source_data != expected_data:
                    raise StoreError("private publication source changed")
                target_snapshot = self._read_optional_file_at(
                    target_parent,
                    target_name,
                    f"private destination file {target}",
                )
                if target_snapshot is not None:
                    self._remove_equal_publication_source(
                        source_parent,
                        source_name,
                        source_metadata,
                        target_parent,
                        target_name,
                        target_snapshot,
                        expected_data,
                    )
                    return False
                self._move_and_verify_publication(
                    source_parent,
                    source_name,
                    source_metadata,
                    target_parent,
                    target_name,
                    target,
                    expected_data,
                )
                return True
            finally:
                os.close(source_descriptor)
