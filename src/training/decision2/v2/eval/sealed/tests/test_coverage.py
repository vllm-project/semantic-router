from __future__ import annotations

import bz2
import contextlib
import gzip
import hashlib
import io
import json
import lzma
import os
import stat
import struct
import sys
import tarfile
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from v2.eval.sealed import coverage, overlap

try:
    import pyarrow
    import pyarrow.parquet
except ImportError:
    pyarrow = None

OLD = time.time() - 2 * 86400
FAKE_ZSTD = f"""#!{sys.executable}
import sys
data = sys.stdin.buffer.read()
if data.startswith(b"BAD"):
    sys.exit(1)
sys.stdout.buffer.write(data.removeprefix(b"\\x28\\xb5\\x2f\\xfd"))
"""


def words(stem: str, count: int = 12) -> str:
    return " ".join(f"{stem}{i}" for i in range(count))


PASSAGES = {
    key: words(key)
    for key in ("inplace", "notes", "member", "gzjson", "pruned", "unrelated")
}


def lines(rows: list[Any]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def tar_bytes(
    members: dict[str, bytes], mode: str = "w:gz", links: tuple[str, ...] = ()
) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode=mode) as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        for name in links:
            link = tarfile.TarInfo(name)
            link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
            archive.addfile(link)
    return buffer.getvalue()


def noise(size: int) -> bytes:
    blocks = (hashlib.sha256(str(i).encode()).digest() for i in range(size // 32 + 1))
    return b"".join(blocks)[:size]


def zstd_compress(data: bytes) -> bytes:
    if coverage.zstandard is not None:
        return coverage.zstandard.ZstdCompressor().compress(data)
    return coverage.stdzstd.compress(data)


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "data"
        self.root.mkdir()
        self.runs = 0

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, rel: str, data: str | bytes, root: Path | None = None) -> Path:
        path = (root or self.root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode() if isinstance(data, str) else data)
        os.utime(path, (OLD, OLD))
        return path

    def build(self, *extra: str, roots: tuple[str, ...] = (), work: Path | None = None):
        self.runs += 1
        out = self.base / f"out-{self.runs}"
        work = work or self.base / f"work-{self.runs}"
        argv = ["build", "--node", "node-t", "--work", str(work), "--workers", "1"]
        for spec in roots or (f"train={self.root}",):
            argv += ["--root", spec]
        argv += ["--output", str(out / "manifest.json")]
        argv += ["--receipt", str(out / "receipt.json"), *extra]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(coverage.main(argv), 0)
        manifest = out / "manifest.json"
        return SimpleNamespace(
            manifest=json.loads(manifest.read_text()) if manifest.exists() else None,
            receipt=json.loads((out / "receipt.json").read_text()),
            summary=json.loads(stdout.getvalue()),
            out=out,
            work=work / "train",
        )

    def refused(self, *argv: str) -> None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            with self.assertRaises(SystemExit):
                coverage.main(["build", "--node", "node-t", *argv])

    def files(self, result: SimpleNamespace, label: str = "train") -> dict[str, Any]:
        return {f["path"]: f for f in result.manifest["labels"][label]["files"]}

    def make_tree(self) -> None:
        text = PASSAGES
        planted = f"before {text['inplace']} after"
        self.put("rows.jsonl", lines([{"t": "short filler row"}, {"t": planted}]))
        self.put("tables/table.csv", "id,text\n1,a csv cell with some words\n")
        if pyarrow is not None:
            table = pyarrow.table({"t": ["a parquet cell with some words"]})
            pyarrow.parquet.write_table(table, self.root / "tables/t.parquet")
            os.utime(self.root / "tables/t.parquet", (OLD, OLD))
        self.put("notes.txt", f"first line of the notes\n{text['notes']}\nlast line\n")
        self.put("docs/readme.md", "# Title\n\nSome markdown body text.\n")
        document = json.dumps([{"q": text["gzjson"]}]).encode()
        self.put("extract.json.gz", gzip.compress(document))
        self.put("sheet.csv.xz", lzma.compress(b"a,b\n1,a compressed csv cell\n"))
        members = {
            "inner/rows.jsonl": lines([{"t": text["member"]}]).encode(),
            "inner/table.csv": b"x,y\n1,2\n",
            "inner/extra.jsonl.bz2": bz2.compress(b'{"t": "a bz2 member row"}\n'),
            "inner/config.json": b"{}",
            "inner/nested.zip": zip_bytes({"a.jsonl": b"{}\n"}),
            "inner/image.png": b"\x89PNG",
            "../evil.jsonl": lines([{"t": text["pruned"]}]).encode(),
        }
        self.put("bundle.tar.gz", tar_bytes(members, links=("inner/link.jsonl",)))
        zipped = {"docs/a.txt": b"zip line one\nzip line two\n"}
        self.put("bundle.zip", zip_bytes(zipped))
        self.put("tokenizer.json", "{}")
        self.put("model.safetensors.index.json", "{}")
        self.put("weights.safetensors", b"\0" * 64)
        for rel in (
            "triton_cache/kernel.json",
            "skip/x.jsonl",
            "scratch-1/y.jsonl",
            ".locks/z.jsonl",
            "private/sealed/c1/protected.jsonl",
        ):
            self.put(rel, lines([{"t": text["pruned"]}]))
        self.put("blobs/0123abcd", lines([{"t": "blob text behind a snapshot link"}]))
        (self.root / "snapshots/rev").mkdir(parents=True)
        os.symlink("../../blobs/0123abcd", self.root / "snapshots/rev/data.jsonl")
        os.symlink("snapshots", self.root / "linked")
        os.symlink("private/sealed/c1/protected.jsonl", self.root / "leak.jsonl")
        hot = self.root / "hot.jsonl"
        hot.write_text(lines([{"t": "a file that is still being written"}]))

    def excludes(self) -> list[str]:
        return ["--exclude", str(self.root / "skip"), "--exclude", "*/scratch-*"]

    def build_tree(self, *extra: str) -> SimpleNamespace:
        self.make_tree()
        return self.build(*self.excludes(), *extra)


class ClassifyTest(unittest.TestCase):
    def test_classes_and_work_names(self):
        for name, expected in (
            ("a.jsonl", ("scan", "a.jsonl")),
            ("A.JSONL.GZ", ("scan", "A.JSONL.GZ")),
            ("t.parquet", ("scan", "t.parquet")),
            ("x.TSV", ("scan", "x.TSV")),
            ("notes.TXT", ("text", "notes.TXT.jsonl")),
            ("r.md", ("text", "r.md.jsonl")),
            ("x.json.gz", ("compressed", "x.json")),
            ("n.txt.xz", ("compressed", "n.txt.jsonl")),
            ("d.csv.bz2", ("compressed", "d.csv")),
            ("e.jsonl.zst", ("compressed", "e.jsonl")),
            ("b.tar.gz", ("archive", "b.tar.gz")),
            ("b.TGZ", ("archive", "b.TGZ")),
            ("b.tar", ("archive", "b.tar")),
            ("b.zip", ("archive", "b.zip")),
            ("tokenizer.json", ("metadata", "tokenizer.json")),
            ("merges.txt", ("metadata", "merges.txt")),
            ("w.bin.index.json", ("metadata", "w.bin.index.json")),
            ("config.json.gz", ("metadata", "config.json.gz")),
            ("x.parquet.gz", ("other", "x.parquet.gz")),
            ("b.tar.zst", ("other", "b.tar.zst")),
            ("0123abcd", ("other", "0123abcd")),
        ):
            with self.subTest(name=name):
                self.assertEqual(coverage.classify(name), expected)

    def test_member_paths(self):
        for name, expected in (
            ("./a/./b.jsonl", ("a/b.jsonl", "")),
            ("a//b.jsonl", ("a/b.jsonl", "")),
            ("/etc/x.jsonl", ("", "absolute")),
            ("../evil.jsonl", ("", "parent")),
            ("a/../../b.jsonl", ("", "parent")),
            ("a\\..\\b.jsonl", ("", "parent")),
            ("./", ("", "empty")),
        ):
            with self.subTest(name=name):
                self.assertEqual(coverage._member_path(name), expected)


class BuildTest(Case):
    def test_manifest_work_copies_and_receipt(self):
        result = self.build_tree()
        root, work = self.root, result.work
        files = self.files(result)
        expected = [
            root / "rows.jsonl",
            root / "snapshots/rev/data.jsonl",
            root / "tables/table.csv",
            work / "bundle.tar.gz.d/inner/extra.jsonl",
            work / "bundle.tar.gz.d/inner/rows.jsonl",
            work / "bundle.tar.gz.d/inner/table.csv",
            work / "bundle.zip.d/docs/a.txt.jsonl",
            work / "blobs/0123abcd.json",
            work / "docs/readme.md.jsonl",
            work / "extract.json",
            work / "hot.jsonl",
            work / "notes.txt.jsonl",
            work / "sheet.csv",
        ]
        if pyarrow is not None:
            expected.append(root / "tables/t.parquet")
        self.assertEqual(result.manifest["schema"], "c1-corpora/1")
        self.assertEqual(list(result.manifest["labels"]), ["train"])
        self.assertEqual(result.manifest["labels"]["train"]["kind"], "training")
        self.assertEqual(list(files), sorted(str(path) for path in expected))
        for path, entry in files.items():
            data = Path(path).read_bytes()
            self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(entry["bytes"], len(data))
        self.assertEqual(
            (work / "notes.txt.jsonl").read_bytes(), (root / "notes.txt").read_bytes()
        )
        self.assertEqual(
            (work / "notes.txt.jsonl").read_text().splitlines()[1], PASSAGES["notes"]
        )
        self.assertEqual(
            json.loads((work / "extract.json").read_text()),
            [{"q": PASSAGES["gzjson"]}],
        )
        self.assertEqual(
            (work / "sheet.csv").read_bytes(), b"a,b\n1,a compressed csv cell\n"
        )
        self.assertEqual(
            (work / "bundle.tar.gz.d/inner/extra.jsonl").read_text(),
            '{"t": "a bz2 member row"}\n',
        )
        self.assertEqual(
            (work / "bundle.zip.d/docs/a.txt.jsonl").read_text(),
            "zip line one\nzip line two\n",
        )
        self.assertEqual(
            (work / "hot.jsonl").read_bytes(), (root / "hot.jsonl").read_bytes()
        )
        self.assertEqual(list(self.base.rglob("evil.jsonl")), [])
        for top, dirs, names in os.walk(work.parent):
            self.assertEqual(stat.S_IMODE(os.stat(top).st_mode), 0o700, top)
            for name in names:
                mode = stat.S_IMODE(os.stat(os.path.join(top, name)).st_mode)
                self.assertEqual(mode, 0o600, name)
        for name in ("manifest.json", "receipt.json"):
            mode = stat.S_IMODE(os.stat(result.out / name).st_mode)
            self.assertEqual(mode, 0o600)

        receipt = result.receipt
        parquet = int(pyarrow is not None)
        cells = receipt["labels"]["train"]
        self.assertEqual(
            {kind: cell["files"] for kind, cell in cells.items()},
            {
                "scan": 4 + parquet,
                "hot": 1,
                "text": 2,
                "compressed": 2,
                "archive": 2,
                "sniffed": 1,
                "unsniffed": 0,
                "metadata": 2,
                "other": 1,
                "listed": 13 + parquet,
            },
        )
        self.assertEqual(cells["hot"]["bytes"], (root / "hot.jsonl").stat().st_size)
        self.assertEqual(receipt["totals"], cells)
        self.assertEqual(
            receipt["listed"],
            {
                "files": len(files),
                "bytes": sum(entry["bytes"] for entry in files.values()),
            },
        )
        self.assertEqual(
            receipt["metadata_skipped"],
            {"model.safetensors.index.json": 1, "tokenizer.json": 1},
        )
        self.assertEqual(
            receipt["pruned"],
            {"dirs": {".locks": 1, "exclude": 1, "glob": 1, "triton": 1}, "files": {}},
        )
        self.assertEqual(receipt["sealed_guard"], 2)
        self.assertEqual(
            (receipt["symlink_dirs"], receipt["broken_links"], receipt["renamed"]),
            (1, 0, 0),
        )
        blob = (root / "blobs/0123abcd").stat().st_size
        self.assertEqual(
            receipt["other_suffixes"], {".safetensors": {"files": 1, "bytes": 64}}
        )
        self.assertEqual(receipt["sniffed"], {"json": {"files": 1, "bytes": blob}})
        self.assertEqual(
            (work / "blobs/0123abcd.json").read_bytes(),
            (root / "blobs/0123abcd").read_bytes(),
        )
        members = receipt["archive_members"]
        self.assertEqual(members["extracted"]["files"], 4)
        self.assertEqual(
            {key: n for key, n in members["rejected"].items() if n},
            {"parent": 1, "link": 1},
        )
        self.assertEqual((members["nested"], members["metadata"]), (1, 1))
        self.assertEqual(members["other"], {"files": 1, "bytes": 4})
        self.assertEqual(
            (receipt["decompression_failures"], receipt["read_failures"]), ([], [])
        )
        self.assertEqual(receipt["roots"], {"train": str(root)})
        self.assertEqual(
            receipt["excludes"],
            {"paths": [str(root / "skip")], "globs": ["*/scratch-*"]},
        )
        data = (result.out / "manifest.json").read_bytes()
        self.assertEqual(receipt["manifest_sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(result.summary["manifest_sha256"], receipt["manifest_sha256"])
        self.assertEqual(result.summary["files"]["listed"], 13 + parquet)
        text = (result.out / "receipt.json").read_text()
        for needle in (*PASSAGES.values(), "rows.jsonl", "notes.txt", "a bz2 member"):
            self.assertNotIn(needle, text)

    def test_overlap_scan_finds_passages_through_the_manifest(self):
        result = self.build_tree()
        protected = self.base / "protected.jsonl"
        keys = list(PASSAGES)
        protected.write_text(
            lines(
                [
                    {
                        "source": "s",
                        "task": "s/t",
                        "source_item_id": str(number),
                        "state": {},
                        "overlap_texts": [PASSAGES[key]],
                    }
                    for number, key in enumerate(keys)
                ]
            )
        )
        corpora, checks, _ = overlap.manifest_corpora(result.out / "manifest.json")
        receipt, hits = overlap.scan(protected, corpora, checks=checks)
        by_key = {key: hits[number] for number, key in enumerate(keys)}
        work = result.work
        for key, path in (
            ("inplace", self.root / "rows.jsonl"),
            ("notes", work / "notes.txt.jsonl"),
            ("member", work / "bundle.tar.gz.d/inner/rows.jsonl"),
            ("gzjson", work / "extract.json"),
        ):
            with self.subTest(key=key):
                hit = by_key[key]
                self.assertEqual(hit["verdict"], "OVERLAP")
                self.assertEqual((hit["label"], hit["file"]), ("train", str(path)))
        self.assertEqual(by_key["notes"]["row"], 1)
        for key in ("pruned", "unrelated"):
            self.assertEqual(by_key[key]["verdict"], "CLEAN", key)
        self.assertEqual(receipt["resources"]["manifest_files_verified"], len(checks))

    def test_dry_run_counts_without_writing(self):
        failing = mock.Mock(side_effect=AssertionError("sniffed in a dry run"))
        with mock.patch.object(coverage, "sniff", failing):
            result = self.build_tree("--dry-run")
        self.assertIsNone(result.manifest)
        self.assertFalse((result.out / "manifest.json").exists())
        self.assertFalse(result.work.parent.exists())
        receipt = result.receipt
        parquet = int(pyarrow is not None)
        self.assertTrue(receipt["dry_run"])
        self.assertEqual(receipt["totals"]["scan"]["files"], 4 + parquet)
        self.assertEqual(receipt["totals"]["hot"]["files"], 1)
        self.assertEqual(receipt["totals"]["archive"]["files"], 2)
        self.assertIsNone(receipt["totals"]["listed"])
        self.assertIsNone(receipt["labels"]["train"]["listed"])
        blob = (self.root / "blobs/0123abcd").stat().st_size
        self.assertEqual(receipt["totals"]["unsniffed"], {"files": 1, "bytes": blob})
        self.assertEqual(receipt["totals"]["sniffed"], {"files": 0, "bytes": 0})
        self.assertEqual(receipt["totals"]["other"]["files"], 1)
        failing.assert_not_called()
        for key in ("listed", "archive_members", "sniffed", "manifest_sha256"):
            self.assertIsNone(receipt[key], key)
        self.assertEqual(receipt["sealed_guard"], 2)
        self.assertEqual(result.summary["files"]["text"], 2)

    def test_parallel_matches_serial(self):
        serial = self.build_tree()
        parallel = self.build(*self.excludes(), "--workers", "3")

        def relative(result: SimpleNamespace) -> list[tuple[str, str, int]]:
            return [
                (path.replace(str(result.work), "WORK"), f["sha256"], f["bytes"])
                for path, f in self.files(result).items()
            ]

        self.assertEqual(relative(serial), relative(parallel))
        self.assertEqual(
            serial.receipt["archive_members"], parallel.receipt["archive_members"]
        )

    def test_taken_work_name_moves_under_source_name(self):
        self.put("a.jsonl", lines([{"t": "plain"}]))
        os.utime(self.root / "a.jsonl")
        self.put("a.jsonl.xz", lzma.compress(lines([{"t": "packed"}]).encode()))
        result = self.build()
        work = result.work
        self.assertEqual(
            list(self.files(result)),
            [str(work / "a.jsonl"), str(work / "a.jsonl.xz.d/a.jsonl")],
        )
        self.assertIn("packed", (work / "a.jsonl.xz.d/a.jsonl").read_text())
        self.assertEqual(result.receipt["renamed"], 1)

    def test_nested_root_and_work_are_pruned(self):
        outer = self.base / "all"
        self.put("a.jsonl", lines([{"t": "outer"}]), outer)
        self.put("n.md", "outer notes\n", outer)
        self.put("inner/b.jsonl", lines([{"t": "inner"}]), outer)
        work = outer / "work"
        work.mkdir()
        result = self.build(
            roots=(f"outer={outer}", f"inner={outer / 'inner'}"), work=work
        )
        self.assertEqual(list(result.manifest["labels"]), ["inner", "outer"])
        self.assertEqual(
            list(self.files(result, "inner")), [str(outer / "inner/b.jsonl")]
        )
        self.assertEqual(
            list(self.files(result, "outer")),
            [str(outer / "a.jsonl"), str(work / "outer/n.md.jsonl")],
        )
        self.assertEqual(result.receipt["pruned"]["dirs"], {"root": 1, "work": 1})

    def test_sealed_roots_are_never_walked(self):
        self.put("private/sealed/c1/protected.jsonl", lines([{"t": "sealed"}]))
        os.symlink(self.root / "private/sealed", self.base / "link")
        result = self.build(
            roots=(f"a={self.root / 'private/sealed/c1'}", f"b={self.base / 'link'}")
        )
        labels = result.manifest["labels"]
        self.assertEqual(
            {label: entry["files"] for label, entry in labels.items()},
            {"a": [], "b": []},
        )
        self.assertEqual(result.receipt["sealed_guard"], 2)
        self.assertEqual(list(result.work.parent.rglob("*")), [])

    def test_failures_are_recorded_and_leave_nothing_behind(self):
        self.put("broken.json.gz", b"not gzip at all")
        self.put("bad.zip", b"not a zip either")
        self.put("cut.tar.gz", tar_bytes({"ok.jsonl": noise(20000)})[:2000])
        self.put("fine.jsonl", "{}\n")
        result = self.build()
        failures = result.receipt["decompression_failures"]
        errors = {Path(f["path"]).name: f["error"] for f in failures if not f["member"]}
        self.assertEqual(sorted(errors), ["bad.zip", "broken.json.gz", "cut.tar.gz"])
        self.assertEqual(errors["bad.zip"], "BadZipFile")
        self.assertEqual(errors["broken.json.gz"], "BadGzipFile")
        self.assertEqual(list(self.files(result)), [str(self.root / "fine.jsonl")])
        self.assertFalse((result.work / "broken.json").exists())
        self.assertFalse((result.work / "cut.tar.gz.d/ok.jsonl").exists())
        self.assertEqual(result.summary["failures"], len(failures))


class ZstdTest(Case):
    def test_unsupported_without_module_or_binary(self):
        self.put("z.jsonl.zst", b"whatever")
        self.put("blob", b"\x28\xb5\x2f\xfd whatever")
        empty = self.base / "empty-bin"
        empty.mkdir()
        with mock.patch.object(coverage, "zstandard", None), mock.patch.object(
            coverage, "stdzstd", None
        ), mock.patch.dict(os.environ, {"PATH": str(empty)}):
            result = self.build()
        self.assertEqual(result.receipt["zst_unsupported"], 2)
        self.assertEqual(
            result.receipt["sniffed"], {"unsupported.zst": {"files": 1, "bytes": 13}}
        )
        self.assertIsNone(result.receipt["zstd"])
        self.assertEqual(result.manifest["labels"]["train"]["files"], [])

    def test_binary_on_files_and_archive_members(self):
        tools = self.base / "bin"
        tools.mkdir()
        (tools / "zstd").write_text(FAKE_ZSTD)
        (tools / "zstd").chmod(0o755)
        self.put("good.jsonl.zst", '{"t": "via zstd"}\n')
        self.put("bad.json.zst", "BAD frame")
        self.put("pack.tar", tar_bytes({"m.jsonl.zst": b'{"t": "member"}\n'}, "w:"))
        big = lines(
            [{"t": f"row {i} of a blob larger than a pipe"} for i in range(9000)]
        )
        self.put("zblob", b"\x28\xb5\x2f\xfd" + big.encode())
        self.put("zbinary", b"\x28\xb5\x2f\xfd" + noise(300000))
        with mock.patch.object(coverage, "zstandard", None), mock.patch.object(
            coverage, "stdzstd", None
        ), mock.patch.dict(os.environ, {"PATH": str(tools)}):
            result = self.build()
        work = result.work
        self.assertEqual(result.receipt["zstd"], "zstd")
        self.assertEqual(
            list(self.files(result)),
            [
                str(work / "good.jsonl"),
                str(work / "pack.tar.d/m.jsonl"),
                str(work / "zblob.json"),
            ],
        )
        self.assertEqual((work / "zblob.json").read_text(), big)
        self.assertEqual(
            {kind: cell["files"] for kind, cell in result.receipt["sniffed"].items()},
            {"binary.zst": 1, "json.zst": 1},
        )
        self.assertEqual((work / "good.jsonl").read_text(), '{"t": "via zstd"}\n')
        self.assertEqual((work / "pack.tar.d/m.jsonl").read_text(), '{"t": "member"}\n')
        self.assertEqual(
            result.receipt["decompression_failures"],
            [
                {
                    "path": str(self.root / "bad.json.zst"),
                    "member": None,
                    "error": "CalledProcessError",
                }
            ],
        )
        self.assertFalse((work / "bad.json").exists())

    @unittest.skipIf(
        coverage.zstandard is None and coverage.stdzstd is None,
        "no zstandard module and no compression.zstd",
    )
    def test_module_reads_every_frame(self):
        frames = zstd_compress(b'{"t": "one"}\n') + zstd_compress(b'{"t": "two"}\n')
        self.put("multi.jsonl.zst", frames)
        result = self.build()
        self.assertEqual(
            (result.work / "multi.jsonl").read_text(), '{"t": "one"}\n{"t": "two"}\n'
        )


class SniffTest(Case):
    def test_heads(self):
        weights = struct.pack("<Q", 40) + b'{"w": {"dtype": "F32"}}'
        cut = ("x" * 63 + "é").encode()[:64]
        for head, size, expected in (
            (b"PAR1\x15\x04", 100, "parquet"),
            (gzip.compress(b"{}"), 100, "gz"),
            (lzma.compress(b"{}"), 100, "xz"),
            (bz2.compress(b"{}"), 100, "bz2"),
            (b"\x28\xb5\x2f\xfd\x00", 100, "zst"),
            (zip_bytes({"a": b"b"}), 100, "zip"),
            (weights, 1000, "weights"),
            (weights, 40, "binary"),
            (b"{" + b"\0" * 7 + b"{}", 1000, "weights"),
            (b'{"abc": {"x": 1}}', 17, "json"),
            (b' \n [{"a": 1}]', 13, "json"),
            (cut, 100, "text"),
            (cut, 64, "binary"),
            ("plain words\n".encode(), 12, "text"),
            (b"", 0, "text"),
            (b"abc\0def", 7, "binary"),
            (b"\xff\xfeabc", 5, "binary"),
        ):
            with self.subTest(head=head[:12], size=size):
                self.assertEqual(coverage.sniff(head[:64], size), expected)

    def test_suffixless_files_are_sniffed_and_copied(self):
        rows = lines([{"t": "a gzip blob row"}]).encode()
        text = b"xz blob text line\n" * 10
        members = {"docs/x.txt": b"zip blob text\n", "a.jsonl": b'{"t": "m"}\n'}
        weights = struct.pack("<Q", 24) + b'{"w":{"dtype":"F32"}}   ' + noise(64)
        self.put("blobs/parq", b"PAR1" + noise(100) + b"PAR1")
        self.put("blobs/gzrows", gzip.compress(rows))
        self.put("blobs/xztext", lzma.compress(text))
        self.put("blobs/bzparq", bz2.compress(b"PAR1" + noise(100)))
        self.put("blobs/gztar", gzip.compress(tar_bytes({"a.jsonl": rows}, "w:")))
        self.put("blobs/gzzip", gzip.compress(zip_bytes(members)))
        self.put("blobs/zipped", zip_bytes(members))
        self.put("blobs/weights", weights)
        self.put("blobs/plain", "plain text blob\n")
        self.put("blobs/plain.txt", "a text file whose WORK name the blob wants\n")
        self.put("blobs/noise", noise(200))
        self.put("blobs/broken", b"\x1f\x8b" + noise(100))
        self.put("blobs/gzweights", gzip.compress(weights))
        for rel in ("blobs/.hidden", "blobs/x.lock", "blobs/y.incomplete"):
            self.put(rel, "text that is never sniffed\n")
        result = self.build()
        work = result.work
        self.assertEqual(
            list(self.files(result)),
            sorted(
                str(work / rel)
                for rel in (
                    "blobs/gzrows.json",
                    "blobs/parq.parquet",
                    "blobs/plain.d/plain.txt.jsonl",
                    "blobs/plain.txt.jsonl",
                    "blobs/xztext.txt.jsonl",
                    "blobs/zipped.d/a.jsonl",
                    "blobs/zipped.d/docs/x.txt.jsonl",
                )
            ),
        )
        self.assertEqual((work / "blobs/gzrows.json").read_bytes(), rows)
        self.assertEqual((work / "blobs/xztext.txt.jsonl").read_bytes(), text)
        self.assertEqual(
            (work / "blobs/plain.d/plain.txt.jsonl").read_text(), "plain text blob\n"
        )
        self.assertEqual(
            (work / "blobs/parq.parquet").read_bytes(),
            (self.root / "blobs/parq").read_bytes(),
        )
        self.assertEqual(list(work.rglob("*.part")), [])
        receipt = result.receipt
        size = {path.name: path.stat().st_size for path in self.root.rglob("*")}
        self.assertEqual(
            receipt["sniffed"],
            {
                kind: {"files": 1, "bytes": size[name]}
                for kind, name in (
                    ("binary", "noise"),
                    ("binary.gz", "gztar"),
                    ("broken.gz", "broken"),
                    ("json.gz", "gzrows"),
                    ("parquet", "parq"),
                    ("parquet.bz2", "bzparq"),
                    ("text", "plain"),
                    ("text.xz", "xztext"),
                    ("weights", "weights"),
                    ("weights.gz", "gzweights"),
                    ("zip", "zipped"),
                    ("zip.gz", "gzzip"),
                )
            },
        )
        cells = receipt["totals"]
        self.assertEqual(cells["sniffed"]["files"], 12)
        self.assertEqual((cells["other"]["files"], cells["text"]["files"]), (3, 1))
        self.assertEqual(receipt["renamed"], 1)
        self.assertEqual(receipt["archive_members"]["extracted"]["files"], 2)
        self.assertEqual(
            receipt["decompression_failures"],
            [
                {
                    "path": str(self.root / "blobs/broken"),
                    "member": None,
                    "error": "BadGzipFile",
                }
            ],
        )
        for path in work.rglob("*"):
            mode = 0o700 if path.is_dir() else 0o600
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), mode, path)

    def test_unreadable_blob_is_a_read_failure(self):
        self.put("blobs/locked", "text\n").chmod(0)
        if os.access(self.root / "blobs/locked", os.R_OK):
            self.skipTest("running as a user that reads mode-000 files")
        result = self.build()
        self.assertEqual(
            result.receipt["read_failures"],
            [{"path": str(self.root / "blobs/locked"), "error": "PermissionError"}],
        )
        self.assertEqual(
            result.receipt["sniffed"], {"unreadable": {"files": 1, "bytes": 5}}
        )


ENDPOINT = "https://hf.test"
API = f"{ENDPOINT}/api/datasets/org/data"
TOKEN = "hf_SECRETtoken123"
A0, BASE, C1, C2 = (
    hashlib.sha1(name.encode()).hexdigest() for name in "a0 b c1 c2".split()
)
CONTENT = {
    "a1": b'{"t": "a v1"}\n',
    "a2": b'{"t": "a v2"}\n',
    "a3": b'{"t": "a v3"}\n',
    "big1": b"PAR1 big v1 PAR1",
    "big3": b"PAR1 big v3 PAR1",
    "keep": b"unchanged\n",
    "new": b'{"t": "new file"}\n',
}


def blob(path: str, key: str) -> dict[str, Any]:
    data = CONTENT[key]
    oid = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    return {"type": "file", "path": path, "oid": oid, "size": len(data)}


def lfs(path: str, key: str) -> dict[str, Any]:
    data = CONTENT[key]
    pointer = hashlib.sha1(key.encode()).hexdigest()
    digest = hashlib.sha256(data).hexdigest()
    return {
        "type": "file",
        "path": path,
        "oid": pointer,
        "size": 134,
        "lfs": {"oid": digest, "size": len(data), "pointerSize": 134},
    }


VERSIONS = {
    BASE: {"data/a.jsonl": "a1", "keep.txt": "keep", "data/big.parquet": "big1"},
    C1: {
        "data/a.jsonl": "a2",
        "keep.txt": "keep",
        "data/big.parquet": "big1",
        "new/n.jsonl": "new",
    },
    C2: {
        "data/a.jsonl": "a3",
        "keep.txt": "keep",
        "new/n.jsonl": "new",
        "data/big.parquet": "big3",
    },
}
TREES = {
    commit: [
        {"type": "directory", "path": "data", "oid": commit, "size": 0},
        *(
            (lfs if path.endswith(("parquet", "n.jsonl")) else blob)(path, key)
            for path, key in files.items()
        ),
    ]
    for commit, files in VERSIONS.items()
}


class Response(io.BytesIO):
    headers: dict[str, str] = {}


class FakeHub:
    """Serves the fake repo's API and files; records every request."""

    def __init__(self) -> None:
        self.requests: list[urllib.request.Request] = []
        self.routes: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.json(f"{API}/revision/main", {"sha": C2})
        history = f"{API}/commits/{C2}"
        self.json(history, [{"id": C2}, {"id": C1}], f"{history}?p=1")
        self.json(f"{history}?p=1", [{"id": BASE}, {"id": A0}])
        for commit, tree in TREES.items():
            url = f"{API}/tree/{commit}?recursive=true&expand=false"
            relative = url.removeprefix(ENDPOINT) + "&cursor=2"
            self.json(url, tree[:3], relative)
            self.json(f"{url}&cursor=2", tree[3:])
        for commit, files in VERSIONS.items():
            for path, key in files.items():
                self.routes[self.file(commit, path)] = (CONTENT[key], {})

    def json(self, url: str, value: Any, following: str = "") -> None:
        headers = {"Link": f'<{following}>; rel="next"'} if following else {}
        self.routes[url] = (json.dumps(value).encode(), headers)

    @staticmethod
    def file(commit: str, path: str) -> str:
        return f"{ENDPOINT}/datasets/org/data/resolve/{commit}/{path}"

    def urlopen(self, request: urllib.request.Request, timeout: float = 0) -> Response:
        self.requests.append(request)
        if request.full_url not in self.routes:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
        data, headers = self.routes[request.full_url]
        response = Response(data)
        response.headers = headers
        return response


class HfDeltaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.token = self.base / "token"
        self.token.write_text(f"  {TOKEN}\n")
        self.hub = FakeHub()
        self.runs = 0

    def tearDown(self):
        self.tmp.cleanup()

    def run_delta(self, *extra: str, base: str = BASE[:10]) -> SimpleNamespace:
        self.runs += 1
        out = self.base / f"delta-{self.runs}"
        argv = ["hf-delta", "--repo", "org/data", "--repo-type", "dataset"]
        argv += ["--base", base, "--token-file", str(self.token)]
        argv += ["--output", str(out / "files"), "--receipt", str(out / "receipt.json")]
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch("urllib.request.urlopen", self.hub.urlopen):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = coverage.main([*argv, *extra])
        return SimpleNamespace(
            code=code,
            out=out,
            files=out / "files",
            receipt=out / "receipt.json",
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
        )

    def downloaded(self, result: SimpleNamespace) -> dict[str, bytes]:
        return {
            str(path.relative_to(result.files)): path.read_bytes()
            for path in sorted(result.files.rglob("*"))
            if path.is_file()
        }

    def test_downloads_every_version_new_since_base(self):
        result = self.run_delta("--endpoint", ENDPOINT)
        self.assertEqual(result.code, 0, result.stderr)
        c1, c2 = C1[:12], C2[:12]
        self.assertEqual(
            self.downloaded(result),
            {
                f"{c1}/data/a.jsonl": CONTENT["a2"],
                f"{c1}/new/n.jsonl": CONTENT["new"],
                f"{c2}/data/a.jsonl": CONTENT["a3"],
                f"{c2}/data/big.parquet": CONTENT["big3"],
            },
        )
        for path in result.files.rglob("*"):
            mode = 0o700 if path.is_dir() else 0o600
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), mode, path)
        self.assertEqual(stat.S_IMODE(result.receipt.stat().st_mode), 0o600)
        receipt = json.loads(result.receipt.read_text())
        self.assertRegex(receipt.pop("utc_start"), r"^\d{4}-\d\d-\d\dT")
        self.assertRegex(receipt.pop("utc_end"), r"^\d{4}-\d\d-\d\dT")
        size = sum(len(CONTENT[key]) for key in ("a2", "new", "a3", "big3"))
        self.assertEqual(
            receipt,
            {
                "schema": "dev2-c1-hf-delta/1",
                "repo": "org/data",
                "repo_type": "dataset",
                "base": BASE,
                "head": C2,
                "later_commits": [C1, C2],
                "downloaded": {"versions": 4, "bytes": size},
                "top_level": {"data": 3, "new": 1},
            },
        )
        self.assertEqual(
            json.loads(result.stdout),
            {
                "repo": "org/data",
                "base": BASE,
                "head": C2,
                "later_commits": 2,
                "versions": 4,
                "bytes": size,
            },
        )
        urls = [request.full_url for request in self.hub.requests]
        self.assertIn(f"{API}/commits/{C2}?p=1", urls)
        self.assertIn(f"{API}/tree/{C2}?recursive=true&expand=false&cursor=2", urls)
        resolved = sorted(url for url in urls if "/resolve/" in url)
        self.assertEqual(
            resolved,
            sorted(
                FakeHub.file(commit, path)
                for commit, path in (
                    (C1, "data/a.jsonl"),
                    (C1, "new/n.jsonl"),
                    (C2, "data/a.jsonl"),
                    (C2, "data/big.parquet"),
                )
            ),
        )
        for request in self.hub.requests:
            self.assertEqual(
                request.unredirected_hdrs.get("Authorization"), f"Bearer {TOKEN}"
            )
            self.assertNotIn("Authorization", request.headers)
            self.assertNotIn(TOKEN, request.full_url)
        redirect = urllib.request.HTTPRedirectHandler().redirect_request(
            self.hub.requests[-1], None, 302, "Found", {}, "https://cdn.test/blob"
        )
        self.assertFalse(redirect.has_header("Authorization"))
        self.assertNotIn(TOKEN, result.stdout + result.stderr)
        for path in self.base.rglob("*"):
            if path.is_file() and path != self.token:
                self.assertNotIn(TOKEN.encode(), path.read_bytes(), path)

    def test_mismatch_or_http_error_refuses_and_removes_the_file(self):
        for name, commit, path in (
            ("lfs sha256", C2, "data/big.parquet"),
            ("git blob sha1", C1, "data/a.jsonl"),
            ("http error", C2, "data/a.jsonl"),
        ):
            with self.subTest(name=name):
                self.hub = FakeHub()
                url = FakeHub.file(commit, path)
                if name == "http error":
                    del self.hub.routes[url]
                else:
                    self.hub.routes[url] = (b"tampered content", {})
                result = self.run_delta("--endpoint", ENDPOINT)
                self.assertEqual(result.code, 1)
                self.assertIn("hf-delta refused", result.stderr)
                self.assertFalse((result.files / commit[:12] / path).exists())
                self.assertFalse(result.receipt.exists())
                self.assertNotIn(TOKEN, result.stdout + result.stderr)

    def test_base_not_in_history_refuses(self):
        with mock.patch.dict(os.environ, {"HF_ENDPOINT": ENDPOINT}):
            result = self.run_delta(base="abcdef12")
        self.assertEqual(result.code, 1)
        self.assertIn("not in the history", result.stderr)
        self.assertFalse(result.files.exists())
        self.assertFalse(result.receipt.exists())
        self.assertEqual(
            [request.full_url for request in self.hub.requests],
            [f"{API}/revision/main", f"{API}/commits/{C2}", f"{API}/commits/{C2}?p=1"],
        )

    def test_pagination_off_the_endpoint_refuses(self):
        history = f"{API}/commits/{C2}"
        self.hub.json(history, [{"id": C2}], "https://evil.test/next")
        result = self.run_delta("--endpoint", ENDPOINT)
        self.assertEqual(result.code, 1)
        self.assertIn("leaves hf.test", result.stderr)
        self.assertNotIn("evil", " ".join(r.full_url for r in self.hub.requests))

    def test_refused_arguments(self):
        (self.base / "busy").mkdir()
        (self.base / "busy/x").write_text("x")
        (self.base / "taken.json").write_text("{}")
        for name, argv in (
            ("bad repo", ["--repo", "../x"]),
            ("bad base", ["--base", "main"]),
            ("busy output", ["--output", str(self.base / "busy")]),
            ("existing receipt", ["--receipt", str(self.base / "taken.json")]),
            ("sealed output", ["--output", str(self.base / "private/sealed/x")]),
        ):
            with self.subTest(name=name), self.assertRaises(SystemExit):
                self.run_delta("--endpoint", ENDPOINT, *argv)
        self.assertEqual(self.hub.requests, [])
        self.assertEqual((self.base / "taken.json").read_text(), "{}")


class RefusalTest(Case):
    def args(self, name: str = "x") -> list[str]:
        out = self.base / name
        return [
            "--work",
            str(self.base / f"{name}-work"),
            "--output",
            str(out / "manifest.json"),
            "--receipt",
            str(out / "receipt.json"),
        ]

    def test_refused_arguments(self):
        (self.base / "other").mkdir()
        (self.base / "busy").mkdir()
        (self.base / "busy/file").write_text("x")
        root = f"train={self.root}"
        for name, argv in (
            ("missing root", ["--root", f"train={self.base / 'missing'}"]),
            ("duplicate label", ["--root", root, "--root", f"train={self.base}/other"]),
            ("same directory", ["--root", root, "--root", f"again={self.root}/."]),
            ("root under exclude", ["--root", root, "--exclude", str(self.base)]),
            ("root matches glob", ["--root", root, "--exclude", "*/data"]),
            ("bad label", ["--root", f"a/b={self.root}"]),
            ("busy work", ["--root", root, "--work", str(self.base / "busy")]),
            (
                "sealed work",
                ["--root", root, "--work", str(self.base / "private/sealed/w")],
            ),
        ):
            with self.subTest(name=name):
                self.refused(*self.args(name), *argv)
                self.assertFalse((self.base / name).exists())
        for existing in ("manifest.json", "receipt.json"):
            with self.subTest(existing=existing):
                (self.base / existing).mkdir()
                (self.base / existing / existing).write_text("{}")
                self.refused(*self.args(existing), "--root", root)
                self.assertEqual(os.listdir(self.base / existing), [existing])


if __name__ == "__main__":
    unittest.main()
