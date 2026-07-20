from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "README.md", "README.zh-CN.md", "DATA_AVAILABILITY.md", "CITATION.cff",
    "LICENSE", "LICENSE-DATA", "environment/requirements.txt",
    "reproducibility/metadata/algorithm_catalog.csv",
    "reproducibility/metadata/figure_name_mapping.csv",
    "reproducibility/metadata/release_manifest.csv",
}
FORBIDDEN_DIRS = {".venv", "venv", ".idea", "__pycache__", ".pytest_cache", ".mypy_cache"}
FORBIDDEN_NAMES = {
    "author_disambiguation_records.csv", "author_disambiguation_summary.csv",
    "author_merge_candidates.csv", "validated_author_records.csv",
    "author_discipline_count.csv", "author_discipline_entropy.csv",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".svg", ".pdf", ".tiff", ".tif", ".ai"}
ABSOLUTE_PATH = re.compile(r"(?i)(?<![A-Za-z])[A-Za-z]:[\\/]|/(?:home|Users)/")


class Validation:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        (self.passes if condition else self.failures).append(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {".git", "dist"}:
            continue
        files.append(path)
    return sorted(files)


def validate_repository(result: Validation) -> None:
    files = repository_files()
    relative_paths = [path.relative_to(ROOT).as_posix() for path in files]
    result.check(REQUIRED.issubset(set(relative_paths)), "required repository files are present")
    result.check(all(path.isascii() for path in relative_paths), "all repository paths are ASCII")
    directory_names = {
        part
        for relative in relative_paths
        for part in Path(relative).parts[:-1]
    }
    result.check(
        all(re.fullmatch(r"[a-z0-9_.]+", name) for name in directory_names),
        "all repository directory names use lowercase English snake_case",
    )
    result.check(not any(FORBIDDEN_DIRS & set(Path(path).parts) for path in relative_paths), "cache and environment directories are absent")
    result.check(not any(Path(path).name in FORBIDDEN_NAMES for path in relative_paths), "restricted filenames are absent")
    result.check(not any(Path(path).suffix.lower() in {".xls", ".xlsx"} for path in relative_paths), "raw Excel files are absent")
    result.check(not any(path.stat().st_size > 50 * 1024 * 1024 for path in files), "tracked files are at most 50 MiB")

    public_images = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    result.check(all(path.suffix.lower() == ".jpg" for path in public_images), "repository images use JPG only")
    for path in public_images:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                result.check(image.width > 0 and image.height > 0, f"valid JPG dimensions: {path.relative_to(ROOT)}")
        except Exception as exc:
            result.check(False, f"readable JPG: {path.relative_to(ROOT)} ({exc})")

    absolute_hits = []
    for path in files:
        if path.suffix.lower() not in {".py", ".r", ".md", ".csv", ".cff", ".yml", ".yaml", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if ABSOLUTE_PATH.search(text):
            absolute_hits.append(path.relative_to(ROOT).as_posix())
    result.check(not absolute_hits, f"no local absolute paths: {absolute_hits}")

    syntax_errors = []
    python_files = [path for path in files if path.suffix.lower() == ".py"]
    for path in python_files:
        try:
            compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
        except (SyntaxError, UnicodeError) as exc:
            syntax_errors.append(f"{path.relative_to(ROOT)}: {exc}")
    result.check(not syntax_errors, f"Python syntax ({len(python_files)} files): {syntax_errors}")

    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    result.check("README.zh-CN.md" in english and "README.md" in chinese, "bilingual README links are reciprocal")

    code_license = (ROOT / "LICENSE").read_text(encoding="utf-8")
    data_license = (ROOT / "LICENSE-DATA").read_text(encoding="utf-8")
    data_license_single_line = " ".join(data_license.split())
    result.check(
        "MIT License" in code_license
        and "Permission is hereby granted, free of charge" in code_license
        and 'THE SOFTWARE IS PROVIDED "AS IS"' in code_license,
        "root LICENSE contains the complete MIT license grant and disclaimer",
    )
    result.check(
        "Creative Commons Attribution 4.0 International" in data_license
        and "https://creativecommons.org/licenses/by/4.0/legalcode" in data_license,
        "LICENSE-DATA identifies CC BY 4.0 and links its legal code",
    )
    result.check(
        "does not apply to third-party material" in data_license_single_line
        and "Raw Web of Science records are included" in data_license_single_line
        and "are not relicensed by this repository" in data_license_single_line,
        "LICENSE-DATA excludes included WOS records from the project data license",
    )
    result.check(
        "MIT License" in english
        and "CC BY 4.0" in english
        and "are not relicensed by this repository" in english,
        "English README states the code, project-data, and third-party license boundaries",
    )
    result.check(
        "MIT License" in chinese
        and "CC BY 4.0" in chinese
        and "本仓库不对其重新授权" in chinese,
        "Chinese README states the code, project-data, and third-party license boundaries",
    )


def read_checksum_file(path: Path) -> dict[str, str]:
    checksums = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, name = line.split(maxsplit=1)
        checksums[name.strip()] = digest
    return checksums


def validate_zip_members(result: Validation, archive_path: Path, figures_only: bool) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        result.check(bool(members), f"archive is non-empty: {archive_path.name}")
        result.check(all(item.filename.isascii() for item in members), f"archive paths are ASCII: {archive_path.name}")
        result.check(all(".." not in Path(item.filename).parts for item in members), f"archive paths are safe: {archive_path.name}")
        member_suffixes = {Path(item.filename).suffix.lower() for item in members}
        if figures_only:
            result.check(member_suffixes == {".jpg"}, f"figure archive extension set is exactly .jpg: {member_suffixes}")
        else:
            result.check(not (member_suffixes & IMAGE_EXTENSIONS), f"data archive contains no images: {archive_path.name}")

        for item in members:
            suffix = Path(item.filename).suffix.lower()
            if suffix == ".jpg":
                try:
                    with Image.open(io.BytesIO(archive.read(item))) as image:
                        image.verify()
                    with Image.open(io.BytesIO(archive.read(item))) as image:
                        result.check(image.width > 0 and image.height > 0, f"valid Release JPG: {item.filename}")
                except Exception as exc:
                    result.check(False, f"readable Release JPG: {item.filename} ({exc})")
        if archive_path.name.startswith("complex-coauthorship-network-community-data"):
            names = {item.filename for item in members}
            required_originals = {
                "reproducibility/data/01_data_preparation/wos_1900_2025/1900_30_1000.xls",
                "reproducibility/data/01_data_preparation/wos_2000_2025/2006_1000.xls",
                "reproducibility/results/01_data_preparation/author_disambiguation/author_disambiguation_records.csv",
                "reproducibility/results/01_data_preparation/author_disambiguation/author_disambiguation_summary.csv",
                "reproducibility/results/01_data_preparation/author_disambiguation/author_merge_candidates.csv",
                "reproducibility/results/01_data_preparation/author_disambiguation/validated_author_records.csv",
            }
            result.check(
                required_originals.issubset(names),
                "network/community Release contains raw WOS and full author-disambiguation artifacts",
            )


def validate_release(result: Validation, release_dir: Path) -> None:
    expected = {
        "complex-coauthorship-network-community-data-v0.1.0.zip",
        "complex-coauthorship-analysis-results-v0.1.0.zip",
        "complex-coauthorship-figures-jpg-v0.1.0.zip",
    }
    checksum_path = release_dir / "SHA256SUMS.txt"
    result.check(checksum_path.exists(), "Release checksum file exists")
    if not checksum_path.exists():
        return
    checksums = read_checksum_file(checksum_path)
    result.check(set(checksums) == expected, "Release checksum lists the three expected archives")
    for name in sorted(expected):
        path = release_dir / name
        result.check(path.exists(), f"Release archive exists: {name}")
        if not path.exists():
            continue
        result.check(sha256(path) == checksums.get(name), f"Release SHA-256 matches: {name}")
        validate_zip_members(result, path, figures_only="figures-jpg" in name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the public reproducibility repository and Release assets.")
    parser.add_argument("--release-dir", type=Path, default=ROOT / "dist" / "v0.1.0")
    parser.add_argument("--skip-release", action="store_true")
    parser.add_argument(
        "--json-report",
        type=Path,
        default=ROOT / "reproducibility" / "metadata" / "validation_report.json",
    )
    args = parser.parse_args()
    result = Validation()
    validate_repository(result)
    if not args.skip_release:
        validate_release(result, args.release_dir.resolve())
    for message in result.passes:
        print(f"[PASS] {message}")
    for message in result.failures:
        print(f"[FAIL] {message}")
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(
        json.dumps(
            {
                "status": "passed" if not result.failures else "failed",
                "passed": len(result.passes),
                "failed": len(result.failures),
                "failures": result.failures,
                "r_syntax": "skipped: Rscript is not installed in the validation environment",
                "release_validation": not args.skip_release,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"summary: passed={len(result.passes)} failed={len(result.failures)}")
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
