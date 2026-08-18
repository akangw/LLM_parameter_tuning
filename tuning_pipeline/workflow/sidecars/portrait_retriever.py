from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
CONTINUOUS_DIR = PROJECT_ROOT / "workflow" / "continuous"
DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "tag_params" / "output" / "params"
DEFAULT_REGISTRY = PROJECT_ROOT / "workflow" / "search_space_compiler" / "registry.yaml"


def _read(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)


def _token(name: str) -> str:
    return name.removeprefix("--").replace("-", "_").lower()


def _portrait_tokens(name: str) -> list[str]:
    """Return equivalent tokens used by generated axes and ParameterYAML.

    Generated Search Limits use snake-case object names and ``__`` path
    separators.  ParameterYAML predates that convention and uses dotted class
    names such as ``SpeculativeConfig.method``.  Keep the equivalence here so
    every automatically admitted axis can resolve evidence without requiring a
    hand-maintained registry alias.
    """

    flattened = _token(name).replace("__", ".")
    return _stable_unique(
        (
            _token(name),
            flattened,
            flattened.replace("speculative_config.", "speculativeconfig."),
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class PortraitRetriever:
    """Retrieve full portraits for changed parameters and one-hop relations.

    This class is deliberately read-only. Duplicate or aliased portraits are
    retained as variants instead of being silently collapsed.
    """

    def __init__(
        self,
        *,
        knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR,
        registry_path: Path = DEFAULT_REGISTRY,
    ) -> None:
        self.knowledge_dir = knowledge_dir.resolve()
        self.registry_path = registry_path.resolve()
        if not self.knowledge_dir.is_dir():
            raise FileNotFoundError(f"Portrait directory not found: {self.knowledge_dir}")
        registry = _read(self.registry_path)
        if not isinstance(registry, dict) or not isinstance(
            registry.get("parameters"), list
        ):
            raise ValueError("Registry must contain a parameters list")

        self.canonical_aliases: dict[str, list[str]] = {}
        self.alias_to_canonical: dict[str, str] = {}
        suffix_candidates: dict[str, set[str]] = {}
        for entry in registry["parameters"]:
            if not isinstance(entry, dict) or not entry.get("canonical_name"):
                continue
            canonical = str(entry["canonical_name"])
            aliases = [
                canonical,
                *[str(value) for value in entry.get("knowledge_names", [])],
            ]
            self.canonical_aliases[canonical] = _stable_unique(aliases)
            for alias in aliases:
                normalized = _token(alias)
                self.alias_to_canonical.setdefault(normalized, canonical)
                suffix_candidates.setdefault(
                    normalized.rsplit(".", 1)[-1], set()
                ).add(canonical)
        self.unique_suffix_to_canonical = {
            suffix: next(iter(canonicals))
            for suffix, canonicals in suffix_candidates.items()
            if len(canonicals) == 1
        }

        allowed_names: set[str] | None = None
        progress_path = self.knowledge_dir.parent / "progress.json"
        if progress_path.is_file():
            progress = _read(progress_path)
            if isinstance(progress, dict) and isinstance(
                progress.get("tagged_params"), list
            ):
                allowed_names = {str(value) for value in progress["tagged_params"]}

        self.profiles: list[dict[str, Any]] = []
        self.by_token: dict[str, list[dict[str, Any]]] = {}
        for path in sorted(self.knowledge_dir.glob("*.yaml")):
            try:
                profile = _read(path)
            except (OSError, yaml.YAMLError):
                continue
            if not isinstance(profile, dict) or not profile.get("name"):
                continue
            logical_name = str(profile["name"])
            if allowed_names is not None and logical_name not in allowed_names:
                continue
            record = {
                "source_file": str(path),
                "source_sha256": _sha256(path),
                "portrait": profile,
            }
            self.profiles.append(record)
            self.by_token.setdefault(_token(logical_name), []).append(record)

    def canonical_name(self, name: str) -> str:
        normalized = _token(name)
        exact = self.alias_to_canonical.get(normalized)
        if exact:
            return exact
        # Portrait prose sometimes spells a registry object as
        # ``compilation_config.field`` while the canonical knowledge alias is
        # ``compilation.field``. Use the final field only when it maps to one
        # and only one registry parameter.
        return self.unique_suffix_to_canonical.get(
            normalized.rsplit(".", 1)[-1],
            normalized,
        )

    def _variants(self, name: str) -> list[dict[str, Any]]:
        canonical = self.canonical_name(name)
        aliases = self.canonical_aliases.get(canonical, [name, canonical])
        tokens = _stable_unique(
            token
            for alias in aliases
            # Automatically compiled Search Limits flatten nested JSON fields
            # while ParameterYAML preserves dotted object/class paths.  These
            # axes remain valid even when the legacy registry lacks an alias.
            for token in _portrait_tokens(alias)
        )
        variants: list[dict[str, Any]] = []
        seen_files: set[str] = set()
        for token in tokens:
            for record in self.by_token.get(token, []):
                if record["source_file"] not in seen_files:
                    seen_files.add(record["source_file"])
                    variants.append(record)
        return variants

    @staticmethod
    def _limit_names(value: Any) -> set[str] | None:
        if not isinstance(value, dict):
            return None
        for field in ("active_search_limits", "search_limits"):
            if isinstance(value.get(field), dict):
                return {str(name) for name in value[field]}
        if isinstance(value.get("active_parameters"), list):
            return {
                str(item.get("canonical_name", item))
                for item in value["active_parameters"]
                if item
            }
        return {str(name) for name in value} if value else None

    def retrieve(
        self,
        changed_parameters: Iterable[str],
        *,
        search_limits: dict[str, Any] | None = None,
        scenario: dict[str, Any] | None = None,
        include_one_hop: bool = True,
    ) -> dict[str, Any]:
        requested = _stable_unique(
            self.canonical_name(str(name)) for name in changed_parameters
        )
        if not requested:
            raise ValueError("At least one changed parameter is required")

        permitted = self._limit_names(search_limits)
        if permitted is not None:
            permitted_canonical = {self.canonical_name(name) for name in permitted}
            outside = sorted(set(requested) - permitted_canonical)
            if outside:
                raise ValueError(
                    f"Changed parameters are outside supplied Search Limits: {outside}"
                )

        groups: dict[str, dict[str, Any]] = {}
        unresolved: list[str] = []
        related_names: list[str] = []
        for canonical in requested:
            variants = self._variants(canonical)
            if not variants:
                unresolved.append(canonical)
            groups[canonical] = {
                "canonical_name": canonical,
                "retrieval_role": "changed",
                "variant_count": len(variants),
                "variants": variants,
            }
            for record in variants:
                relations = record["portrait"].get("related_parameters", [])
                if not isinstance(relations, list):
                    continue
                for relation in relations:
                    if isinstance(relation, dict) and relation.get("name"):
                        related_names.append(self.canonical_name(str(relation["name"])))

        if include_one_hop:
            for canonical in _stable_unique(related_names):
                if canonical in groups:
                    continue
                variants = self._variants(canonical)
                if not variants:
                    unresolved.append(canonical)
                    continue
                groups[canonical] = {
                    "canonical_name": canonical,
                    "retrieval_role": "one_hop_related",
                    "variant_count": len(variants),
                    "variants": variants,
                }

        changed_groups = [groups[name] for name in requested]
        related_groups = [
            group
            for name, group in groups.items()
            if name not in set(requested)
        ]
        return {
            "schema_version": 1,
            "generated_at": dt.datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "mode": "offline_read_only",
            "request": {
                "changed_parameters": requested,
                "include_one_hop": include_one_hop,
                "search_limits_supplied": search_limits is not None,
            },
            "scenario": scenario,
            "summary": {
                "changed_parameter_groups": len(changed_groups),
                "related_parameter_groups": len(related_groups),
                "portrait_variants": sum(
                    group["variant_count"] for group in groups.values()
                ),
                "unresolved_names": len(set(unresolved)),
            },
            "changed_parameters": changed_groups,
            "one_hop_related_parameters": related_groups,
            "unresolved_names": sorted(set(unresolved)),
            "retrieval_policy": {
                "duplicates": "retain_all_variants",
                "natural_language_constraints": "preserved_verbatim_in_portraits",
                "execution": "none",
            },
        }


def write_evidence(result: dict[str, Any], output: Path) -> Path:
    output = output.resolve()
    if _inside(output, CONTINUOUS_DIR):
        raise ValueError("Portrait sidecar refuses to write below workflow/continuous")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(result, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter", action="append", required=True)
    parser.add_argument("--knowledge-dir", type=Path, default=DEFAULT_KNOWLEDGE_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--search-limits", type=Path)
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--no-one-hop", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    retriever = PortraitRetriever(
        knowledge_dir=args.knowledge_dir,
        registry_path=args.registry,
    )
    result = retriever.retrieve(
        args.parameter,
        search_limits=_read(args.search_limits) if args.search_limits else None,
        scenario=_read(args.scenario) if args.scenario else None,
        include_one_hop=not args.no_one_hop,
    )
    if args.output:
        print(write_evidence(result, args.output))
    else:
        print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
