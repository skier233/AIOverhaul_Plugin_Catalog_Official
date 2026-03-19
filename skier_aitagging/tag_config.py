from __future__ import annotations

import csv
import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, Literal, Tuple

from .stash_handler import resolve_ai_tag_reference

_log = logging.getLogger(__name__)

_CONFIG_FILENAME = "tag_settings.csv"
_TEMPLATE_FILENAME = "tag_settings.template.csv"


# TODO: We shouldn't be running db queries directly here to get plugin settings
def _get_tag_suffix() -> str:
    """Load tag_suffix from plugin settings, defaulting to '_AI'."""
    try:
        from sqlalchemy import select
        from stash_ai_server.db.session import SessionLocal
        from stash_ai_server.models.plugin import PluginSetting

        with SessionLocal() as session:
            row = session.execute(
                select(PluginSetting).where(
                    PluginSetting.plugin_name == "skier_aitagging",
                    PluginSetting.key == "tag_suffix",
                )
            ).scalar_one_or_none()

            if row is not None:
                value = row.value if row.value is not None else row.default_value
                if value is not None and isinstance(value, str):
                    return value
    except Exception as exc:
        _log.warning("Failed to load tag_suffix from plugin settings: %s", exc)

    return "_AI"


@dataclass(slots=True)
class SceneTagDurationRequirement:
    unit: Literal["seconds", "percent"]
    value: float

    def as_seconds(self, scene_duration: float | None) -> float | None:
        if self.unit == "seconds":
            return self.value
        if scene_duration is None or scene_duration <= 0:
            return None
        return (self.value / 100.0) * scene_duration


@dataclass(slots=True)
class TagSettings:
    tag_name: str
    stash_name: str | None
    markers_enabled: bool
    scene_tag_enabled: bool
    image_enabled: bool
    required_scene_tag_duration: SceneTagDurationRequirement | None
    min_marker_duration: float | None
    max_gap: float | None
    merge_strategy: str
    merge_params: Tuple[
        float | None, float | None, float | None, float | None, float | None
    ]


# TODO: This whole pattern is overcomplicated; we should simplify
@dataclass(slots=True)
class TagSettingsOverride:
    stash_name: str | None = None
    markers_enabled: bool | None = None
    scene_tag_enabled: bool | None = None
    image_enabled: bool | None = None
    required_scene_tag_duration: SceneTagDurationRequirement | None = None
    min_marker_duration: float | None = None
    max_gap: float | None = None
    merge_strategy: str | None = None
    merge_params: Tuple[
        float | None, float | None, float | None, float | None, float | None
    ] = (
        None,
        None,
        None,
        None,
        None,
    )


def _base_settings() -> TagSettings:
    return TagSettings(
        tag_name="__global__",
        stash_name=None,
        markers_enabled=True,
        scene_tag_enabled=True,
        image_enabled=False,
        required_scene_tag_duration=None,
        min_marker_duration=None,
        max_gap=None,
        merge_strategy="default",
        merge_params=(None, None, None, None, None),
    )


def _migrate_csv_schema(config_path: Path, template_path: Path) -> bool:
    """Migrate existing CSV to include new columns from template.

    If the user's CSV is missing columns that exist in the template (e.g.,
    'category'), this function adds them with sensible defaults while preserving
    all existing user data.

    Args:
        config_path: Path to the user's tag_settings.csv
        template_path: Path to the template CSV with all columns

    Returns:
        True if migration was performed, False if no migration needed
    """
    import os
    import tempfile

    if not config_path.exists():
        return False

    # Read template to get expected columns and default values by tag
    template_headers: list[str] = []
    template_rows_by_tag: dict[str, dict[str, str]] = {}

    if template_path.exists():
        try:
            with template_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                template_headers = list(reader.fieldnames) if reader.fieldnames else []
                for row in reader:
                    tag_name = (
                        (row.get("tag_name") or row.get("tag") or "").strip().lower()
                    )
                    if tag_name and tag_name not in {"", "*", "default", "__default__"}:
                        template_rows_by_tag[tag_name] = dict(row)
        except Exception as exc:
            _log.warning("Failed to read template CSV for migration: %s", exc)
            template_headers = []

    # Read existing CSV
    try:
        with config_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_headers = list(reader.fieldnames) if reader.fieldnames else []
            existing_rows = list(reader)
    except Exception as exc:
        _log.warning("Failed to read existing CSV for migration: %s", exc)
        return False

    # Determine which columns are missing
    missing_headers = [h for h in template_headers if h not in existing_headers]
    if not missing_headers:
        return False  # No migration needed

    _log.info("Migrating CSV schema: adding columns %s", missing_headers)

    # Build new header list preserving existing order, adding new columns at end
    # (or in template order if we want to be fancy)
    new_headers = existing_headers.copy()
    for h in template_headers:
        if h not in new_headers:
            new_headers.append(h)

    # Update rows with defaults for new columns
    migrated_rows = []
    for row in existing_rows:
        tag_name = (row.get("tag_name") or row.get("tag") or "").strip().lower()
        template_row = template_rows_by_tag.get(tag_name, {})

        for header in missing_headers:
            if header == "category":
                # Try to get category from template, otherwise leave empty
                row[header] = template_row.get("category", "")
            else:
                # For other columns, try template default or empty
                row[header] = template_row.get(header, "")

        migrated_rows.append(row)

    # Atomic write to avoid corruption
    temp_fd, temp_path = tempfile.mkstemp(
        prefix="tag_settings_migrate_", suffix=".csv", dir=config_path.parent, text=True
    )
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=new_headers)
            writer.writeheader()
            writer.writerows(migrated_rows)

        os.replace(temp_path, config_path)
        _log.info("CSV schema migration complete: %s", config_path)
        return True
    except Exception as exc:
        _log.exception("Failed to write migrated CSV: %s", exc)
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass
        return False


class TagConfiguration:
    def __init__(
        self,
        *,
        source_path: Path,
        global_settings: TagSettings,
        overrides: Dict[str, TagSettingsOverride],
        tag_suffix: str,
    ) -> None:
        self._source_path = source_path
        self._global_settings = global_settings
        self._overrides = overrides
        self._tag_suffix = tag_suffix

    @property
    def source_path(self) -> Path:
        return self._source_path

    @property
    def global_settings(self) -> TagSettings:
        return self._global_settings

    @property
    def tag_suffix(self) -> str:
        return self._tag_suffix

    def resolve(self, tag_name: str) -> TagSettings:
        normalized = (tag_name or "").strip()
        override = self._overrides.get(normalized.lower()) if normalized else None

        suffix = (self._tag_suffix or "").strip()
        base_name = normalized

        if not override and normalized and suffix:
            norm_lower = normalized.lower()
            suffix_lower = suffix.lower()
            if norm_lower.endswith(suffix_lower) and len(normalized) > len(suffix):
                stripped = normalized[: -len(suffix)].strip()
                if stripped:
                    base_name = stripped
                    potential_override = self._overrides.get(stripped.lower())
                    if potential_override:
                        override = potential_override

        effective = replace(
            self._global_settings, tag_name=base_name or self._global_settings.tag_name
        )
        if override:
            _apply_override(effective, override)

        if not effective.stash_name:
            stash_base = base_name or normalized
            if (
                suffix
                and stash_base
                and not stash_base.lower().endswith(suffix.lower())
            ):
                effective.stash_name = stash_base + suffix
            else:
                effective.stash_name = stash_base or None

        if not effective.merge_strategy:
            effective.merge_strategy = "default"
        return effective

    def iter_overrides(self) -> Iterable[Tuple[str, TagSettingsOverride]]:
        return self._overrides.items()

    def update_tag_settings(self, tag_settings_map: Dict[str, Dict]) -> None:
        """Update full tag settings for tags in the CSV file.

        Args:
            tag_settings_map: Dictionary mapping tag names (lowercase) to settings dict with keys:
                - markers_enabled: bool (optional)
                - required_scene_tag_duration: str (optional, e.g., "15", "15s", "35%")
                - min_marker_duration: float (optional)
                - max_gap: float (optional)
        """
        import os
        import tempfile

        if not self._source_path.exists():
            _log.warning(
                "Cannot update tag settings: CSV file does not exist at %s",
                self._source_path,
            )
            return

        # Read existing CSV
        rows = []
        fieldnames = None
        try:
            with self._source_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames) if reader.fieldnames else []

                # Ensure all required columns exist
                required_cols = [
                    "markers_enabled",
                    "RequiredSceneTagDuration",
                    "min_marker_duration",
                    "max_gap",
                ]
                for col in required_cols:
                    if col not in fieldnames:
                        # Insert after appropriate column
                        if col == "markers_enabled" and "tag_name" in fieldnames:
                            fieldnames.insert(fieldnames.index("tag_name") + 1, col)
                        elif (
                            col == "RequiredSceneTagDuration"
                            and "image_enabled" in fieldnames
                        ):
                            fieldnames.insert(
                                fieldnames.index("image_enabled") + 1, col
                            )
                        elif (
                            col == "min_marker_duration"
                            and "RequiredSceneTagDuration" in fieldnames
                        ):
                            fieldnames.insert(
                                fieldnames.index("RequiredSceneTagDuration") + 1, col
                            )
                        elif col == "max_gap" and "min_marker_duration" in fieldnames:
                            fieldnames.insert(
                                fieldnames.index("min_marker_duration") + 1, col
                            )
                        else:
                            fieldnames.append(col)

                for row in reader:
                    # Get tag name (normalized)
                    tag_name = (row.get("tag_name") or row.get("tag") or "").strip()
                    if not tag_name or tag_name.lower() in {
                        "",
                        "*",
                        "default",
                        "__default__",
                    }:
                        # Keep default row as-is
                        rows.append(row)
                        continue

                    normalized_tag = tag_name.lower()
                    if normalized_tag in tag_settings_map:
                        settings = tag_settings_map[normalized_tag]

                        # Update scene_tag_enabled
                        if "scene_tag_enabled" in settings and settings["scene_tag_enabled"] is not None:
                            row["scene_tag_enabled"] = "TRUE" if settings["scene_tag_enabled"] else "FALSE"

                        # Update markers_enabled
                        if "markers_enabled" in settings and settings["markers_enabled"] is not None:
                            row["markers_enabled"] = (
                                "TRUE" if settings["markers_enabled"] else "FALSE"
                            )

                        # Update image_enabled
                        if "image_enabled" in settings and settings["image_enabled"] is not None:
                            row["image_enabled"] = "TRUE" if settings["image_enabled"] else "FALSE"

                        # Update required_scene_tag_duration
                        if (
                            "required_scene_tag_duration" in settings
                            and settings["required_scene_tag_duration"] is not None
                        ):
                            row["RequiredSceneTagDuration"] = str(
                                settings["required_scene_tag_duration"]
                            )
                        elif (
                            "required_scene_tag_duration" in settings
                            and settings["required_scene_tag_duration"] == ""
                        ):
                            row["RequiredSceneTagDuration"] = ""

                        # Update min_marker_duration
                        if "min_marker_duration" in settings:
                            if settings["min_marker_duration"] is not None:
                                row["min_marker_duration"] = str(
                                    settings["min_marker_duration"]
                                )
                            else:
                                row["min_marker_duration"] = ""

                        # Update max_gap
                        if "max_gap" in settings:
                            if settings["max_gap"] is not None:
                                row["max_gap"] = str(settings["max_gap"])
                            else:
                                row["max_gap"] = ""

                    rows.append(row)
        except Exception as exc:
            _log.exception("Failed to read CSV file for update: %s", exc)
            raise

        # Write to temporary file first (atomic write)
        temp_fd, temp_path = tempfile.mkstemp(
            prefix="tag_settings_",
            suffix=".csv",
            dir=self._source_path.parent,
            text=True,
        )
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            # Atomic rename
            os.replace(temp_path, self._source_path)
            _log.info("Updated tag settings in %s", self._source_path)

            # Reload configuration
            global _CONFIG_CACHE
            with _CONFIG_LOCK:
                _CONFIG_CACHE = TagConfiguration.load(
                    base_path=self._source_path.parent
                )
        except Exception as exc:
            # Clean up temp file on error
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
            _log.exception("Failed to write updated CSV file: %s", exc)
            raise

    @classmethod
    def load(cls, base_path: Path | None = None) -> "TagConfiguration":
        plugin_root = Path(__file__).resolve().parent
        config_root = base_path or plugin_root
        config_path = config_root / _CONFIG_FILENAME
        overrides: Dict[str, TagSettingsOverride] = {}
        global_settings = _base_settings()
        tag_suffix = _get_tag_suffix()
        # If the config file does not exist, attempt to provision it from the
        # bundled template. This makes it easier for users to get started.
        if not config_path.exists():
            template_path = plugin_root / _TEMPLATE_FILENAME
            if template_path.exists():
                try:
                    shutil.copyfile(template_path, config_path)
                    _log.info(
                        "Copied tag settings template %s to %s",
                        template_path,
                        config_path,
                    )
                except Exception as exc:  # pragma: no cover - best-effort copy
                    _log.warning(
                        "Failed to copy tag settings template %s -> %s: %s",
                        template_path,
                        config_path,
                        exc,
                    )
            else:
                _log.info(
                    "Tag settings file %s was not found and no template available; using built-in defaults",
                    config_path,
                )

        # Migrate existing CSV if it's missing columns from the template
        # This handles users upgrading from older versions
        template_path = plugin_root / _TEMPLATE_FILENAME
        if config_path.exists() and template_path.exists():
            _migrate_csv_schema(config_path, template_path)

        # If we now have a config file, parse it. Otherwise fall back to defaults.
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames is None:
                        _log.warning(
                            "Tag settings file %s is missing a header row", config_path
                        )
                    for idx, raw_row in enumerate(reader, start=2):
                        tag_key, override = _parse_row(idx, raw_row)
                        if override is None:
                            continue
                        if tag_key is None:
                            _apply_override(global_settings, override)
                            continue
                        overrides[tag_key] = override
            except Exception:
                _log.exception(
                    "Failed to read tag settings file %s; using defaults", config_path
                )

        return cls(
            source_path=config_path,
            global_settings=global_settings,
            overrides=overrides,
            tag_suffix=tag_suffix,
        )


_CONFIG_CACHE: TagConfiguration | None = None
_CONFIG_LOCK = Lock()


def get_tag_configuration(*, reload: bool = False) -> TagConfiguration:
    global _CONFIG_CACHE
    if reload:
        with _CONFIG_LOCK:
            _CONFIG_CACHE = TagConfiguration.load()
            return _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        with _CONFIG_LOCK:
            if _CONFIG_CACHE is None:
                _CONFIG_CACHE = TagConfiguration.load()
    return _CONFIG_CACHE


def resolve_backend_to_stash_tag_id(
    backend_label: str, tag_config, category: str | None
) -> int | None:
    """Resolve a backend tag label to a Stash tag ID.

    Always resolves the ID so that raw results and aggregates are persisted
    in the DB (used by the recommender).  Per-tag flags (scene_tag_enabled,
    markers_enabled, image_enabled) are checked later when tags/markers are
    actually applied to Stash.
    """
    settings = tag_config.resolve(backend_label)
    stash_name = settings.stash_name or backend_label
    if not stash_name:
        return None
    return resolve_ai_tag_reference(stash_name)


def _parse_row(
    row_number: int, raw_row: dict[str, str]
) -> tuple[str | None, TagSettingsOverride | None]:
    normalized = {}
    for key, value in (raw_row or {}).items():
        if key is None:
            continue
        norm_key = _normalize_key(key)
        if isinstance(value, str):
            normalized[norm_key] = value.strip()
        else:
            normalized[norm_key] = value
    if not any(
        str(value).strip() for value in normalized.values() if value is not None
    ):
        return None, None

    tag_value = normalized.get("tagname") or normalized.get("tag")
    tag_value = tag_value.strip() if isinstance(tag_value, str) else ""
    override = TagSettingsOverride()
    override.stash_name = _normalize_string(normalized.get("stashname"))
    override.markers_enabled = _parse_bool(normalized.get("markersenabled"))
    override.scene_tag_enabled = _parse_bool(normalized.get("scenetagenabled"))
    override.image_enabled = _parse_bool(normalized.get("imageenabled"))
    override.required_scene_tag_duration = _parse_required_scene_duration(
        normalized.get("requiredscenetagduration")
    )
    override.min_marker_duration = _parse_float(normalized.get("minmarkerduration"))
    override.max_gap = _parse_float(normalized.get("maxgap"))
    override.merge_strategy = _normalize_string(normalized.get("mergestrategy"))
    override.merge_params = tuple(
        _parse_float(normalized.get(f"markermergeparam{idx}")) for idx in range(1, 6)
    )

    if tag_value in {"", "*", "default", "__default__"}:
        return None, override
    normalized_key = tag_value.lower()
    return normalized_key, override


def _apply_override(base: TagSettings, override: TagSettingsOverride) -> None:
    if override.stash_name is not None:
        base.stash_name = override.stash_name or None
    if override.markers_enabled is not None:
        base.markers_enabled = override.markers_enabled
    if override.scene_tag_enabled is not None:
        base.scene_tag_enabled = override.scene_tag_enabled
    if override.image_enabled is not None:
        base.image_enabled = override.image_enabled
    if override.required_scene_tag_duration is not None:
        base.required_scene_tag_duration = override.required_scene_tag_duration
    if override.min_marker_duration is not None:
        base.min_marker_duration = override.min_marker_duration
    if override.max_gap is not None:
        base.max_gap = override.max_gap
    if override.merge_strategy is not None:
        base.merge_strategy = override.merge_strategy
    if override.merge_params:
        merged = list(base.merge_params)
        for idx, value in enumerate(override.merge_params):
            if value is None:
                continue
            merged[idx] = value
        base.merge_params = tuple(merged)


def _normalize_key(key: str) -> str:
    key = key.strip().lower()
    key = key.replace(" ", "")
    key = key.replace("-", "")
    key = key.replace("_", "")
    return key


def _normalize_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    result = value.strip()
    return result or None


def _parse_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        _log.warning("Unable to parse numeric value '%s' in tag configuration", value)
        return None


def _parse_required_scene_duration(value: object) -> SceneTagDurationRequirement | None:
    if value is None:
        return None
    if isinstance(value, SceneTagDurationRequirement):
        return value
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric < 0:
            _log.warning(
                "Negative required scene tag duration '%s' will be treated as zero",
                value,
            )
            numeric = 0.0
        return SceneTagDurationRequirement(unit="seconds", value=numeric)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered.endswith("s"):
            numeric_text = lowered[:-1].strip()
            try:
                numeric = float(numeric_text)
            except (TypeError, ValueError):
                _log.warning(
                    "Unable to parse seconds duration '%s' in tag configuration", value
                )
                return None
            if numeric < 0:
                _log.warning(
                    "Negative required scene tag duration '%s' will be treated as zero",
                    value,
                )
                numeric = 0.0
            return SceneTagDurationRequirement(unit="seconds", value=numeric)
        if lowered.endswith("%"):
            numeric_text = lowered[:-1].strip()
            try:
                numeric = float(numeric_text)
            except (TypeError, ValueError):
                _log.warning(
                    "Unable to parse percentage duration '%s' in tag configuration",
                    value,
                )
                return None
            return SceneTagDurationRequirement(unit="percent", value=numeric)
        try:
            numeric = float(text)
        except (TypeError, ValueError):
            _log.warning(
                "Unable to parse required scene tag duration '%s' in tag configuration",
                value,
            )
            return None
        if numeric < 0:
            _log.warning(
                "Negative required scene tag duration '%s' will be treated as zero",
                value,
            )
            numeric = 0.0
        return SceneTagDurationRequirement(unit="seconds", value=numeric)

    _log.warning(
        "Unsupported required scene tag duration value '%s' in tag configuration", value
    )
    return None


__all__ = [
    "SceneTagDurationRequirement",
    "TagSettings",
    "TagSettingsOverride",
    "TagConfiguration",
    "get_tag_configuration",
    "_CONFIG_FILENAME",
    "_TEMPLATE_FILENAME",
]
