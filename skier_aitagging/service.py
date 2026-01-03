from __future__ import annotations
from stash_ai_server.services.base import RemoteServiceBase
from stash_ai_server.services.registry import services
from stash_ai_server.actions.registry import action
from stash_ai_server.actions.models import ContextRule, ContextInput
from stash_ai_server.tasks.models import TaskRecord
from stash_ai_server.utils.stash_api import stash_api
from . import logic


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


class SkierAITaggingService(RemoteServiceBase):
    name = "AI_Tagging"
    description = "AI tagging and analysis service"
    max_concurrency = 10
    ready_endpoint = "/ready"
    readiness_cache_seconds = 30.0
    failure_backoff_seconds = 60.0

    def __init__(self) -> None:
        super().__init__()
        self._api_key: str | None = None
        self.apply_ai_tagged_tag: bool = True
        self.reload_settings()

    def reload_settings(self) -> None:
        """Load settings from DB and environment variables."""
        cfg = self._load_settings()
        
        # Load server URL
        server_setting = cfg.get("server_url")
        if server_setting is not None:
            self.server_url = server_setting or None

        self.apply_ai_tagged_tag = _coerce_bool(cfg.get("apply_ai_tagged_tag"), True)

    # ------------------------------------------------------------------
    # Image actions
    # ------------------------------------------------------------------

    @action(
        id="skier.ai_tag.image",
        label="AI Tag Image",
        description="Generate tag suggestions for an image",
        result_kind="dialog",
        contexts=[ContextRule(pages=["images"], selection="single")],
    )
    async def tag_image_single(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        return await logic.tag_images(self, ctx, params, task_record)

    @action(
        id="skier.ai_tag.image.selected",
        label="Tag Selected Images",
        description="Generate tag suggestions for selected images",
        result_kind="dialog",
        contexts=[ContextRule(pages=["images"], selection="multi")],
    )
    async def tag_image_selected(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        return await logic.tag_images(self, ctx, params, task_record)

    @action(
        id="skier.ai_tag.image.page",
        label="Tag Page Images",
        description="Generate tag suggestions for all images on the current page",
        result_kind="dialog",
        contexts=[ContextRule(pages=["images"], selection="page")],
    )
    async def tag_image_page(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        return await logic.tag_images(self, ctx, params, task_record)

    @action(
        id="skier.ai_tag.image.all",
        label="Tag All Images",
        description="Analyze every image in the library",
        result_kind="dialog",
        contexts=[ContextRule(pages=["images"], selection="none")],
    )
    async def tag_image_all(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        ctx.selected_ids = await stash_api.get_all_images_async()
        return await logic.tag_images(self, ctx, params, task_record)

    # ------------------------------------------------------------------
    # Scene actions - use controller pattern to spawn child tasks
    # ------------------------------------------------------------------

    @action(
        id="skier.ai_tag.scene",
        label="AI Tag Scene",
        description="Analyze a scene for tag segments",
        result_kind="dialog",
        contexts=[ContextRule(pages=["scenes"], selection="single")],
    )
    async def tag_scene_single(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        return await logic.tag_scenes(self, ctx, params, task_record)

    @action(
        id="skier.ai_tag.scene.selected",
        label="Tag Selected Scenes",
        description="Analyze selected scenes for tag segments",
        result_kind="dialog",
        contexts=[ContextRule(pages=["scenes"], selection="multi")],
    )
    async def tag_scene_selected(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        return await logic.tag_scenes(self, ctx, params, task_record)

    @action(
        id="skier.ai_tag.scene.page",
        label="Tag Page Scenes",
        description="Analyze every scene visible in the current list view",
        result_kind="dialog",
        contexts=[ContextRule(pages=["scenes"], selection="page")],
    )
    async def tag_scene_page(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        return await logic.tag_scenes(self, ctx, params, task_record)

    @action(
        id="skier.ai_tag.scene.all",
        label="Tag All Scenes",
        description="Analyze every scene in the library",
        result_kind="dialog",
        contexts=[ContextRule(pages=["scenes"], selection="none")],
    )
    async def tag_scene_all(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        ctx.selected_ids = await stash_api.get_all_scenes_async()
        return await logic.tag_scenes(self, ctx, params, task_record)

    # ------------------------------------------------------------------
    # Tag configuration methods (for plugin endpoints)
    # ------------------------------------------------------------------

    async def get_available_tags_data(self, include_disabled: bool = False) -> dict:
        """Get available tags from CSV file.
        
        Args:
            include_disabled: If True, include all tags regardless of enabled status.
                           If False (default), only return enabled tags.
        
        Returns:
            dict with 'tags' and 'models' keys. Tags are a flat list from CSV.
        """
        import csv
        import logging
        from . import tag_config
        
        _log = logging.getLogger(__name__)
        
        # Get tag config
        tag_config_obj = tag_config.get_tag_configuration()
        
        # Read tags directly from CSV file
        tags_list = []
        csv_path = tag_config_obj.source_path
        
        if not csv_path.exists():
            _log.warning("Tag settings CSV file does not exist at %s", csv_path)
            return {'tags': [], 'models': []}
        
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    _log.warning("Tag settings CSV file is missing a header row")
                    return {'tags': [], 'models': []}
                
                for row in reader:
                    # Get tag name from CSV
                    tag_name = (row.get('tag_name') or row.get('tag') or '').strip()
                    
                    # Skip default/empty rows
                    if not tag_name or tag_name.lower() in {'', '*', 'default', '__default__'}:
                        continue
                    
                    # Check if tag should be included based on enabled status
                    if not include_disabled:
                        is_enabled = tag_config_obj.get_tag_enabled_status(tag_name)
                        if not is_enabled:
                            continue
                    
                    # Add tag to list (normalized name is used as key, but we keep original for display)
                    tags_list.append({
                        'tag': tag_name,
                        'name': tag_name,  # For compatibility
                    })
        except Exception as exc:
            _log.exception("Failed to read tags from CSV file %s: %s", csv_path, exc)
            return {'tags': [], 'models': [], 'error': f'Failed to read CSV: {str(exc)}'}
        
        return {'tags': tags_list, 'models': []}

    def get_enabled_tags_list(self) -> list[str]:
        """Get list of enabled tag names (normalized, lowercase)."""
        from . import tag_config
        tag_config_obj = tag_config.get_tag_configuration()
        return tag_config_obj.get_enabled_tags()

    async def get_all_tag_statuses(self) -> dict[str, bool]:
        """Get all tag enabled statuses from CSV file.
        
        Returns:
            Dictionary mapping tag names (normalized, lowercase) to enabled status.
        """
        from . import tag_config
        
        # Get statuses from CSV only (CSV is the source of truth)
        tag_config_obj = tag_config.get_tag_configuration()
        return tag_config_obj.get_all_tag_statuses()

    def update_tag_enabled_status(self, tag_statuses: dict[str, bool] | None = None, enabled_tags: list[str] | None = None, disabled_tags: list[str] | None = None) -> dict:
        """Update enabled status for tags.
        
        Args:
            tag_statuses: Dictionary mapping tag names (normalized, lowercase) to enabled status (preferred)
            enabled_tags: List of tag names to enable (alternative to tag_statuses)
            disabled_tags: List of tag names to disable (alternative to tag_statuses)
        
        Returns:
            dict with 'status' and 'updated' count
        """
        from . import tag_config
        tag_config_obj = tag_config.get_tag_configuration()
        
        # If tag_statuses provided, use it directly
        if tag_statuses is not None:
            tag_config_obj.update_tag_enabled_status(tag_statuses)
            return {'status': 'ok', 'updated': len(tag_statuses)}
        
        # Otherwise, get current statuses and update based on enabled/disabled lists
        current_statuses = tag_config_obj.get_all_tag_statuses()
        updated_map = dict(current_statuses)
        
        if enabled_tags:
            for tag in enabled_tags:
                updated_map[tag.lower()] = True
        
        if disabled_tags:
            for tag in disabled_tags:
                updated_map[tag.lower()] = False
        
        tag_config_obj.update_tag_enabled_status(updated_map)
        return {'status': 'ok', 'updated': len(updated_map)}

def register():
    services.register(SkierAITaggingService())