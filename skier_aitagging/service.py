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
        """Get available tags from CSV file with full settings.
        
        Args:
            include_disabled: If True, include all tags regardless of enabled status.
                           If False (default), only return enabled tags.
        
        Returns:
            dict with 'tags' (full settings), 'models', and 'defaults' keys.
        """
        import csv
        import logging
        from . import tag_config
        from .http_handler import get_active_scene_models
        
        _log = logging.getLogger(__name__)
        
        # Get tag config
        tag_config_obj = tag_config.get_tag_configuration()
        
        # Read tags directly from CSV file
        tags_list = []
        defaults = {}
        csv_path = tag_config_obj.source_path
        
        if not csv_path.exists():
            _log.warning("Tag settings CSV file does not exist at %s", csv_path)
            return {'tags': [], 'models': [], 'defaults': {}}
        
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    _log.warning("Tag settings CSV file is missing a header row")
                    return {'tags': [], 'models': [], 'defaults': {}}
                
                for row in reader:
                    # Get tag name from CSV
                    tag_name = (row.get('tag_name') or row.get('tag') or '').strip()
                    
                    # Extract default values from __default__ row
                    if tag_name.lower() == '__default__':
                        defaults['required_scene_tag_duration'] = row.get('RequiredSceneTagDuration', '').strip()
                        defaults['min_marker_duration'] = row.get('min_marker_duration', '').strip()
                        defaults['max_gap'] = row.get('max_gap', '').strip()
                        defaults['markers_enabled'] = row.get('markers_enabled', 'TRUE').strip().upper() == 'TRUE'
                        continue
                    
                    # Skip empty rows
                    if not tag_name or tag_name.lower() in {'', '*', 'default', 'unused1', 'unused2', 'unused3', 'unused4'}:
                        continue
                    
                    # Get resolved settings for this tag
                    settings = tag_config_obj.resolve(tag_name)
                    
                    # Check if tag should be included based on enabled status
                    if not include_disabled:
                        if not settings.enabled:
                            continue
                    
                    # Get category from CSV row
                    category = row.get('category', '').strip() or 'Other'
                    
                    # Format required_scene_tag_duration
                    req_duration_str = None
                    if settings.required_scene_tag_duration:
                        if settings.required_scene_tag_duration.unit == 'percent':
                            req_duration_str = f"{settings.required_scene_tag_duration.value}%"
                        else:
                            req_duration_str = str(settings.required_scene_tag_duration.value)
                    
                    # Add tag with full settings
                    tags_list.append({
                        'tag': tag_name,
                        'name': tag_name,  # For compatibility
                        'category': category,
                        'enabled': settings.enabled,
                        'markers_enabled': settings.markers_enabled,
                        'required_scene_tag_duration': req_duration_str,
                        'min_marker_duration': settings.min_marker_duration,
                        'max_gap': settings.max_gap,
                    })
        except Exception as exc:
            _log.exception("Failed to read tags from CSV file %s: %s", csv_path, exc)
            return {'tags': [], 'models': [], 'defaults': {}, 'error': f'Failed to read CSV: {str(exc)}'}
        
        # Fetch active models from nsfw backend
        active_models = []
        loaded_categories = set()
        try:
            active_models_list = await get_active_scene_models(self)
            if active_models_list:
                for model in active_models_list:
                    # Convert AIModelInfo to dict for JSON serialization
                    model_dict = {
                        'name': model.name,
                        'identifier': model.identifier,
                        'version': model.version,
                        'categories': model.categories,
                        'type': model.type,
                    }
                    active_models.append(model_dict)
                    # Extract all categories from this model
                    if model.categories:
                        loaded_categories.update(model.categories)
        except Exception as exc:
            # If backend is unavailable, log warning but continue (graceful degradation)
            _log.warning("Failed to fetch active models from nsfw backend: %s. Showing all tags.", exc)
        
        return {
            'tags': tags_list,
            'models': active_models,
            'loaded_categories': list(loaded_categories),
            'defaults': defaults
        }

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

    def update_tag_settings(self, tag_settings: dict) -> dict:
        """Update full tag settings for multiple tags.
        
        Args:
            tag_settings: Dictionary mapping tag names (normalized, lowercase) to dicts with:
                - enabled: bool (optional)
                - markers_enabled: bool (optional)
                - required_scene_tag_duration: str (optional, e.g., "15", "15s", "35%")
                - min_marker_duration: float (optional)
                - max_gap: float (optional)
        
        Returns:
            dict with 'status' and 'updated' count
        """
        from . import tag_config
        tag_config_obj = tag_config.get_tag_configuration()
        
        # Convert to format expected by tag_config
        settings_map = {}
        for tag_name, settings in tag_settings.items():
            settings_map[tag_name] = {
                'enabled': settings.get('enabled'),
                'markers_enabled': settings.get('markers_enabled'),
                'required_scene_tag_duration': settings.get('required_scene_tag_duration'),
                'min_marker_duration': settings.get('min_marker_duration'),
                'max_gap': settings.get('max_gap'),
            }
        
        tag_config_obj.update_tag_settings(settings_map)
        return {'status': 'ok', 'updated': len(tag_settings)}

def register():
    services.register(SkierAITaggingService())
    
    # Register plugin router for API endpoints
    from stash_ai_server.plugin_runtime import loader as plugin_loader
    from . import api_endpoints
    plugin_loader.register_plugin_router('skier_aitagging', api_endpoints.register_routes())