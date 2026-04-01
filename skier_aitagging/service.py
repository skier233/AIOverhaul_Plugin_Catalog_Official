from __future__ import annotations
import logging
from stash_ai_server.services.base import RemoteServiceBase
from stash_ai_server.services.registry import services
from stash_ai_server.actions.registry import action, registry as action_registry
from stash_ai_server.actions.models import ContextRule, ContextInput
from stash_ai_server.tasks.models import TaskRecord
from stash_ai_server.utils.stash_api import stash_api
from . import logic

_log = logging.getLogger(__name__)

_FACE_SCAN_ACTION_IDS = [
    "skier.face_scan.image",
    "skier.face_scan.image.selected",
    "skier.face_scan.image.page",
    "skier.face_scan.image.all",
    "skier.face_scan.scene",
    "skier.face_scan.scene.selected",
    "skier.face_scan.scene.page",
    "skier.face_scan.scene.all",
]


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
    max_concurrency = 3
    ready_endpoint = "/ready"
    readiness_cache_seconds = 30.0
    failure_backoff_seconds = 60.0

    def __init__(self) -> None:
        super().__init__()
        self._api_key: str | None = None
        self.apply_ai_tagged_tag: bool = True
        self.tagging_frame_interval: float = 2.0
        self.face_scan_frame_interval: float = 6.0
        # Face recognition settings
        self.auto_apply_performers: bool = True
        self.face_match_auto_threshold: float = 0.50
        self.face_match_review_threshold: float = 0.40
        self.face_max_exemplars_per_cluster: int = 10
        self.face_max_embeddings_per_track: int = 10
        self.face_embedding_dedup_threshold: float = 0.85
        self.face_min_embedding_norm: float = 18.0
        self.face_min_detection_score: float = 0.65
        self.face_hard_min_embedding_norm: float = 10.0
        self.face_hard_min_detection_score: float = 0.30
        self.face_max_embeddings_per_cluster: int = 20
        # Min appearances before auto-creating a performer (0 = no gate)
        self.face_auto_create_min_scenes: int = 0
        self.face_auto_create_min_images: int = 0
        self.reload_settings()

    def reload_settings(self) -> None:
        """Load settings from DB and environment variables."""
        cfg = self._load_settings()
        
        # Load server URL
        server_setting = cfg.get("server_url")
        if server_setting is not None:
            self.server_url = server_setting or None

        self.apply_ai_tagged_tag = _coerce_bool(cfg.get("apply_ai_tagged_tag"), True)
        self.tagging_frame_interval = float(cfg.get("tagging_frame_interval", 2.0))
        self.face_scan_frame_interval = float(cfg.get("face_scan_frame_interval", 6.0))

        # Face recognition settings
        self.auto_apply_performers = _coerce_bool(cfg.get("auto_apply_performers"), True)
        self.face_match_auto_threshold = float(cfg.get("face_match_auto_threshold", 0.50))
        self.face_match_review_threshold = float(cfg.get("face_match_review_threshold", 0.40))
        self.face_max_exemplars_per_cluster = int(float(cfg.get("face_max_exemplars_per_cluster", 10)))
        self.face_max_embeddings_per_track = int(float(cfg.get("face_max_embeddings_per_track", 10)))
        self.face_embedding_dedup_threshold = float(cfg.get("face_embedding_dedup_threshold", 0.85))
        self.face_min_embedding_norm = float(cfg.get("face_min_embedding_norm", 18.0))
        self.face_min_detection_score = float(cfg.get("face_min_detection_score", 0.65))
        self.face_hard_min_embedding_norm = float(cfg.get("face_hard_min_embedding_norm", 10.0))
        self.face_hard_min_detection_score = float(cfg.get("face_hard_min_detection_score", 0.30))
        self.face_max_embeddings_per_cluster = int(float(cfg.get("face_max_embeddings_per_cluster", 20)))
        self.face_auto_create_min_scenes = int(float(cfg.get("face_auto_create_min_scenes", 0)))
        self.face_auto_create_min_images = int(float(cfg.get("face_auto_create_min_images", 0)))

    # ------------------------------------------------------------------
    # Backend capabilities
    # ------------------------------------------------------------------

    _backend_capabilities: dict | None = None
    _face_actions_removed: bool = False

    async def _fetch_backend_capabilities(self) -> dict | None:
        """Fetch /v3/capabilities/ from the remote AI backend (best-effort)."""
        try:
            resp = await self.http.get("/v3/capabilities/")
            if isinstance(resp, dict):
                return resp
            # httpx.Response fallback
            if hasattr(resp, "json"):
                return resp.json()
        except Exception as exc:
            _log.debug("Failed to fetch backend capabilities: %s", exc)
        return None

    async def _sync_face_actions(self) -> None:
        """Remove face-scan actions if the backend lacks face_recognition."""
        caps = await self._fetch_backend_capabilities()
        if caps is None:
            return  # can't determine — leave actions as-is
        self._backend_capabilities = caps
        face_supported = bool(caps.get("face_recognition"))
        if not face_supported and not self._face_actions_removed:
            removed = action_registry.unregister_actions(_FACE_SCAN_ACTION_IDS)
            self._face_actions_removed = True
            _log.info(
                "Backend does not support face_recognition — "
                "removed %d face-scan actions", removed,
            )

    async def _probe_capabilities_on_startup(self) -> None:
        """Best-effort startup probe: fetch capabilities as soon as backend responds.

        Retries briefly so that face-scan actions are hidden before the user
        sees the action list, even if the backend takes a few seconds to boot.
        """
        import asyncio as _asyncio
        if not self.server_url:
            return
        for delay in (0, 2, 5, 10):
            if delay:
                await _asyncio.sleep(delay)
            ready = await self.ensure_remote_ready()
            if ready:
                return  # _sync_face_actions already called inside ensure_remote_ready
            if self._backend_capabilities is not None:
                return  # already resolved on a previous retry
        _log.debug("Startup capabilities probe exhausted retries — will check on first task run")

    async def ensure_remote_ready(self, *, force: bool = False) -> bool:
        ready = await super().ensure_remote_ready(force=force)
        if ready and self._backend_capabilities is None:
            await self._sync_face_actions()
        return ready

    def schedule_startup_probe(self) -> None:
        """Fire a background capabilities probe without blocking the caller."""
        import asyncio as _asyncio
        try:
            loop = _asyncio.get_running_loop()
            loop.create_task(self._probe_capabilities_on_startup())
        except RuntimeError:
            pass  # no running loop yet — probe will happen on first task

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
    # Face-scan-only actions (no tag classification, just face detection)
    # ------------------------------------------------------------------

    @action(
        id="skier.face_scan.image",
        label="Scan Image Faces",
        description="Detect and identify faces in an image (no tagging)",
        result_kind="dialog",
        contexts=[ContextRule(pages=["images"], selection="single")],
    )
    async def face_scan_image_single(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        return await logic.face_scan_images(self, ctx, params, task_record)

    @action(
        id="skier.face_scan.image.selected",
        label="Scan Selected Image Faces",
        description="Detect and identify faces in selected images",
        result_kind="dialog",
        contexts=[ContextRule(pages=["images"], selection="multi")],
    )
    async def face_scan_image_selected(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        return await logic.face_scan_images(self, ctx, params, task_record)

    @action(
        id="skier.face_scan.image.page",
        label="Scan Page Image Faces",
        description="Detect and identify faces in all images on the current page",
        result_kind="dialog",
        contexts=[ContextRule(pages=["images"], selection="page")],
    )
    async def face_scan_image_page(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        return await logic.face_scan_images(self, ctx, params, task_record)

    @action(
        id="skier.face_scan.image.all",
        label="Scan All Image Faces",
        description="Detect faces in every image in the library",
        result_kind="dialog",
        contexts=[ContextRule(pages=["images"], selection="none")],
    )
    async def face_scan_image_all(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        ctx.selected_ids = await stash_api.get_all_images_async()
        return await logic.face_scan_images(self, ctx, params, task_record)

    @action(
        id="skier.face_scan.scene",
        label="Scan Scene Faces",
        description="Detect and identify faces in a scene (no tagging)",
        result_kind="dialog",
        contexts=[ContextRule(pages=["scenes"], selection="single")],
    )
    async def face_scan_scene_single(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        return await logic.face_scan_scenes(self, ctx, params, task_record)

    @action(
        id="skier.face_scan.scene.selected",
        label="Scan Selected Scene Faces",
        description="Detect and identify faces in selected scenes",
        result_kind="dialog",
        contexts=[ContextRule(pages=["scenes"], selection="multi")],
    )
    async def face_scan_scene_selected(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        return await logic.face_scan_scenes(self, ctx, params, task_record)

    @action(
        id="skier.face_scan.scene.page",
        label="Scan Page Scene Faces",
        description="Detect and identify faces in every scene on the current page",
        result_kind="dialog",
        contexts=[ContextRule(pages=["scenes"], selection="page")],
    )
    async def face_scan_scene_page(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        return await logic.face_scan_scenes(self, ctx, params, task_record)

    @action(
        id="skier.face_scan.scene.all",
        label="Scan All Scene Faces",
        description="Detect faces in every scene in the library",
        result_kind="dialog",
        contexts=[ContextRule(pages=["scenes"], selection="none")],
    )
    async def face_scan_scene_all(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        ctx.selected_ids = await stash_api.get_all_scenes_async()
        return await logic.face_scan_scenes(self, ctx, params, task_record)

def register():
    svc = SkierAITaggingService()
    services.register(svc)
    svc.schedule_startup_probe()
    
    # Register plugin routers for API endpoints
    from stash_ai_server.plugin_runtime import loader as plugin_loader
    from . import api_endpoints
    from . import face_api
    from . import xray_api
    plugin_loader.register_plugin_router('skier_aitagging', api_endpoints.register_routes())
    plugin_loader.register_plugin_router('skier_aitagging', face_api.register_routes())
    plugin_loader.register_plugin_router('skier_aitagging', xray_api.register_routes())