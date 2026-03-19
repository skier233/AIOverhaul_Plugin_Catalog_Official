"""
API endpoints for Skier AI Tagging plugin tag list editor.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from sqlalchemy.orm import Session
from stash_ai_server.db.session import get_db
from stash_ai_server.api.plugins import _require_plugin_active
from stash_ai_server.services import registry as services_registry
import logging
from . import logic

logger = logging.getLogger(__name__)

# Create router with prefix for tag endpoints
# Note: main.py will add /api/v1/plugins prefix, so this should be relative to that
router = APIRouter(prefix="/settings/skier_aitagging/tags", tags=["skier_aitagging"])

PLUGIN_NAME = "skier_aitagging"


class TagSettingUpdate(BaseModel):
    """Single tag setting update."""
    scene_tag_enabled: Optional[bool] = None
    markers_enabled: Optional[bool] = None
    image_enabled: Optional[bool] = None
    required_scene_tag_duration: Optional[str] = None  # Can be "15", "15s", "35%%", etc.
    min_marker_duration: Optional[float] = None
    max_gap: Optional[float] = None


class TagSettingsUpdate(BaseModel):
    """Bulk tag settings update."""
    tag_settings: Dict[str, TagSettingUpdate]  # Map of tag name (normalized) to settings


@router.get("/available")
async def get_plugin_available_tags(db: Session = Depends(get_db)):
    """Get available tags for a plugin that supports tag editing.

    Uses the plugin service method to get tags. Service handles CSV vs legacy mode internally.
    """
    _require_plugin_active(db, PLUGIN_NAME)

    service = None
    for svc in services_registry.services.list():
        if getattr(svc, "plugin_name", None) == PLUGIN_NAME:
            service = svc
            break

    if service is None:
        raise HTTPException(status_code=404, detail="Plugin service not found")

    try:
        result = await logic.get_available_tags_data(service=service)
        return result
    except Exception as exc:
        logger.exception("Failed to get available tags for plugin %s", PLUGIN_NAME)
        raise HTTPException(status_code=500, detail=str(exc))


@router.put("/settings")
async def update_plugin_tag_settings(payload: TagSettingsUpdate, db: Session = Depends(get_db)):
    """Update full tag settings for a plugin."""
    _require_plugin_active(db, PLUGIN_NAME)

    try:
        # Convert Pydantic models to dicts
        settings_dict = {}
        for tag_name, setting_update in payload.tag_settings.items():
            settings_dict[tag_name] = {
                "scene_tag_enabled": setting_update.scene_tag_enabled,
                "markers_enabled": setting_update.markers_enabled,
                "image_enabled": setting_update.image_enabled,
                "required_scene_tag_duration": setting_update.required_scene_tag_duration,
                "min_marker_duration": setting_update.min_marker_duration,
                "max_gap": setting_update.max_gap,
            }

        result = logic.update_tag_settings(settings_dict)
        return result
    except Exception as exc:
        logger.exception("Failed to update tag settings for plugin %s", PLUGIN_NAME)
        raise HTTPException(status_code=500, detail=str(exc))


def register_routes() -> APIRouter:
    """Register and return the router for this plugin API endpoints."""
    return router
