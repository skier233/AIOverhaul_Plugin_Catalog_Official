"""
API endpoints for Skier AI Tagging plugin tag list editor.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import select
from stash_ai_server.db.session import SessionLocal
from stash_ai_server.models.plugin import PluginMeta
from stash_ai_server.services import registry as services_registry
import logging

logger = logging.getLogger(__name__)

# Create router with prefix for tag endpoints
# Note: main.py will add /api/v1/plugins prefix, so this should be relative to that
router = APIRouter(prefix='/settings/skier_aitagging/tags', tags=['skier_aitagging'])

PLUGIN_NAME = 'skier_aitagging'


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _require_plugin_active(db: Session, plugin_name: str):
    meta = db.execute(select(PluginMeta).where(PluginMeta.name == plugin_name)).scalar_one_or_none()
    if meta and meta.status == 'active':
        return meta
    status = meta.status if meta else 'missing'
    message = (meta.last_error if meta and meta.last_error else 'Plugin did not activate successfully.')
    raise HTTPException(
        status_code=409,
        detail={
            'code': 'PLUGIN_INACTIVE',
            'plugin': plugin_name,
            'status': status,
            'message': message,
        },
    )


class TagStatusUpdate(BaseModel):
    tag_statuses: Optional[Dict[str, bool]] = None
    enabled_tags: Optional[List[str]] = None
    disabled_tags: Optional[List[str]] = None


@router.get('/available')
async def get_plugin_available_tags(db: Session = Depends(get_db)):
    """Get available tags for a plugin that supports tag editing.
    
    Uses the plugin's service method to get tags. Service handles CSV vs legacy mode internally.
    """
    _require_plugin_active(db, PLUGIN_NAME)
    
    # Find the plugin's service
    service = None
    for svc in services_registry.services.list():
        if getattr(svc, 'plugin_name', None) == PLUGIN_NAME:
            service = svc
            break
    
    if not service:
        raise HTTPException(status_code=404, detail='PLUGIN_SERVICE_NOT_FOUND')
    
    # Check if service has the method
    if not hasattr(service, 'get_available_tags_data'):
        raise HTTPException(status_code=400, detail='PLUGIN_DOES_NOT_SUPPORT_TAG_EDITING')
    
    try:
        # Get ALL tags (including disabled ones) for the editor
        # Service method handles CSV vs legacy mode internally
        result = await service.get_available_tags_data(include_disabled=True)
        return result
    except Exception as exc:
        logger.exception("Failed to get available tags for plugin %s", PLUGIN_NAME)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get('/statuses')
async def get_plugin_tag_statuses(db: Session = Depends(get_db)):
    """Get all tag enabled statuses for a plugin."""
    _require_plugin_active(db, PLUGIN_NAME)
    
    service = None
    for svc in services_registry.services.list():
        if getattr(svc, 'plugin_name', None) == PLUGIN_NAME:
            service = svc
            break
    
    if not service:
        raise HTTPException(status_code=404, detail='PLUGIN_SERVICE_NOT_FOUND')
    
    if not hasattr(service, 'get_all_tag_statuses'):
        raise HTTPException(status_code=400, detail='PLUGIN_DOES_NOT_SUPPORT_TAG_EDITING')
    
    try:
        # get_all_tag_statuses is async and returns dict from CSV
        result = await service.get_all_tag_statuses()
        return {'statuses': result}
    except Exception as exc:
        logger.exception("Failed to get tag statuses for plugin %s", PLUGIN_NAME)
        raise HTTPException(status_code=500, detail=str(exc))


@router.put('/statuses')
async def update_plugin_tag_statuses(payload: TagStatusUpdate, db: Session = Depends(get_db)):
    """Update tag enabled statuses for a plugin."""
    _require_plugin_active(db, PLUGIN_NAME)
    
    service = None
    for svc in services_registry.services.list():
        if getattr(svc, 'plugin_name', None) == PLUGIN_NAME:
            service = svc
            break
    
    if not service:
        raise HTTPException(status_code=404, detail='PLUGIN_SERVICE_NOT_FOUND')
    
    if not hasattr(service, 'update_tag_enabled_status'):
        raise HTTPException(status_code=400, detail='PLUGIN_DOES_NOT_SUPPORT_TAG_EDITING')
    
    try:
        result = service.update_tag_enabled_status(
            tag_statuses=payload.tag_statuses,
            enabled_tags=payload.enabled_tags,
            disabled_tags=payload.disabled_tags
        )
        return result
    except Exception as exc:
        logger.exception("Failed to update tag statuses for plugin %s", PLUGIN_NAME)
        raise HTTPException(status_code=500, detail=str(exc))


def register_routes() -> APIRouter:
    """Register and return the router for this plugin's API endpoints."""
    return router
