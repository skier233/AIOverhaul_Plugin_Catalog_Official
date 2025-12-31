import logging
from stash_ai_server.utils.stash_api import stash_api

_log = logging.getLogger(__name__)

# Lazy initialization cache
_tag_cache: dict[str, int | None] = {}
_ai_tags_cache: dict[str, int] | None = None
_vr_tag_name: str | None = None

AI_Base_Tag_Name = "AI"
AI_Error_Tag_Name = "AI_Errored"
AI_Tagged_Tag_Name = "AI_Tagged"
AI_Reprocess_Tag_Name = "AI_Reprocess"


def _to_int_list(values: list[int | str] | None) -> list[int]:
    out: list[int] = []
    if not values:
        return out
    for v in values:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def _ensure_stash_interface() -> bool:
    """Ensure stash_interface is initialized before use."""
    if stash_api.stash_interface is None:
        _log.warning("Stash interface not initialized yet")
        return False
    return True


def _get_ai_base_tag_id() -> int | None:
    """Get the AI base tag ID, initializing if needed."""
    if not _ensure_stash_interface():
        return None
    if "AI_Base_Tag_Id" not in _tag_cache:
        _tag_cache["AI_Base_Tag_Id"] = stash_api.fetch_tag_id(AI_Base_Tag_Name, create_if_missing=True)
    return _tag_cache["AI_Base_Tag_Id"]


def _get_ai_error_tag_id() -> int | None:
    """Get the AI error tag ID, initializing if needed."""
    if not _ensure_stash_interface():
        return None
    if "AI_Error_Tag_Id" not in _tag_cache:
        base_id = _get_ai_base_tag_id()
        if base_id is None:
            return None
        _tag_cache["AI_Error_Tag_Id"] = stash_api.fetch_tag_id(AI_Error_Tag_Name, parent_id=base_id, create_if_missing=True)
    return _tag_cache["AI_Error_Tag_Id"]


def _get_ai_tagged_tag_id() -> int | None:
    """Get the AI tagged tag ID, initializing if needed."""
    if not _ensure_stash_interface():
        return None
    if "AI_Tagged_Tag_Id" not in _tag_cache:
        _tag_cache["AI_Tagged_Tag_Id"] = stash_api.fetch_tag_id(AI_Tagged_Tag_Name, create_if_missing=True)
    return _tag_cache["AI_Tagged_Tag_Id"]


def _get_ai_reprocess_tag_id() -> int | None:
    """Get the AI reprocess tag ID, initializing if needed."""
    if not _ensure_stash_interface():
        return None
    if "AI_Reprocess_Tag_Id" not in _tag_cache:
        base_id = _get_ai_base_tag_id()
        if base_id is None:
            return None
        _tag_cache["AI_Reprocess_Tag_Id"] = stash_api.fetch_tag_id(AI_Reprocess_Tag_Name, parent_id=base_id, create_if_missing=True)
    return _tag_cache["AI_Reprocess_Tag_Id"]


def _get_vr_tag_name() -> str | None:
    """Get the VR tag name from Stash configuration."""
    global _vr_tag_name
    if _vr_tag_name is None and _ensure_stash_interface():
        try:
            config = stash_api.stash_interface.get_configuration()
            _vr_tag_name = config.get("ui", {}).get("vrTag", None)
        except Exception:
            _log.exception("Failed to get VR tag name from configuration")
            _vr_tag_name = None
    return _vr_tag_name


def _get_vr_tag_id() -> int | None:
    """Get the VR tag ID, initializing if needed."""
    if not _ensure_stash_interface():
        return None
    if "VR_Tag_Id" not in _tag_cache:
        vr_name = _get_vr_tag_name()
        _tag_cache["VR_Tag_Id"] = stash_api.fetch_tag_id(vr_name) if vr_name else None
    return _tag_cache["VR_Tag_Id"]


def _get_ai_tags_cache() -> dict[str, int]:
    """Get the AI tags cache, initializing if needed."""
    global _ai_tags_cache
    if _ai_tags_cache is None:
        _ai_tags_cache = {}
        if _ensure_stash_interface():
            base_id = _get_ai_base_tag_id()
            if base_id is not None:
                try:
                    _ai_tags_cache = stash_api.get_tags_with_parent(parent_tag_id=base_id)
                except Exception:
                    _log.exception("Failed to get AI tags with parent")
                    _ai_tags_cache = {}
                
                # Add the standard AI tags to cache
                error_id = _get_ai_error_tag_id()
                if error_id is not None:
                    _ai_tags_cache[AI_Error_Tag_Name] = error_id
                
                tagged_id = _get_ai_tagged_tag_id()
                if tagged_id is not None:
                    _ai_tags_cache[AI_Tagged_Tag_Name] = tagged_id
                
                reprocess_id = _get_ai_reprocess_tag_id()
                if reprocess_id is not None:
                    _ai_tags_cache[AI_Reprocess_Tag_Name] = reprocess_id
    return _ai_tags_cache


# Backward compatibility: module-level accessors that lazily initialize
def get_ai_base_tag_id() -> int | None:
    """Get the AI base tag ID."""
    return _get_ai_base_tag_id()


def get_ai_error_tag_id() -> int | None:
    """Get the AI error tag ID."""
    return _get_ai_error_tag_id()


def get_ai_tagged_tag_id() -> int | None:
    """Get the AI tagged tag ID."""
    return _get_ai_tagged_tag_id()


def get_ai_reprocess_tag_id() -> int | None:
    """Get the AI reprocess tag ID."""
    return _get_ai_reprocess_tag_id()


def get_vr_tag_id() -> int | None:
    """Get the VR tag ID."""
    return _get_vr_tag_id()


def get_ai_tags_cache() -> dict[str, int]:
    """Get the AI tags cache."""
    return _get_ai_tags_cache()


def has_ai_tagged(tags: list[int | str]) -> bool:
    """Check if the scene has the AI_Tagged tag."""
    tagged_id = get_ai_tagged_tag_id()
    normalized = _to_int_list(tags)
    return tagged_id in normalized if tagged_id else False


def has_ai_reprocess(tags: list[int | str]) -> bool:
    """Check if the AI_Reprocess tag is applied."""
    reprocess_id = get_ai_reprocess_tag_id()
    normalized = _to_int_list(tags)
    return reprocess_id in normalized if reprocess_id else False


def remove_ai_tags_from_images(image_ids: list[int | str]) -> None:
    """Remove all AI tags from the given images."""
    cache = get_ai_tags_cache()
    if not cache:
        _log.warning("No AI tags in cache; nothing to remove")
        return
    stash_api.remove_tags_from_images(_to_int_list(image_ids), list(cache.values()))


def add_error_tag_to_images(image_ids: list[int | str]) -> None:
    """Add the AI_Errored tag to the given images."""
    error_id = get_ai_error_tag_id()
    if error_id is None:
        _log.warning("AI_Error_Tag_Id is None; cannot add error tag")
        return
    stash_api.add_tags_to_images(_to_int_list(image_ids), [error_id])


async def remove_reprocess_tag_from_scene(scene_id: int) -> None:
    """Remove AI_Reprocess from a scene once reprocessing is finished."""
    reprocess_id = get_ai_reprocess_tag_id()
    if reprocess_id is None:
        return
    try:
        await stash_api.remove_tags_from_scene_async(scene_id, [reprocess_id])
    except Exception:
        _log.exception("Failed to remove AI_Reprocess tag from scene_id=%s", scene_id)


async def remove_reprocess_tag_from_images(image_ids: list[int | str]) -> None:
    """Remove AI_Reprocess from images that finished reprocessing."""
    reprocess_id = get_ai_reprocess_tag_id()
    if reprocess_id is None or not image_ids:
        return
    try:
        await stash_api.remove_tags_from_images_async(_to_int_list(image_ids), [reprocess_id])
    except Exception:
        _log.exception("Failed to remove AI_Reprocess tag from image_ids=%s", image_ids)


def get_ai_tag_ids_from_names(tag_names: list[str]) -> list[int]:
    """Get tag IDs for the given tag names, creating them if necessary."""
    base_id = get_ai_base_tag_id()
    cache = get_ai_tags_cache()
    return [stash_api.fetch_tag_id(tag, parent_id=base_id, create_if_missing=True, add_to_cache=cache) for tag in tag_names]


def resolve_ai_tag_reference(label: str) -> int | None:
    """Resolve (and ensure) the Stash tag id for a label used in AI results."""
    if not label:
        return None
    try:
        base_id = get_ai_base_tag_id()
        cache = get_ai_tags_cache()
        return stash_api.fetch_tag_id(
            label,
            parent_id=base_id,
            create_if_missing=True,
            add_to_cache=cache,
        )
    except Exception:
        _log.exception("Failed to resolve AI tag reference for label=%s", label)
        return None


def is_vr_scene(tag_ids: list[int | str]) -> bool:
    """Check if the scene is tagged as VR."""
    vr_id = get_vr_tag_id()
    normalized = _to_int_list(tag_ids)
    return vr_id in normalized if vr_id else False
