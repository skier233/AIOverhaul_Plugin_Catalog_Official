from __future__ import annotations

import asyncio
import base64
import logging
import urllib.request

from stash_ai_server.utils.stash_api import stash_api

_log = logging.getLogger(__name__)


async def get_scene_file_info(scene_id: int) -> dict | None:
    """Fetch file path, codec, size, duration, and tag names for a scene.

    Returns dict with keys: path, video_codec, size, duration, tags — or None.
    """
    def _query():
        if not stash_api.stash_interface:
            return None
        scene = stash_api.stash_interface.find_scene(
            id=scene_id,
            fragment="files { path video_codec size duration } tags { name }",
        )
        if not scene or not scene.get("files"):
            return None
        f = scene["files"][0]
        tag_names = {(t.get("name") or "").lower() for t in scene.get("tags", []) if t.get("name")}
        return {
            "path": f.get("path"),
            "video_codec": (f.get("video_codec") or "").lower(),
            "size": int(f.get("size") or 0),
            "duration": float(f.get("duration") or 0),
            "tags": tag_names,
        }

    return await asyncio.to_thread(_query)


async def get_full_scene_metadata(scene_id: int) -> dict | None:
    """Fetch rich scene metadata for copying to a new scene."""
    def _query():
        if not stash_api.stash_interface:
            return None
        fragment = (
            "id title code date director tags { id } performers { id } "
            "studio { id } galleries { id } groups { group { id } scene_index } "
            "rating100 details urls organized o_counter "
            "paths { screenshot }"
        )
        return stash_api.stash_interface.find_scene(id=scene_id, fragment=fragment)

    return await asyncio.to_thread(_query)


async def find_scene_by_path(file_path: str) -> int | None:
    """Find a scene ID by its file path."""
    def _query():
        if not stash_api.stash_interface:
            return None
        scenes = stash_api.stash_interface.find_scenes(
            f={"path": {"value": file_path, "modifier": "EQUALS"}},
            fragment="id",
        )
        if scenes:
            return int(scenes[0]["id"])
        return None

    return await asyncio.to_thread(_query)


async def copy_scene_metadata(source_scene_id: int, target_scene_id: int) -> None:
    """Copy tags, performers, studio, galleries, rating, etc. from source to target."""
    meta = await get_full_scene_metadata(source_scene_id)
    if not meta:
        _log.warning("No metadata found for source scene %s", source_scene_id)
        return

    def _update():
        # Use singular sceneUpdate (not bulk) for full field support
        payload: dict = {"id": target_scene_id}

        # Scalar fields
        for field in ("title", "code", "date", "director", "details"):
            if meta.get(field):
                payload[field] = meta[field]
        if meta.get("rating100") is not None:
            payload["rating100"] = meta["rating100"]
        if meta.get("o_counter") is not None:
            payload["o_counter"] = meta["o_counter"]
        if meta.get("organized") is not None:
            payload["organized"] = meta["organized"]

        # URLs (Stash v0.25+ uses plural `urls`)
        urls = meta.get("urls")
        if urls:
            payload["urls"] = urls
        elif meta.get("url"):
            payload["urls"] = [meta["url"]]

        # Relationship IDs (singular mutation uses plain lists, not mode objects)
        tag_ids = [t["id"] for t in meta.get("tags", []) if "id" in t]
        if tag_ids:
            payload["tag_ids"] = tag_ids

        performer_ids = [p["id"] for p in meta.get("performers", []) if "id" in p]
        if performer_ids:
            payload["performer_ids"] = performer_ids

        studio = meta.get("studio")
        if studio and "id" in studio:
            payload["studio_id"] = studio["id"]

        gallery_ids = [g["id"] for g in meta.get("galleries", []) if "id" in g]
        if gallery_ids:
            payload["gallery_ids"] = gallery_ids

        # Groups (formerly movies in Stash <v0.27)
        groups = meta.get("groups") or meta.get("movies") or []
        group_entries = []
        for g in groups:
            grp = g.get("group") or g.get("movie") or {}
            if "id" in grp:
                group_entries.append({"group_id": grp["id"], "scene_index": g.get("scene_index")})
        if group_entries:
            payload["groups"] = group_entries

        stash_api.stash_interface.call_GQL(
            """mutation SceneUpdate($input: SceneUpdateInput!) {
                sceneUpdate(input: $input) { id }
            }""",
            variables={"input": payload},
        )

    await asyncio.to_thread(_update)

    # Copy cover image separately — requires downloading the screenshot
    # and re-uploading as a base64 data URI via sceneUpdate.
    screenshot_url = (meta.get("paths") or {}).get("screenshot")
    if screenshot_url:
        await _copy_cover_image(target_scene_id, screenshot_url)

    _log.info("Copied metadata from scene %s to scene %s", source_scene_id, target_scene_id)


async def _copy_cover_image(target_scene_id: int, screenshot_url: str) -> None:
    """Download the cover image from a URL and set it on the target scene."""
    def _do():
        if not stash_api.stash_interface:
            return
        try:
            # Build request with API key if configured
            req = urllib.request.Request(screenshot_url)
            api_key = getattr(stash_api.stash_interface, "api_key", None)
            if api_key:
                req.add_header("ApiKey", api_key)
            with urllib.request.urlopen(req, timeout=30) as resp:
                img_data = resp.read()
                content_type = resp.headers.get("Content-Type", "image/jpeg")
        except Exception as exc:
            _log.warning("Failed to download cover image from %s: %s", screenshot_url, exc)
            return

        b64 = base64.b64encode(img_data).decode("ascii")
        data_uri = f"data:{content_type};base64,{b64}"

        stash_api.stash_interface.call_GQL(
            """mutation SceneUpdate($input: SceneUpdateInput!) {
                sceneUpdate(input: $input) { id }
            }""",
            variables={"input": {"id": target_scene_id, "cover_image": data_uri}},
        )

    await asyncio.to_thread(_do)


async def tag_scene(scene_id: int, tag_name: str) -> None:
    """Add a tag to a scene by name, creating the tag if it doesn't exist."""
    def _do():
        if not stash_api.stash_interface:
            return
        # Find or create the tag
        tag = stash_api.stash_interface.find_tag(tag_name)
        if not tag:
            result = stash_api.stash_interface.create_tag({"name": tag_name})
            tag_id = result["id"] if result else None
        else:
            tag_id = tag["id"]
        if not tag_id:
            _log.warning("Could not find or create tag %r", tag_name)
            return
        # Add tag to scene
        stash_api.stash_interface.update_scenes(
            {"ids": [scene_id], "tag_ids": {"ids": [tag_id], "mode": "ADD"}}
        )

    await asyncio.to_thread(_do)
    _log.info("Tagged scene %s with %r", scene_id, tag_name)


async def untag_scene(scene_id: int, tag_name: str) -> None:
    """Remove a tag from a scene by name. No-op if the tag doesn't exist."""
    def _do():
        if not stash_api.stash_interface:
            return
        tag = stash_api.stash_interface.find_tag(tag_name)
        if not tag:
            return  # Tag doesn't exist, nothing to remove
        tag_id = tag["id"]
        stash_api.stash_interface.update_scenes(
            {"ids": [scene_id], "tag_ids": {"ids": [tag_id], "mode": "REMOVE"}}
        )

    await asyncio.to_thread(_do)
    _log.info("Removed tag %r from scene %s", tag_name, scene_id)


async def destroy_scene(scene_id: int) -> None:
    """Delete a scene entry from Stash (does NOT delete the file on disk)."""
    def _do():
        if not stash_api.stash_interface:
            return
        stash_api.stash_interface.call_GQL(
            """mutation ScenesDestroy($input: ScenesDestroyInput!) {
                scenesDestroy(input: $input)
            }""",
            variables={"input": {"ids": [scene_id], "delete_file": False, "delete_generated": True}},
        )

    await asyncio.to_thread(_do)
    _log.info("Destroyed scene %s from Stash", scene_id)


async def trigger_rescan(file_path: str) -> None:
    """Trigger a Stash metadata scan for a specific file path."""
    def _scan():
        if not stash_api.stash_interface:
            _log.warning("Stash interface not configured; cannot trigger rescan")
            return
        stash_api.stash_interface.call_GQL(
            """mutation MetadataScan($input: ScanMetadataInput!) {
                metadataScan(input: $input)
            }""",
            variables={"input": {"paths": [file_path], "rescan": True}},
        )

    await asyncio.to_thread(_scan)
    _log.info("Triggered rescan for path: %s", file_path)
