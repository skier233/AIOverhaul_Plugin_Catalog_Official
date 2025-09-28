from app.services.registry import ServiceBase, services
from app.actions.registry import action, registry as action_registry
from app.actions.models import ContextRule, ContextInput
from app.tasks.manager import manager as task_manager
from app.tasks.models import TaskPriority
import asyncio, time

class SkierAITaggingService(ServiceBase):
    name = 'skier.ai_tagging'
    description = 'AI tagging and analysis service'
    server_url = None
    max_concurrency = 2

    @action(
        id='skier.ai_tag.image',
        label='AI Tag Image',
        description='Generate tag suggestions for an image',
        service='ai',
        result_kind='dialog',
        contexts=[ContextRule(pages=['images'], selection='single')],
    )
    async def tag_image_single(self, ctx: ContextInput, params: dict):
        selected = ctx.selected_ids or ([ctx.entity_id] if ctx.entity_id else [])
        return {'targets': selected,'tags': [{'name':'outdoor','confidence':0.92},{'name':'portrait','confidence':0.81}]}

    @action(
        id='skier.ai_tag.image',
        label='AI Tag Images',
        description='Generate tag suggestions for images',
        service='ai',
        result_kind='dialog',
        contexts=[ContextRule(pages=['images'], selection='multi')],
    )
    async def tag_image_bulk(self, ctx: ContextInput, params: dict):
        selected = ctx.selected_ids or ([ctx.entity_id] if ctx.entity_id else [])
        return {'targets': selected,'tags': [{'name':'outdoor','confidence':0.92},{'name':'portrait','confidence':0.81}]}

    @action(
        id='skier.ai_tag.scene',
        label='AI Tag Scene',
        description='Analyze a scene for tag segments',
        service='ai',
        result_kind='dialog',
        contexts=[ContextRule(pages=['scenes'], selection='single')],
    )
    async def tag_scene_single(self, ctx: ContextInput, params: dict):
        selected = ctx.selected_ids or ([ctx.entity_id] if ctx.entity_id else [])
        if not selected: selected = ['demo-scene-1']
        per_scene = []
        for sid in selected:
            remaining = 0.5
            while remaining > 0:
                chunk = min(0.05, remaining); await asyncio.sleep(chunk); remaining -= chunk
            per_scene.append({'scene_id': sid,'suggested_tags': [{'name':'hard-light','confidence':0.74},{'name':'dialogue-heavy','confidence':0.63}],'notes':'Mock inference data'})
        return {'targets': selected,'scenes': per_scene,'summary': f'{len(selected)} scene(s) processed (stub single)'}

    @action(
        id='skier.ai_tag.batch.spawn.scenes',
        label='AI Batch Spawn Scenes',
        description='Spawn individual tagging subtasks for each selected scene',
        service='ai',
        result_kind='dialog',
        contexts=[ContextRule(pages=['scenes'], selection='multi')],
        controller=True,
    )
    async def batch_spawn_scenes(self, ctx: ContextInput, params: dict, task_record):
        selected = ctx.selected_ids or []
        if not selected: return {'message':'No scenes selected'}
        reg = action_registry; spawned: list[str] = []
        for sid in selected:
            detail_ctx = ContextInput(page='scenes', entityId=sid, isDetailView=True, selectedIds=[])
            resolved = reg.resolve('ai.tag.scenes', detail_ctx)
            if not resolved: continue
            definition, handler = resolved
            child = task_manager.submit(definition, handler, detail_ctx, {}, TaskPriority.high, group_id=task_record.id)
            spawned.append(child.id)
        hold = params.get('hold_children', True)
        min_hold = float(params.get('hold_parent_seconds', 0) or 0)
        start_time = time.time()
        if hold:
            while True:
                children = [t for t in task_manager.tasks.values() if t.group_id == task_record.id]
                pending = [c for c in children if c.status not in ('completed','failed','cancelled')]
                elapsed = time.time() - start_time
                if not pending and elapsed >= min_hold: break
                await asyncio.sleep(0.1)
                if getattr(task_record, 'cancel_requested', False): break
        return {'spawned': spawned,'count': len(spawned),'held': bool(hold),'min_hold': (min_hold if min_hold else None)}

    @action(
        id='skier.ai_tag.scene',
        label='AI Tag Scenes',
        description='Analyze scenes for tag segments',
        service='ai',
        result_kind='dialog',
        contexts=[ContextRule(pages=['scenes'], selection='multi')],
    )
    async def tag_scene_bulk(self, ctx: ContextInput, params: dict):
        selected = ctx.selected_ids or ([ctx.entity_id] if ctx.entity_id else [])
        if not selected: selected = ['demo-scene-1']
        per_scene = []
        for sid in selected:
            per_scene.append({'scene_id': sid,'suggested_tags': [{'name':'hard-light','confidence':0.74},{'name':'dialogue-heavy','confidence':0.63}],'notes':'Mock inference data'})
        return {'targets': selected,'scenes': per_scene,'summary': f'{len(selected)} scene(s) processed (stub)'}


def register():
    services.register(SkierAITaggingService())
