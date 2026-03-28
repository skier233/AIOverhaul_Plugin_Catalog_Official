(function () {
  var register = window.AIRegisterResultFormatter;
  if (!register) return;

  // Single scene tagging
  register({
    id: "aitagging:single",
    priority: 5,
    match: function (r) {
      return r && typeof r === "object" && "scene_id" in r && "tags_applied" in r;
    },
    format: function (r) {
      var tagsCount = r.tags_applied || 0;
      var sceneId = r.scene_id;
      return {
        message: "Applied " + tagsCount + " tag" + (tagsCount !== 1 ? "s" : "") + " to scene",
        type: "success",
        link: { url: window.location.origin + "/scenes/" + sceneId + "/", text: "view" },
        fullDetails: r,
      };
    },
  });

  // Batch tagging
  register({
    id: "aitagging:batch",
    priority: 5,
    match: function (r) {
      return r && typeof r === "object" && "scenes_completed" in r && !("total_savings_mb" in r);
    },
    format: function (r) {
      var ok = r.scenes_completed || 0;
      var failed = r.scenes_failed || 0;
      var parts = [];
      parts.push(ok + " scene" + (ok !== 1 ? "s" : "") + " tagged");
      if (failed > 0) parts.push(failed + " scene" + (failed !== 1 ? "s" : "") + " failed");
      return {
        message: parts.join(", "),
        type: failed > 0 && ok === 0 ? "error" : "success",
        fullDetails: r,
      };
    },
  });
})();
