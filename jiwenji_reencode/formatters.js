(function () {
  var register = window.AIRegisterResultFormatter;
  if (!register) return;

  function formatSavingsSize(mb) {
    if (mb >= 10000 * 1024) return (mb / (1024 * 1024)).toFixed(1) + "TB";
    if (mb >= 10000) return (mb / 1024).toFixed(1) + "GB";
    return mb.toFixed(1) + "MB";
  }

  function formatSavingsSizeBytes(bytes) {
    return formatSavingsSize(bytes / (1024 * 1024));
  }

  // Single scene reencode
  register({
    id: "reencode:single",
    priority: 10,
    match: function (r) {
      return r && typeof r === "object" && "scene_id" in r && "method_used" in r;
    },
    format: function (r) {
      var sceneId = r.scene_id;
      var tagged = r.tag_queued;
      var msg;
      if (r.status === "skipped") {
        msg = r.message || "Scene skipped";
      } else if (r.status === "failed") {
        msg = r.message || "Re-encode failed";
      } else {
        var savings = "";
        if (r.savings_pct) {
          var savingsStr = r.original_size && r.new_size ? formatSavingsSizeBytes(r.original_size - r.new_size) : null;
          savings = savingsStr ? " (" + savingsStr + ", " + r.savings_pct + "% saved)" : " (" + r.savings_pct + "% saved)";
        }
        msg = tagged ? "Re-encoded and tagged scene" + savings : "Re-encoded scene" + savings;
      }
      return {
        message: msg,
        type: r.status === "failed" ? "error" : "success",
        link: { url: window.location.origin + "/scenes/" + sceneId + "/", text: "view" },
        fullDetails: r,
      };
    },
  });

  // Batch reencode
  register({
    id: "reencode:batch",
    priority: 10,
    match: function (r) {
      return r && typeof r === "object" && "scenes_completed" in r && "total_savings_mb" in r;
    },
    format: function (r) {
      var ok = r.scenes_completed || 0;
      var failed = r.scenes_failed || 0;
      var skipped = r.scenes_skipped || 0;
      var savingsMB = r.total_savings_mb || 0;
      var savingsPct = r.total_savings_pct || 0;
      var parts = [];
      if (ok > 0) parts.push(ok + " re-encoded");
      if (skipped > 0) parts.push(skipped + " skipped");
      if (failed > 0) parts.push(failed + " failed");
      if (savingsMB > 0) {
        var sizeStr = formatSavingsSize(savingsMB);
        parts.push(savingsPct > 0 ? "(" + sizeStr + ", " + savingsPct + "% saved)" : "(" + sizeStr + " saved)");
      }
      return {
        message: parts.join(", "),
        type: failed > 0 && ok === 0 ? "error" : "success",
        fullDetails: r,
      };
    },
  });
})();
