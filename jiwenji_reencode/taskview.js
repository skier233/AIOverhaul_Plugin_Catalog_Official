(function () {
  var register = window.AIRegisterTaskViewRenderer;
  if (!register) return;

  function formatSavings(mb) {
    if (mb >= 10000) return (mb / 1024).toFixed(1) + " GB";
    return mb.toFixed(1) + " MB";
  }

  function truncFilename(name, max) {
    if (!name) return "";
    if (name.length <= max) return name;
    var ext = "";
    var dot = name.lastIndexOf(".");
    if (dot > 0) {
      ext = name.substring(dot);
      name = name.substring(0, dot);
    }
    return name.substring(0, max - ext.length - 1) + "\u2026" + ext;
  }

  register({
    id: "reencode:active",
    priority: 10,
    match: function (task) {
      return (
        task.service === "jiwenji_reencode" &&
        (task.status === "running" || task.status === "queued")
      );
    },
    render: function (task, React, helpers) {
      var h = React.createElement;
      var p = task.progress || {};
      var total = p.total || 0;
      var completed = p.completed || 0;
      var failed = p.failed || 0;
      var skipped = p.skipped || 0;
      var success = p.success || 0;
      var running = p.running || 0;
      var savingsMb = p.savings_mb || 0;
      var workers = p.workers || [];
      var tagAfterReencode = !!p.tag_after_reencode;
      var tagging = p.tagging || null;
      var pct = total > 0 ? ((completed / total) * 100).toFixed(1) : "0.0";
      var taskIsCancelling = helpers.isCancelling(task.id);

      // Title — include "and tagging" when tagging is enabled
      var titleVerb = tagAfterReencode ? "Re-encoding and tagging" : "Re-encoding";
      var title =
        total > 1
          ? titleVerb + " " + total + " scenes"
          : total === 1
            ? titleVerb + " scene"
            : "Re-encode " + (task.status === "queued" ? "(queued)" : "");

      // Total progress bar (striped animated)
      var progressBar = h("div", { key: "pbar", className: "ai-tv__progress-track" }, [
        h("div", {
          key: "fill",
          className: "ai-tv__progress-fill ai-tv__progress-fill--striped",
          style: { width: pct + "%" },
        }),
        h(
          "span",
          { key: "lbl", className: "ai-tv__progress-label" },
          completed + " / " + total
        ),
      ]);

      // Counters row — encode phase
      var counters = [];
      if (running > 0)
        counters.push(
          h("span", { key: "run", className: "ai-tv__counter ai-tv__counter--running" }, running + " encoding")
        );
      if (success > 0)
        counters.push(
          h("span", { key: "ok", className: "ai-tv__counter ai-tv__counter--success" }, success + " encoded")
        );
      if (failed > 0)
        counters.push(
          h("span", { key: "fail", className: "ai-tv__counter ai-tv__counter--failed" }, failed + " failed")
        );
      if (skipped > 0)
        counters.push(
          h("span", { key: "skip", className: "ai-tv__counter ai-tv__counter--skipped" }, skipped + " skipped")
        );
      if (savingsMb > 0)
        counters.push(
          h("span", { key: "sav", className: "ai-tv__counter ai-tv__counter--savings" }, formatSavings(savingsMb) + " saved")
        );

      // Tagging counters
      if (tagging && tagging.total > 0) {
        if (tagging.running > 0)
          counters.push(
            h("span", { key: "tag-run", className: "ai-tv__counter ai-tv__counter--tagging" }, tagging.running + " tagging")
          );
        if (tagging.success > 0)
          counters.push(
            h("span", { key: "tag-ok", className: "ai-tv__counter ai-tv__counter--tagged" }, tagging.success + " tagged")
          );
        if (tagging.failed > 0)
          counters.push(
            h("span", { key: "tag-fail", className: "ai-tv__counter ai-tv__counter--tag-failed" }, tagging.failed + " tag failed")
          );
        if (tagging.queued > 0)
          counters.push(
            h("span", { key: "tag-q", className: "ai-tv__counter ai-tv__counter--tagging" }, tagging.queued + " tag queued")
          );
      }

      // Worker sections
      var workerEls = workers.map(function (w, i) {
        var wPct = (w.percent || 0).toFixed(1);
        var fname = truncFilename(w.filename || "", 50);
        var speedStr = w.speed || "";
        var fpsVal = w.fps || 0;

        var meta = [];
        if (speedStr) meta.push(speedStr);
        if (fpsVal > 0) meta.push(Math.round(fpsVal) + " fps");

        return h("div", { key: "w" + i, className: "ai-tv__worker" }, [
          h("div", { key: "hdr", className: "ai-tv__worker-header" }, [
            h("span", { key: "label", className: "ai-tv__worker-label" }, "Worker " + (i + 1)),
            meta.length > 0
              ? h("span", { key: "meta", className: "ai-tv__worker-meta" }, meta.join("  "))
              : null,
            h("span", { key: "pct", className: "ai-tv__worker-pct" }, wPct + "%"),
          ]),
          h("div", { key: "bar", className: "ai-tv__worker-track" }, [
            h("div", {
              key: "fill",
              className: "ai-tv__worker-fill ai-tv__progress-fill--striped",
              style: { width: wPct + "%" },
            }),
          ]),
          fname
            ? h("div", { key: "fn", className: "ai-tv__worker-filename" }, fname)
            : null,
        ]);
      });

      // Cancel button
      var cancelBtn =
        task.status === "queued" || task.status === "running"
          ? h(
              "button",
              {
                key: "cancel",
                className: "ai-tv__cancel-btn",
                disabled: taskIsCancelling,
                onClick: function () {
                  helpers.cancelTask(task.id);
                },
              },
              taskIsCancelling ? "Cancelling..." : "Cancel"
            )
          : null;

      // Time
      var timeEl = h(
        "div",
        { key: "time", className: "ai-tv__time" },
        "Started " + helpers.formatTs(task.started_at)
      );

      return h("div", { className: "ai-task-view" }, [
        h("div", { key: "header", className: "ai-tv__header" }, [
          h("div", { key: "title", className: "ai-tv__title" }, title),
          h("div", { key: "actions", className: "ai-tv__actions" }, [timeEl, cancelBtn]),
        ]),
        progressBar,
        counters.length > 0
          ? h("div", { key: "counters", className: "ai-tv__counters" }, counters)
          : null,
        workerEls.length > 0
          ? h("div", { key: "workers", className: "ai-tv__workers" }, workerEls)
          : null,
      ]);
    },
  });
})();
