// Re-encode Advanced Settings — collapsible panel with all encoder options
// Loaded by PluginSettings CustomFieldLoader for type "reencode_settings"
// All advanced settings are stored as a single JSON object under the "reencode_advanced" key.
(function() {
  'use strict';
  var w = window;

  function waitForReact(cb, max) {
    max = max || 50;
    var n = 0;
    (function check() {
      var P = w.PluginApi;
      if (P && P.React) cb(P.React);
      else if (++n < max) setTimeout(check, 100);
    })();
  }

  waitForReact(function(React) {

    // ── Advanced field definitions with hardcoded defaults ──
    var FIELDS = [
      { key: 'worker_url', label: 'Worker URL', type: 'string', default: 'http://localhost:4154', desc: 'URL of the reencode_worker sidecar container' },
      { key: 'max_concurrent_encodes', label: 'Max Concurrent Encodes', type: 'number', default: -1, desc: 'Set to -1 to auto detect number of available encoder engines' },
      { key: 'cq', label: 'Quality Level (CQ)', type: 'number', default: 28, desc: 'NVENC constant-quality level (0\u201351, lower = better)' },
      { key: 'cq_low_bitrate', label: 'Low-Bitrate CQ', type: 'number', default: 34, desc: 'CQ for already-compact files' },
      { key: 'preset', label: 'NVENC Preset', type: 'select', default: 'p7', desc: 'p1 = fastest, p7 = best compression',
        options: [
          { value: 'p1', label: 'p1 (fastest)' },
          { value: 'p2', label: 'p2' },
          { value: 'p3', label: 'p3' },
          { value: 'p4', label: 'p4 (balanced)' },
          { value: 'p5', label: 'p5' },
          { value: 'p6', label: 'p6' },
          { value: 'p7', label: 'p7 (best compression)' }
        ]
      },
      { key: 'skip_hevc', label: 'Skip HEVC Files', type: 'boolean', default: true, desc: 'Skip files already in H.265/HEVC' },
      { key: 'skip_failed_tag', label: 'Skip Previously Failed', type: 'boolean', default: true, desc: 'Skip scenes tagged with the failure tag. Disable to retry failed files (e.g. after changing CQ settings)' },
      { key: 'skip_incompatible_container', label: 'Skip Incompatible Containers', type: 'boolean', default: false, desc: 'Skip files in containers that cannot hold HEVC (WMV, AVI, FLV, 3GP, MPEG) instead of remuxing to MP4' },
      { key: 'output_suffix', label: 'Output Filename Suffix', type: 'string', default: '', desc: 'e.g. "_hevc". Empty = replace in-place' },
      { key: 'min_savings_pct', label: 'Minimum Savings %', type: 'number', default: 15, desc: 'Reject encodes below this savings threshold' },
      { key: 'gpu_index', label: 'GPU Index', type: 'number', default: 0, desc: 'GPU device index for multi-GPU systems' },
      { key: 'enable_retries', label: 'Enable Retries', type: 'boolean', default: true, desc: 'Retry with more aggressive settings on failure' },
      { key: 'aggressive_cq', label: 'Aggressive Retry CQ', type: 'number', default: 34, desc: 'CQ for first retry attempt' },
      { key: 'ultra_aggressive_cq', label: 'Ultra-Aggressive CQ Ceiling', type: 'number', default: 40, desc: 'Max CQ for ultra-aggressive retry chain' },
      { key: 'copy_metadata_on_suffix', label: 'Copy Metadata on Suffix', type: 'boolean', default: true, desc: 'Copy tags/performers/etc. to new scene when using suffix' },
      { key: 'tag_after_reencode', label: 'Tag After Re-encode', type: 'boolean', default: false, desc: 'Chain into AI tagging after successful encode' },
      { key: 'tag_in_parallel', label: 'Run AI Tagging in Parallel', type: 'boolean', default: false, desc: 'OFF (default): wait for any running AI tagging jobs before encoding, and defer chained tagging until all encodes finish. ON: run encoding and tagging simultaneously (requires more resources, may cause instability)' }
    ];

    var PLUGIN_NAME = 'jiwenji_reencode';
    var SETTING_KEY = 'reencode_advanced';

    // Build a map of defaults for quick lookup
    var DEFAULTS = {};
    FIELDS.forEach(function(f) { DEFAULTS[f.key] = f.default; });

    function getApiHeaders() {
      var headers = { 'Content-Type': 'application/json' };
      try {
        var helper = w.AISharedApiKeyHelper;
        if (helper && typeof helper.get === 'function') {
          var key = helper.get();
          if (key) headers['x-ai-api-key'] = key;
        }
      } catch(e) {}
      return headers;
    }

    function createReencodeSettingsComponent(props) {
      var savePluginSetting = props.savePluginSetting;
      var setError = props.setError;
      var backendBase = props.backendBase || '';

      var expandedState = React.useState(false);
      var expanded = expandedState[0];
      var setExpanded = expandedState[1];

      var valuesState = React.useState({});
      var values = valuesState[0];
      var setValues = valuesState[1];

      var loadedState = React.useState(false);
      var loaded = loadedState[0];
      var setLoaded = loadedState[1];

      // Load current advanced values blob when expanded
      React.useEffect(function() {
        if (!expanded || loaded) return;
        var url = backendBase + '/api/v1/plugins/settings/' + PLUGIN_NAME;
        fetch(url, { headers: getApiHeaders() })
          .then(function(r) { return r.json(); })
          .then(function(settings) {
            if (Array.isArray(settings)) {
              // Find the reencode_advanced blob
              var advRow = null;
              for (var i = 0; i < settings.length; i++) {
                if (settings[i].key === SETTING_KEY) {
                  advRow = settings[i];
                  break;
                }
              }
              var blob = (advRow && advRow.value && typeof advRow.value === 'object') ? advRow.value : {};
              setValues(blob);
            }
            setLoaded(true);
          })
          .catch(function() { setLoaded(true); });
      }, [expanded, loaded]);

      // Persist the entire advanced blob directly via API — avoids calling
      // savePluginSetting which triggers parent re-render and remounts this component
      function persistBlob(newValues) {
        var url = backendBase + '/api/v1/plugins/settings/' + PLUGIN_NAME + '/' + encodeURIComponent(SETTING_KEY);
        fetch(url, {
          method: 'PUT',
          headers: getApiHeaders(),
          body: JSON.stringify({ value: newValues })
        }).catch(function(e) {
          if (setError) setError('Failed to save advanced settings: ' + (e.message || e));
        });
      }

      function saveSetting(key, val) {
        setValues(function(prev) {
          var next = Object.assign({}, prev);
          // If value equals default, remove it from blob to keep it clean
          if (val === DEFAULTS[key]) {
            delete next[key];
          } else {
            next[key] = val;
          }
          persistBlob(next);
          return next;
        });
      }

      function resetSetting(key) {
        setValues(function(prev) {
          var next = Object.assign({}, prev);
          delete next[key];
          persistBlob(next);
          return next;
        });
      }

      // Styles
      var wrap = { position: 'relative', padding: '4px 4px 6px', border: '1px solid #2a2a2a', borderRadius: 4, background: '#101010' };
      var headerStyle = {
        display: 'flex', alignItems: 'center', cursor: 'pointer', userSelect: 'none',
        padding: '8px 12px', background: '#1a1a1a', borderRadius: 4
      };
      var arrowStyle = { marginRight: 8, fontSize: 12, fontWeight: 'bold', color: '#777' };
      var titleStyle = { flex: 1, fontSize: 13, fontWeight: 'bold', color: '#999' };
      var inputStyle = { padding: '4px 6px', background: '#222', color: '#eee', border: '1px solid #333', borderRadius: 3, fontSize: 12, minWidth: 100 };
      var rowStyle = { display: 'flex', alignItems: 'center', padding: '6px 12px', borderBottom: '1px solid #1a1a1a', gap: 12 };
      var labelStyle = { width: 200, minWidth: 200, fontSize: 12, color: '#bbb' };
      var resetBtnStyle = { fontSize: 9, padding: '1px 4px', cursor: 'pointer', background: '#333', color: '#aaa', border: '1px solid #555', borderRadius: 2, marginLeft: 8 };
      var defaultHintStyle = { fontSize: 10, color: '#666', marginLeft: 6 };

      return React.createElement('div', { style: wrap },
        React.createElement('div', { style: headerStyle, onClick: function() { setExpanded(!expanded); } },
          React.createElement('span', { style: arrowStyle }, expanded ? '\u25BC' : '\u25B6'),
          React.createElement('span', { style: titleStyle }, 'Advanced Settings'),
          React.createElement('span', { style: { fontSize: 11, color: '#555' } }, expanded ? 'click to collapse' : 'click to expand')
        ),
        expanded && React.createElement('div', { style: { marginTop: 8, border: '1px solid #2a2a2a', borderRadius: 4, background: '#111' } },
          !loaded
            ? React.createElement('div', { style: { padding: 20, textAlign: 'center', fontSize: 12, color: '#666' } }, 'Loading...')
            : FIELDS.map(function(field) {
                // Value comes from the blob; fall back to hardcoded default
                var blobVal = values[field.key];
                var val = (blobVal !== undefined && blobVal !== null) ? blobVal : field.default;
                // A field is "changed" if it exists in the blob (i.e. user explicitly set it)
                var isChanged = blobVal !== undefined && blobVal !== null;

                var control;
                if (field.type === 'boolean') {
                  control = React.createElement('input', {
                    type: 'checkbox',
                    checked: !!val,
                    onChange: function(e) { saveSetting(field.key, e.target.checked); },
                    style: { cursor: 'pointer' }
                  });
                } else if (field.type === 'number') {
                  control = React.createElement('input', {
                    type: 'number',
                    value: val === undefined || val === null ? '' : val,
                    onChange: function(e) {
                      var v = e.target.value;
                      saveSetting(field.key, v === '' ? field.default : parseFloat(v));
                    },
                    style: Object.assign({}, inputStyle, { width: 80 })
                  });
                } else if (field.type === 'select') {
                  control = React.createElement('select', {
                    value: val || field.default,
                    onChange: function(e) { saveSetting(field.key, e.target.value); },
                    style: Object.assign({}, inputStyle, { width: 180 })
                  }, (field.options || []).map(function(opt) {
                    return React.createElement('option', { key: opt.value, value: opt.value }, opt.label);
                  }));
                } else {
                  control = React.createElement('input', {
                    type: 'text',
                    value: val === undefined || val === null ? '' : String(val),
                    onChange: function(e) { saveSetting(field.key, e.target.value); },
                    style: Object.assign({}, inputStyle, { width: 240 })
                  });
                }

                return React.createElement('div', { key: field.key, style: rowStyle, title: field.desc },
                  React.createElement('span', { style: labelStyle },
                    field.label,
                    isChanged
                      ? React.createElement('span', { style: { color: '#ffa657', fontSize: 10, marginLeft: 4 } }, '\u2022')
                      : React.createElement('span', { style: defaultHintStyle }, '(default: ' + String(field.default) + ')')
                  ),
                  control,
                  React.createElement('button', {
                    style: Object.assign({}, resetBtnStyle, isChanged ? {} : { opacity: 0.3, cursor: 'default' }),
                    onClick: isChanged ? function() { resetSetting(field.key); } : undefined,
                    disabled: !isChanged,
                    title: 'Reset to default (' + field.default + ')'
                  }, 'Reset')
                );
              })
        )
      );
    }

    // Register globally
    w.reencode_settings_Renderer = createReencodeSettingsComponent;
    w.jiwenji_reencode_reencode_settings_Renderer = createReencodeSettingsComponent;
  });
})();
