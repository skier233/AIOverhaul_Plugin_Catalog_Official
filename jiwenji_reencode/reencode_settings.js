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
      { key: 'max_concurrent_encodes', label: 'Max Concurrent Encodes', type: 'number', default: -1, desc: 'Set to -1 to auto detect number of available encoder engines', info: 'gpu_engines' },
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
      { key: 'skip_codecs', label: 'Skip Codecs', type: 'codec_chips', default: ['hevc', 'av1', 'vp9'], desc: 'Skip files already encoded in these codecs',
        chips: [
          { value: 'hevc', label: 'H.265' },
          { value: 'av1',  label: 'AV1' },
          { value: 'vp9',  label: 'VP9' },
          { value: 'vp8',  label: 'VP8' }
        ]
      },
      { key: 'skip_failed_tag', label: 'Skip Previously Failed', type: 'boolean', default: true, desc: 'Skip scenes tagged with the failure tag. Disable to retry failed files (e.g. after changing CQ settings)' },
      { key: 'remux_incompatible_container', label: 'Remux Incompatible Containers', type: 'boolean', default: true, desc: 'Remux files in containers that cannot hold HEVC (WMV, AVI, FLV, 3GP, MPEG) to MP4. When disabled, these files are skipped.' },
      { key: 'output_suffix', label: 'Output Filename Suffix', type: 'string', default: '', desc: 'e.g. "_hevc". Empty = replace in-place' },
      { key: 'copy_metadata_on_suffix', label: 'Copy Metadata on Suffix', type: 'boolean', default: true, desc: 'Copy tags/performers/etc. to new scene when using suffix' },
      { key: 'min_savings_pct', label: 'Minimum Savings %', type: 'number', default: 15, desc: 'Reject encodes below this savings threshold' },
      { key: 'gpu_index', label: 'GPU Index', type: 'number', default: 0, desc: 'GPU device index for multi-GPU systems' },
      { key: 'enable_retries', label: 'Enable Aggressive Retries', type: 'boolean', default: true, desc: 'Retry with more aggressive quality settings if initial encode fails to meet savings threshold' },
      { key: 'aggressive_cq', label: 'Aggressive Retry CQ', type: 'number', default: 34, desc: 'CQ for first retry attempt' },
      { key: 'ultra_aggressive_cq', label: 'Ultra-Aggressive CQ Ceiling', type: 'number', default: 40, desc: 'Max CQ for ultra-aggressive retry chain' },
      { key: 'tag_in_parallel', label: 'Run AI Tagging in Parallel - Requires Skier AI Tagging Plugin', type: 'boolean', default: true, desc: 'OFF: runs an AI tagging job at the end, after all re-encoding has finished. ON (default): run encoding and tagging simultaneously (uses more resources)' },
      { key: 'rescan_after_tagging', label: 'Delay Rescan', type: 'boolean', default: true, desc: 'Delay Stash rescan until AI tagging completes. Required for generating marker preview images from tagger-created markers. When disabled, rescan happens immediately after encode.' }
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

      function arraysEqual(a, b) {
        if (!Array.isArray(a) || !Array.isArray(b)) return false;
        var sa = a.slice().sort(), sb = b.slice().sort();
        if (sa.length !== sb.length) return false;
        for (var i = 0; i < sa.length; i++) { if (sa[i] !== sb[i]) return false; }
        return true;
      }

      function saveSetting(key, val) {
        setValues(function(prev) {
          var next = Object.assign({}, prev);
          var def = DEFAULTS[key];
          // If value equals default, remove it from blob to keep it clean
          var isDefault = Array.isArray(def) ? arraysEqual(val, def) : val === def;
          if (isDefault) {
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

      var infoPopupState = React.useState(null);
      var infoPopup = infoPopupState[0];
      var setInfoPopup = infoPopupState[1];

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
                } else if (field.type === 'codec_chips') {
                  var chipArr = Array.isArray(val) ? val : (field.default || []);
                  control = React.createElement('div', { style: { display: 'flex', gap: 6, flexWrap: 'wrap' } },
                    (field.chips || []).map(function(chip) {
                      var active = chipArr.indexOf(chip.value) !== -1;
                      return React.createElement('span', {
                        key: chip.value,
                        onClick: function() {
                          var updated = chipArr.slice();
                          var idx = updated.indexOf(chip.value);
                          if (idx !== -1) updated.splice(idx, 1);
                          else updated.push(chip.value);
                          saveSetting(field.key, updated);
                        },
                        style: {
                          display: 'inline-block', padding: '4px 12px', borderRadius: 12,
                          fontSize: 12, fontWeight: 'bold', cursor: 'pointer', userSelect: 'none',
                          background: active ? '#1a6b3a' : '#2a2a2a',
                          color: active ? '#7cfc7c' : '#777',
                          border: '1px solid ' + (active ? '#2d8a4e' : '#444'),
                          transition: 'background 0.15s, color 0.15s'
                        }
                      }, chip.label);
                    })
                  );
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

                var infoBtn = null;
                if (field.info === 'gpu_engines') {
                  infoBtn = React.createElement('span', {
                    style: {
                      display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                      width: 18, height: 18, borderRadius: '50%', fontSize: 11, fontWeight: 'bold',
                      background: '#333', color: '#aaa', border: '1px solid #555', cursor: 'pointer',
                      marginLeft: 6, userSelect: 'none', flexShrink: 0
                    },
                    onClick: function() { setInfoPopup(infoPopup === field.key ? null : field.key); },
                    title: 'Show GPU encoder engine reference'
                  }, '?');
                }

                // GPU engines modal (rendered via portal at document.body level)
                var infoModal = null;
                if (field.info === 'gpu_engines' && infoPopup === field.key) {
                  var trs = [
                    ['RTX 5090', '3'], ['RTX 5080 / 5070 Ti', '2'], ['RTX 5070 / 5060 Ti / 5060', '1'],
                    ['RTX 4090 / 4080 / 4070 Ti / 4070 / 4060 Ti / 4060', '2'],
                    ['RTX 3090 / 3080 / 3070 / 3060 (all 30-series)', '1'],
                    ['RTX 2080 Ti / 2080 / 2070 / 2060 (all 20-series)', '1'],
                    ['GTX 1660 / 1650 / 1080 / 1070 / 1060 / 1050', '1'],
                    ['RTX PRO 6000 / L40S / L40', '3'],
                    ['RTX 6000 Ada / RTX A4000', '2\u20133'],
                    ['RTX A6000 / A5000 / Quadro RTX', '1'],
                  ];
                  var tableStyle = { fontSize: 12, borderCollapse: 'collapse', width: '100%' };
                  var thStyle = { textAlign: 'left', padding: '6px 12px', borderBottom: '1px solid #444', color: '#999' };
                  var tdStyle = { padding: '5px 12px', borderBottom: '1px solid #2a2a2a', color: '#bbb' };
                  var tdRight = Object.assign({}, tdStyle, { textAlign: 'center', fontWeight: 'bold', color: '#7cfc7c' });
                  var modalBackdrop = {
                    position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
                    background: 'rgba(0,0,0,0.7)', zIndex: 10000,
                    display: 'flex', alignItems: 'center', justifyContent: 'center'
                  };
                  var modalBox = {
                    background: '#1a1a1a', border: '1px solid #444', borderRadius: 8,
                    padding: '20px 24px', maxWidth: 520, width: '90%', maxHeight: '80vh',
                    overflowY: 'auto', position: 'relative'
                  };
                  var closeBtnStyle = {
                    position: 'absolute', top: 10, right: 14, fontSize: 18, color: '#888',
                    cursor: 'pointer', background: 'none', border: 'none', lineHeight: 1
                  };
                  infoModal = React.createElement('div', {
                    style: modalBackdrop,
                    onClick: function(e) { if (e.target === e.currentTarget) setInfoPopup(null); }
                  },
                    React.createElement('div', { style: modalBox },
                      React.createElement('button', { style: closeBtnStyle, onClick: function() { setInfoPopup(null); } }, '\u00D7'),
                      React.createElement('div', { style: { fontSize: 15, fontWeight: 'bold', color: '#eee', marginBottom: 12 } }, 'GPU Encoder Engines Reference'),
                      React.createElement('div', { style: { fontSize: 12, color: '#999', marginBottom: 12, lineHeight: '1.5' } },
                        'Each encoder engine can run one simultaneous encode. Set this value to match your GPU, or use -1 to auto-detect (NVIDIA only).'
                      ),
                      React.createElement('div', { style: { fontSize: 13, fontWeight: 'bold', color: '#7cfc7c', marginBottom: 6 } }, 'NVIDIA'),
                      React.createElement('table', { style: tableStyle },
                        React.createElement('thead', null,
                          React.createElement('tr', null,
                            React.createElement('th', { style: thStyle }, 'GPU Model'),
                            React.createElement('th', { style: Object.assign({}, thStyle, { textAlign: 'center' }) }, 'Engines')
                          )
                        ),
                        React.createElement('tbody', null,
                          trs.map(function(r, i) {
                            return React.createElement('tr', { key: i },
                              React.createElement('td', { style: tdStyle }, r[0]),
                              React.createElement('td', { style: tdRight }, r[1])
                            );
                          })
                        )
                      ),
                      React.createElement('div', { style: { fontSize: 13, fontWeight: 'bold', color: '#ff6b6b', marginTop: 16, marginBottom: 6 } }, 'AMD (Manual Setup Required)'),
                      React.createElement('div', { style: { fontSize: 12, color: '#bbb', lineHeight: '1.6' } },
                        'AMD GPUs are not auto-detected. If you have an AMD card, set this value manually:',
                        React.createElement('ul', { style: { margin: '6px 0', paddingLeft: 20 } },
                          React.createElement('li', null, React.createElement('strong', null, 'RX 7900 XTX / 7900 XT'), ' \u2014 2 engines'),
                          React.createElement('li', null, React.createElement('strong', null, 'All other RDNA 3'), ' (7800 XT, 7700 XT, 7600) \u2014 1 engine'),
                          React.createElement('li', null, React.createElement('strong', null, 'RDNA 2 and older'), ' \u2014 1 engine')
                        ),
                        'The worker will still function with AMD GPUs, but you must set the concurrent encode count yourself.'
                      )
                    )
                  );
                }

                return React.createElement(React.Fragment, { key: field.key },
                  React.createElement('div', { style: rowStyle, title: field.desc },
                    React.createElement('span', { style: labelStyle },
                      field.label,
                      isChanged
                        ? React.createElement('span', { style: { color: '#ffa657', fontSize: 10, marginLeft: 4 } }, '\u2022')
                        : React.createElement('span', { style: defaultHintStyle }, '(default: ' + (Array.isArray(field.default) ? field.default.map(function(v) { var c = (field.chips || []).find(function(ch) { return ch.value === v; }); return c ? c.label : v; }).join(', ') : String(field.default)) + ')')
                    ),
                    control,
                    infoBtn,
                    React.createElement('button', {
                      style: Object.assign({}, resetBtnStyle, isChanged ? {} : { opacity: 0.3, cursor: 'default' }),
                      onClick: isChanged ? function() { resetSetting(field.key); } : undefined,
                      disabled: !isChanged,
                      title: 'Reset to default (' + (Array.isArray(field.default) ? field.default.join(', ') : field.default) + ')'
                    }, 'Reset')
                  ),
                  infoModal
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
