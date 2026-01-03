// Tag List Editor Frontend Component for Skier AI Tagging Plugin
// Patches PluginSettings.FieldRenderer to handle tag_list_editor type fields
(function() {
  'use strict';
  
  const w = window;
  
  // Wait for PluginApi and React to be available
  function waitForReact(callback, maxAttempts) {
    maxAttempts = maxAttempts || 50; // Try for up to 5 seconds
    let attempts = 0;
    
    function check() {
      const PluginApi = w.PluginApi;
      if (PluginApi && PluginApi.React) {
        callback(PluginApi.React);
        } else {
          attempts++;
          if (attempts < maxAttempts) {
            setTimeout(check, 100);
          }
        }
    }
    
    check();
  }
  
  // Initialize when React is available
  waitForReact(function(React) {
    const PluginApi = w.PluginApi;
  
  // Helper to get backend base URL from settings, with fallback to default
  function getBackendBase() {
    // Try AIDefaultBackendBase function first (reads from plugin settings)
    if (typeof w.AIDefaultBackendBase === 'function') {
      const base = w.AIDefaultBackendBase();
      if (base) return base;
    }
    // Fall back to window.AI_BACKEND_URL (set by BackendBase.ts)
    if (typeof w.AI_BACKEND_URL === 'string' && w.AI_BACKEND_URL) {
      return w.AI_BACKEND_URL;
    }
    // Fall back to legacy AIBackendBase if set
    if (w.AIBackendBase) {
      return w.AIBackendBase;
    }
    // Default fallback
    return 'http://localhost:4153';
  }

  // Helper to make API calls
  function jfetch(url, options) {
    const backendBase = getBackendBase();
    const fullUrl = url.startsWith('http') ? url : backendBase + url;
    return fetch(fullUrl, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options && options.headers || {})
      },
      body: options && options.body ? JSON.stringify(options.body) : undefined
    }).then(response => {
      if (!response.ok) {
        return response.json().then(err => {
          throw new Error(err.detail || err.message || 'HTTP ' + response.status);
        }).catch(() => {
          throw new Error('HTTP ' + response.status);
        });
      }
      return response.json();
    });
  }
  
  // Create the TagListEditor component
  function createTagListEditorComponent(props) {
    const field = props.field;
    const pluginName = props.pluginName;
    // Use props.backendBase if provided, otherwise get from settings
    const backendBaseState = React.useState(function() {
      return props.backendBase || getBackendBase();
    });
    const backendBase = backendBaseState[0];
    const setBackendBase = backendBaseState[1];
    const savePluginSetting = props.savePluginSetting;
    const loadPluginSettings = props.loadPluginSettings;
    const setError = props.setError;
    
    // Listen for backend base URL updates from settings
    React.useEffect(function() {
      function handleBackendUpdate(e) {
        const newBase = e.detail || getBackendBase();
        setBackendBase(newBase);
      }
      try {
        w.addEventListener('AIBackendBaseUpdated', handleBackendUpdate);
        return function() {
          try { w.removeEventListener('AIBackendBaseUpdated', handleBackendUpdate); } catch {}
        };
      } catch {}
    }, []);
    
    const modalOpenState = React.useState(false);
    const modalOpen = modalOpenState[0];
    const setModalOpen = modalOpenState[1];
    
    // Full tag settings state - map of normalized tag name to settings object
    const tagSettingsState = React.useState({});
    const tagSettings = tagSettingsState[0];
    const setTagSettings = tagSettingsState[1];
    
    // Default values from __default__ row
    const defaultsState = React.useState({});
    const defaults = defaultsState[0];
    const setDefaults = defaultsState[1];
    
    // Expanded/collapsed sections (start collapsed by default)
    const expandedSectionsState = React.useState(new Set());
    const expandedSections = expandedSectionsState[0];
    const setExpandedSections = expandedSectionsState[1];
    
    const loadingState = React.useState(false);
    const loading = loadingState[0];
    const setLoading = loadingState[1];
    
    const savingState = React.useState(false);
    const saving = savingState[0];
    const setSaving = savingState[1];

    const loadTagData = React.useCallback(async function() {
      setLoading(true);
      try {
        // Use /available endpoint which now returns full tag settings
        const availableUrl = `/api/v1/plugins/settings/${pluginName}/tags/available`;
        
        const availableResponse = await jfetch(availableUrl);
        
        // Extract tags and defaults
        const tags = availableResponse.tags || [];
        const defaultsData = availableResponse.defaults || {};
        
        // Build tag settings map from tags
        const settingsMap = {};
        tags.forEach(function(tagInfo) {
          const tagName = tagInfo.tag || tagInfo.name || '';
          const normalized = tagName.toLowerCase();
          settingsMap[normalized] = {
            tagName: tagName,
            category: tagInfo.category || 'Other',
            enabled: tagInfo.enabled !== false, // Default to true
            markers_enabled: tagInfo.markers_enabled !== false, // Default to true
            required_scene_tag_duration: tagInfo.required_scene_tag_duration || '',
            min_marker_duration: tagInfo.min_marker_duration || '',
            max_gap: tagInfo.max_gap || '',
          };
        });
        
        setTagSettings(settingsMap);
        setDefaults(defaultsData);
      } catch (e) {
        if (setError) setError(e.message || 'Failed to load tag data');
        setTagSettings({});
        setDefaults({});
      } finally {
        setLoading(false);
      }
    }, [pluginName, setError]);

    const saveTagSettings = React.useCallback(async function() {
      console.log('[SkierAITagging] Tag settings to save:', tagSettings);
      
      setSaving(true);
      try {
        // Build tag_settings dict for API
        const settingsToSave = {};
        Object.keys(tagSettings).forEach(function(normalized) {
          const settings = tagSettings[normalized];
          settingsToSave[normalized] = {
            enabled: settings.enabled,
            markers_enabled: settings.markers_enabled,
            required_scene_tag_duration: settings.required_scene_tag_duration || null,
            min_marker_duration: settings.min_marker_duration ? parseFloat(settings.min_marker_duration) : null,
            max_gap: settings.max_gap ? parseFloat(settings.max_gap) : null,
          };
        });
        
        const saveUrl = `/api/v1/plugins/settings/${pluginName}/tags/settings`;
        
        const result = await jfetch(saveUrl, {
          method: 'PUT',
          body: {
            tag_settings: settingsToSave
          }
        });
        
        setModalOpen(false);
        if (loadPluginSettings) {
          await loadPluginSettings(pluginName);
        }
      } catch (e) {
        if (setError) setError(e.message || 'Failed to save tag settings');
      } finally {
        setSaving(false);
      }
    }, [pluginName, tagSettings, loadPluginSettings, setError]);

    const wrap = { position: 'relative', padding: '4px 4px 6px', border: '1px solid #2a2a2a', borderRadius: 4, background: '#101010' };
    const smallBtn = { fontSize: 11, padding: '4px 8px', background: '#2a2a2a', color: '#eee', border: '1px solid #444', borderRadius: 3, cursor: 'pointer' };
    const labelTitle = field && field.description ? String(field.description) : undefined;
    const labelEl = React.createElement('span', { title: labelTitle }, field.label || field.key);

    // Helper functions
    function updateTagSetting(normalized, field, value) {
      setTagSettings(function(prev) {
        const updated = { ...prev };
        if (!updated[normalized]) {
          updated[normalized] = {};
        }
        updated[normalized] = { ...updated[normalized], [field]: value };
        return updated;
      });
    }

    function toggleSection(category) {
      setExpandedSections(function(prev) {
        const newSet = new Set(prev);
        if (newSet.has(category)) {
          newSet.delete(category);
        } else {
          newSet.add(category);
        }
        return newSet;
      });
    }

    function toggleCategoryAll(category, enabled) {
      setTagSettings(function(prev) {
        const updated = { ...prev };
        Object.keys(updated).forEach(function(normalized) {
          if (updated[normalized].category === category) {
            updated[normalized] = { ...updated[normalized], enabled: enabled };
          }
        });
        return updated;
      });
    }

    function isCategoryAllEnabled(category) {
      const categoryTags = Object.values(tagSettings).filter(function(s) { return s.category === category; });
      if (categoryTags.length === 0) return false;
      return categoryTags.every(function(s) { return s.enabled; });
    }

    function getCategoryEnabledCount(category) {
      return Object.values(tagSettings).filter(function(s) { return s.category === category && s.enabled; }).length;
    }

    function getCategoryTotalCount(category) {
      return Object.values(tagSettings).filter(function(s) { return s.category === category; }).length;
    }
    
    // Group tags by category
    const tagsByCategory = {};
    Object.keys(tagSettings).forEach(function(normalized) {
      const settings = tagSettings[normalized];
      const category = settings.category || 'Other';
      if (!tagsByCategory[category]) {
        tagsByCategory[category] = [];
      }
      tagsByCategory[category].push({ normalized: normalized, settings: settings });
    });

    // Sort tags within each category
    Object.keys(tagsByCategory).forEach(function(category) {
      tagsByCategory[category].sort(function(a, b) {
        return a.settings.tagName.localeCompare(b.settings.tagName);
      });
    });

    // Category order
    const categoryOrder = ['Sexual Actions', 'Body Parts', 'BDSM', 'Positions', 'Other'];
    const sortedCategories = categoryOrder.filter(function(cat) { return tagsByCategory[cat] && tagsByCategory[cat].length > 0; })
      .concat(Object.keys(tagsByCategory).filter(function(cat) { return categoryOrder.indexOf(cat) < 0; }));

    return React.createElement(React.Fragment, null,
      React.createElement('div', { style: wrap },
        React.createElement('div', { style: { fontSize: 12, marginBottom: 6 } }, labelEl),
        React.createElement('button', {
          style: smallBtn,
          onClick: function() {
            setModalOpen(true);
            loadTagData();
          }
        }, 'Open Tagging Configuration Editor')
      ),
      modalOpen && React.createElement('div', {
        style: {
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 10000
        },
        onClick: function() {
          if (!saving) setModalOpen(false);
        }
      },
        React.createElement('div', {
          style: {
            background: '#1e1e1e',
            border: '1px solid #444',
            borderRadius: 8,
            padding: 20,
            maxWidth: '90vw',
            maxHeight: '90vh',
            width: 800,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden'
          },
          onClick: function(e) { e.stopPropagation(); }
        },
          React.createElement('div', {
            style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }
          },
            React.createElement('h3', { style: { margin: 0, fontSize: 18 } }, 'Edit Tagging Configuration'),
            React.createElement('button', {
              style: smallBtn,
              onClick: function() { setModalOpen(false); },
              disabled: saving
            }, '×')
          ),
          React.createElement('div', {
            style: { fontSize: 11, color: '#aaa', marginBottom: 16, lineHeight: 1.4, padding: '0 4px' }
          }, 'Configure tag settings. Unchecked tags will be excluded from tag generation. Changes are saved to tag_settings.csv.'),
          loading ? React.createElement('div', {
            style: { padding: 40, textAlign: 'center', fontSize: 12, opacity: 0.7 }
          }, 'Loading tags from CSV...') :
          Object.keys(tagSettings).length === 0 ? React.createElement('div', {
            style: { padding: 40, textAlign: 'center', fontSize: 12, opacity: 0.7 }
          }, 'No tags available in tag_settings.csv.') :
          React.createElement(React.Fragment, null,
            React.createElement('div', {
              style: { flex: 1, overflow: 'auto', border: '1px solid #333', borderRadius: 4, padding: 8, background: '#111', marginBottom: 16, maxHeight: '60vh' }
            },
              sortedCategories.map(function(category) {
                const categoryTags = tagsByCategory[category] || [];
                const isExpanded = expandedSections.has(category);
                const allEnabled = isCategoryAllEnabled(category);
                const enabledCount = getCategoryEnabledCount(category);
                const totalCount = getCategoryTotalCount(category);
                
                return React.createElement('div', {
                  key: category,
                  style: { marginBottom: 8, border: '1px solid #2a2a2a', borderRadius: 4, background: '#151515' }
                },
                  // Section header
                  React.createElement('div', {
                    style: {
                      display: 'flex',
                      alignItems: 'center',
                      padding: '8px 12px',
                      background: '#1a1a1a',
                      borderBottom: isExpanded ? '1px solid #2a2a2a' : 'none',
                      cursor: 'pointer',
                      userSelect: 'none'
                    },
                    onClick: function() { toggleSection(category); }
                  },
                    React.createElement('span', {
                      style: { marginRight: 8, fontSize: 12, fontWeight: 'bold', color: '#ddd' }
                    }, isExpanded ? '▼' : '▶'),
                    React.createElement('span', {
                      style: { flex: 1, fontSize: 13, fontWeight: 'bold', color: '#eee' }
                    }, category),
                    React.createElement('span', {
                      style: { fontSize: 11, color: '#999', marginRight: 12 }
                    }, enabledCount + '/' + totalCount),
                    React.createElement('button', {
                      style: Object.assign({}, smallBtn, {
                        fontSize: 10,
                        padding: '2px 6px',
                        background: allEnabled ? '#2d5a3d' : '#2a2a2a'
                      }),
                      onClick: function(e) {
                        e.stopPropagation();
                        toggleCategoryAll(category, !allEnabled);
                      }
                    }, allEnabled ? 'Uncheck All' : 'Check All')
                  ),
                  // Section content
                  isExpanded && React.createElement('div', {
                    style: { padding: '4px 0' }
                  },
                    categoryTags.map(function(tagData) {
                      const normalized = tagData.normalized;
                      const settings = tagData.settings;
                      const tagName = settings.tagName;
                      const isEnabled = settings.enabled;
                      const isDisabled = !isEnabled;
                      
                      // Get default value for required_scene_tag_duration
                      const defaultReqDuration = defaults.required_scene_tag_duration || '15';
                      const showDefaultReqDuration = !settings.required_scene_tag_duration;
                      
                      return React.createElement('div', {
                        key: normalized,
                        style: {
                          padding: '6px 12px',
                          borderBottom: '1px solid #1a1a1a',
                          background: isDisabled ? '#0f0f0f' : 'transparent',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '12px',
                          flexWrap: 'wrap'
                        }
                      },
                        // Enabled checkbox
                        React.createElement('input', {
                          type: 'checkbox',
                          checked: isEnabled,
                          onChange: function(e) {
                            updateTagSetting(normalized, 'enabled', e.target.checked);
                          },
                          style: { cursor: 'pointer', flexShrink: 0 }
                        }),
                        // Tag name (bold)
                        React.createElement('span', {
                          style: {
                            fontSize: 12,
                            fontWeight: 'bold',
                            color: isDisabled ? '#666' : '#ddd',
                            minWidth: '120px',
                            flexShrink: 0
                          }
                        }, tagName),
                        // Inline controls
                        React.createElement('div', {
                          style: {
                            display: 'flex',
                            alignItems: 'center',
                            gap: '12px',
                            flex: 1,
                            opacity: isDisabled ? 0.5 : 1,
                            flexWrap: 'wrap'
                          }
                        },
                          // Markers enabled
                          React.createElement('label', {
                            style: { display: 'flex', alignItems: 'center', fontSize: 11 }
                          },
                            React.createElement('input', {
                              type: 'checkbox',
                              checked: settings.markers_enabled,
                              disabled: isDisabled,
                              onChange: function(e) {
                                updateTagSetting(normalized, 'markers_enabled', e.target.checked);
                              },
                              style: { marginRight: 6, cursor: isDisabled ? 'not-allowed' : 'pointer' }
                            }),
                            React.createElement('span', { style: { color: '#aaa' } }, 'Markers')
                          ),
                          // Required scene tag duration
                          React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 4 } },
                            React.createElement('span', {
                              style: { fontSize: 10, color: '#999', whiteSpace: 'nowrap' }
                            }, 'Req. Duration:'),
                            React.createElement('input', {
                              type: 'text',
                              value: settings.required_scene_tag_duration || '',
                              disabled: isDisabled,
                              placeholder: showDefaultReqDuration ? defaultReqDuration : '',
                              onChange: function(e) {
                                updateTagSetting(normalized, 'required_scene_tag_duration', e.target.value);
                              },
                              style: {
                                width: '60px',
                                padding: '2px 4px',
                                fontSize: 11,
                                background: isDisabled ? '#1a1a1a' : '#222',
                                border: '1px solid #333',
                                color: showDefaultReqDuration && !settings.required_scene_tag_duration ? '#666' : '#ddd',
                                fontStyle: showDefaultReqDuration && !settings.required_scene_tag_duration ? 'italic' : 'normal',
                                cursor: isDisabled ? 'not-allowed' : 'text'
                              }
                            })
                          ),
                          // Min marker duration
                          React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 4 } },
                            React.createElement('span', {
                              style: { fontSize: 10, color: '#999', whiteSpace: 'nowrap' }
                            }, 'Min Duration:'),
                            React.createElement('input', {
                              type: 'number',
                              value: settings.min_marker_duration || '',
                              disabled: isDisabled,
                              placeholder: defaults.min_marker_duration || '',
                              onChange: function(e) {
                                updateTagSetting(normalized, 'min_marker_duration', e.target.value);
                              },
                              style: {
                                width: '50px',
                                padding: '2px 4px',
                                fontSize: 11,
                                background: isDisabled ? '#1a1a1a' : '#222',
                                border: '1px solid #333',
                                color: '#ddd',
                                cursor: isDisabled ? 'not-allowed' : 'text'
                              }
                            })
                          ),
                          // Max gap
                          React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 4 } },
                            React.createElement('span', {
                              style: { fontSize: 10, color: '#999', whiteSpace: 'nowrap' }
                            }, 'Max Gap:'),
                            React.createElement('input', {
                              type: 'number',
                              value: settings.max_gap || '',
                              disabled: isDisabled,
                              placeholder: defaults.max_gap || '',
                              onChange: function(e) {
                                updateTagSetting(normalized, 'max_gap', e.target.value);
                              },
                              style: {
                                width: '50px',
                                padding: '2px 4px',
                                fontSize: 11,
                                background: isDisabled ? '#1a1a1a' : '#222',
                                border: '1px solid #333',
                                color: '#ddd',
                                cursor: isDisabled ? 'not-allowed' : 'text'
                              }
                            })
                          )
                        )
                      );
                    })
                  )
                );
              })
            ),
            React.createElement('div', {
              style: { display: 'flex', justifyContent: 'flex-end', gap: 8 }
            },
              React.createElement('button', {
                style: smallBtn,
                onClick: function() { setModalOpen(false); },
                disabled: saving
              }, 'Cancel'),
              React.createElement('button', {
                style: Object.assign({}, smallBtn, { background: saving ? '#444' : '#2d5a3d', borderColor: saving ? '#555' : '#4a7c59' }),
                onClick: saveTagSettings,
                disabled: saving
              }, saving ? 'Saving...' : 'Save')
            )
          )
        )
      )
    );
  }
  
    // Store globally so PluginSettings can access it
    // Register with standard naming conventions
    w.tag_list_editor_Renderer = createTagListEditorComponent; // Standard naming convention
    w.skier_aitagging_tag_list_editor_Renderer = createTagListEditorComponent; // Plugin-specific naming
    
    // Try to patch PluginSettings dynamically
    function patchPluginSettings() {
      // Wait for PluginSettings to be available
      if (!w.AIPluginSettings) {
        setTimeout(patchPluginSettings, 100);
        return;
      }
      
      // Try to use PluginApi.patch if available
      if (PluginApi.patch && PluginApi.patch.before) {
        try {
          // This won't work directly since FieldRenderer is internal, but we can try
          // The real solution is PluginSettings needs to check for our component
        } catch (e) {
        }
      }
      
      // Store a flag that PluginSettings can check
      w.SkierAITaggingTagListEditorReady = true;
    }
    
    // Try patching immediately
    patchPluginSettings();
    
    // Also listen for the ready event
    w.addEventListener('AIPluginSettingsReady', function() {
      patchPluginSettings();
    });
    
    // Also try after a delay
    setTimeout(function() {
      patchPluginSettings();
    }, 2000);
  }); // End of waitForReact callback
})();
