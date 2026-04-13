/**
 * SafeWatch AI — Alert Popup & WebSocket Handler
 * Drop this file into your frontend and call: SafeWatchAlerts.init()
 *
 * WHAT THIS FIXES:
 *   1. Correctly handles WebSocket messages with type='alert'
 *   2. Shows a rich toast popup for every incoming alert
 *   3. Plays a subtle audio chime for critical/high alerts
 *   4. Auto-dismisses after 8s, can be manually dismissed
 *   5. Shows snapshot image if available
 *   6. Stacks multiple alerts — newest on top
 *
 * USAGE:
 *   <script src="safewatch_alerts.js"></script>
 *   <script>SafeWatchAlerts.init('ws://localhost:8000/ws/live')</script>
 *
 *   Or pass your existing WebSocket instance:
 *   SafeWatchAlerts.attachTo(existingWebSocket)
 */

(function (global) {
  'use strict';

  // ── Style injection ────────────────────────────────────────────────────────
  const CSS = `
    #sw-alert-container {
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 99999;
      display: flex;
      flex-direction: column;
      gap: 10px;
      pointer-events: none;
      max-width: 380px;
      width: 100%;
    }

    .sw-alert-toast {
      pointer-events: all;
      background: #0f0f13;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px;
      overflow: hidden;
      box-shadow: 0 8px 32px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.04);
      display: flex;
      flex-direction: column;
      animation: sw-slide-in 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
      font-family: 'Courier New', 'SF Mono', monospace;
    }

    .sw-alert-toast.sw-dismissing {
      animation: sw-slide-out 0.25s ease-in forwards;
    }

    @keyframes sw-slide-in {
      from { transform: translateX(110%); opacity: 0; }
      to   { transform: translateX(0);    opacity: 1; }
    }
    @keyframes sw-slide-out {
      from { transform: translateX(0);    opacity: 1; }
      to   { transform: translateX(110%); opacity: 0; }
    }

    .sw-alert-stripe {
      height: 4px;
      width: 100%;
    }

    .sw-alert-stripe.level-1 { background: linear-gradient(90deg, #ff1a1a, #ff6b6b); }
    .sw-alert-stripe.level-2 { background: linear-gradient(90deg, #ff8c00, #ffd166); }
    .sw-alert-stripe.level-3 { background: linear-gradient(90deg, #00bcd4, #69f0ae); }

    .sw-alert-body {
      display: flex;
      gap: 12px;
      padding: 12px 14px;
    }

    .sw-alert-thumb {
      width: 72px;
      height: 72px;
      border-radius: 6px;
      object-fit: cover;
      flex-shrink: 0;
      background: #1a1a24;
      border: 1px solid rgba(255,255,255,0.1);
    }

    .sw-alert-thumb-placeholder {
      width: 72px;
      height: 72px;
      border-radius: 6px;
      flex-shrink: 0;
      background: #1a1a24;
      border: 1px solid rgba(255,255,255,0.1);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 28px;
    }

    .sw-alert-info {
      flex: 1;
      min-width: 0;
    }

    .sw-alert-badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 1.5px;
      padding: 3px 8px;
      border-radius: 4px;
      margin-bottom: 6px;
    }

    .sw-alert-badge.level-1 { background: rgba(255,26,26,0.2);  color: #ff6b6b; border: 1px solid rgba(255,26,26,0.4); }
    .sw-alert-badge.level-2 { background: rgba(255,140,0,0.2);  color: #ffd166; border: 1px solid rgba(255,140,0,0.4); }
    .sw-alert-badge.level-3 { background: rgba(0,188,212,0.2);  color: #69f0ae; border: 1px solid rgba(0,188,212,0.4); }

    .sw-alert-pulse {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      animation: sw-pulse 1.2s ease-in-out infinite;
    }

    .level-1 .sw-alert-pulse { background: #ff4444; }
    .level-2 .sw-alert-pulse { background: #ffaa00; }
    .level-3 .sw-alert-pulse { background: #00e5ff; }

    @keyframes sw-pulse {
      0%, 100% { opacity: 1; transform: scale(1);   }
      50%       { opacity: 0.5; transform: scale(1.4); }
    }

    .sw-alert-action {
      font-size: 14px;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: 0.5px;
      margin: 0 0 3px;
      text-transform: uppercase;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .sw-alert-meta {
      font-size: 11px;
      color: rgba(255,255,255,0.5);
      line-height: 1.5;
    }

    .sw-alert-meta span {
      display: inline-block;
      margin-right: 10px;
    }

    .sw-alert-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 6px 14px 10px;
    }

    .sw-alert-timer {
      height: 2px;
      background: rgba(255,255,255,0.08);
      border-radius: 2px;
      flex: 1;
      margin-right: 12px;
      overflow: hidden;
    }

    .sw-alert-timer-bar {
      height: 100%;
      border-radius: 2px;
      transition: width 0.1s linear;
    }

    .level-1 .sw-alert-timer-bar { background: #ff4444; }
    .level-2 .sw-alert-timer-bar { background: #ffaa00; }
    .level-3 .sw-alert-timer-bar { background: #00e5ff; }

    .sw-alert-close {
      background: none;
      border: none;
      cursor: pointer;
      color: rgba(255,255,255,0.3);
      font-size: 16px;
      padding: 0 2px;
      line-height: 1;
      transition: color 0.15s;
    }

    .sw-alert-close:hover { color: rgba(255,255,255,0.8); }
  `;

  // ── Level config ──────────────────────────────────────────────────────────
  const LEVEL_CONFIG = {
    1: { label: 'CRITICAL', icon: '🚨', dismiss: 10000 },
    2: { label: 'HIGH',     icon: '⚠️',  dismiss: 8000  },
    3: { label: 'MEDIUM',   icon: '🔵',  dismiss: 6000  },
  };

  // ── Audio chime (Web Audio API — no file needed) ───────────────────────────
  function _chime(level) {
    try {
      const ctx  = new (window.AudioContext || window.webkitAudioContext)();
      const osc  = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type      = 'sine';
      osc.frequency.value = level === 1 ? 880 : level === 2 ? 660 : 440;
      gain.gain.setValueAtTime(0.12, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.6);
      osc.start();
      osc.stop(ctx.currentTime + 0.6);
    } catch (_) { /* audio blocked, silent */ }
  }

  // ── Toast factory ─────────────────────────────────────────────────────────
  let _container = null;

  function _ensureContainer() {
    if (_container) return;
    _container = document.createElement('div');
    _container.id = 'sw-alert-container';
    document.body.appendChild(_container);
  }

  function _showToast(alertData) {
    _ensureContainer();

    const level   = alertData.alert_level || 3;
    const cfg     = LEVEL_CONFIG[level] || LEVEL_CONFIG[3];
    const action  = (alertData.action_detected || alertData.alert_type || 'Unknown').replace(/_/g, ' ').toUpperCase();
    const camera  = alertData.camera_name || alertData.camera_id || 'Unknown Camera';
    const zone    = alertData.zone ? `Zone ${alertData.zone}` : '';
    const time    = new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    // Build DOM
    const toast = document.createElement('div');
    toast.className = 'sw-alert-toast';

    // Thumbnail or placeholder
    const thumbHtml = alertData.snapshot_url
      ? `<img class="sw-alert-thumb" src="${alertData.snapshot_url}" alt="snapshot" onerror="this.style.display='none'">`
      : `<div class="sw-alert-thumb-placeholder">${cfg.icon}</div>`;

    toast.innerHTML = `
      <div class="sw-alert-stripe level-${level}"></div>
      <div class="sw-alert-body">
        ${thumbHtml}
        <div class="sw-alert-info">
          <div class="sw-alert-badge level-${level}">
            <span class="sw-alert-pulse"></span>
            ${cfg.label}
          </div>
          <div class="sw-alert-action">${action}</div>
          <div class="sw-alert-meta">
            <span>📷 ${camera}</span>
            ${zone ? `<span>📍 ${zone}</span>` : ''}
            <span>🕐 ${time}</span>
          </div>
        </div>
      </div>
      <div class="sw-alert-footer">
        <div class="sw-alert-timer">
          <div class="sw-alert-timer-bar level-${level}" style="width:100%"></div>
        </div>
        <button class="sw-alert-close" title="Dismiss">✕</button>
      </div>
    `;

    // Prepend (newest on top)
    _container.insertBefore(toast, _container.firstChild);

    // Dismiss logic
    let dismissed = false;
    function dismiss() {
      if (dismissed) return;
      dismissed = true;
      toast.classList.add('sw-dismissing');
      setTimeout(() => toast.remove(), 280);
    }

    toast.querySelector('.sw-alert-close').addEventListener('click', dismiss);

    // Timer bar countdown
    const bar      = toast.querySelector('.sw-alert-timer-bar');
    const duration = cfg.dismiss;
    const interval = 80;
    let   elapsed  = 0;
    const ticker   = setInterval(() => {
      elapsed += interval;
      bar.style.width = Math.max(0, 100 - (elapsed / duration) * 100) + '%';
      if (elapsed >= duration) { clearInterval(ticker); dismiss(); }
    }, interval);

    // Chime
    _chime(level);

    // Keep max 5 toasts
    const toasts = _container.querySelectorAll('.sw-alert-toast:not(.sw-dismissing)');
    if (toasts.length > 5) toasts[toasts.length - 1].querySelector('.sw-alert-close').click();

    return toast;
  }

  // ── WebSocket handler attachment ──────────────────────────────────────────
  function _attachMessageHandler(ws) {
    const _origOnMessage = ws.onmessage;

    ws.onmessage = function (event) {
      // Run existing handler first
      if (_origOnMessage) _origOnMessage.call(ws, event);

      try {
        const data = typeof event.data === 'string' ? JSON.parse(event.data) : event.data;

        if (data && data.type === 'alert' && data.alert) {
          _showToast(data.alert);
        }
      } catch (e) {
        /* not JSON — ignore */
      }
    };

    // Also handle addEventListener-style listeners
    const _origAddEvent = ws.addEventListener.bind(ws);
    ws._sw_intercepted = true;
  }

  // ── Public API ─────────────────────────────────────────────────────────────
  const SafeWatchAlerts = {
    /**
     * Create a new WebSocket connection and attach alert handling.
     * @param {string} wsUrl - e.g. 'ws://localhost:8000/ws/live'
     */
    init(wsUrl = 'ws://localhost:8000/ws/live') {
      _injectStyles();
      const ws = new WebSocket(wsUrl);
      ws.onopen    = () => console.log('[SafeWatchAlerts] WebSocket connected');
      ws.onclose   = () => console.log('[SafeWatchAlerts] WebSocket closed');
      ws.onerror   = (e) => console.warn('[SafeWatchAlerts] WebSocket error', e);
      _attachMessageHandler(ws);
      return ws;
    },

    /**
     * Attach alert popups to an EXISTING WebSocket instance.
     * Call this if your app already manages its own WebSocket.
     * @param {WebSocket} ws
     */
    attachTo(ws) {
      _injectStyles();
      if (!ws._sw_intercepted) _attachMessageHandler(ws);
    },

    /**
     * Show a test alert popup (useful during development).
     */
    test(level = 2) {
      _injectStyles();
      _ensureContainer();
      _showToast({
        alert_level:     level,
        alert_type:      'shouting',
        action_detected: 'shouting',
        camera_name:     'Main Entrance',
        zone:            2,
        snapshot_url:    null,
      });
    },
  };

  function _injectStyles() {
    if (document.getElementById('sw-alert-styles')) return;
    const style  = document.createElement('style');
    style.id     = 'sw-alert-styles';
    style.textContent = CSS;
    document.head.appendChild(style);
  }

  // Expose globally
  global.SafeWatchAlerts = SafeWatchAlerts;

})(window);