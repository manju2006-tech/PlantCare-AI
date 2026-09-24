// ============================================================
// theme.js
// ============================================================
/**
 * Dark / light theme toggle.
 * This script is loaded synchronously in <head> (see base.html) BEFORE the
 * body renders, so the correct theme is applied immediately with no flash
 * of the wrong theme. Preference is remembered in localStorage.
 */
(function () {
  var STORAGE_KEY = 'plantcare_theme';
  var root = document.documentElement;

  function getPreferredTheme() {
    var stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    document.querySelectorAll('[data-theme-icon]').forEach(function (el) {
      el.className = 'bi ' + (theme === 'dark' ? 'bi-sun-fill' : 'bi-moon-stars-fill');
      el.setAttribute('data-theme-icon', theme);
    });
  }

  applyTheme(getPreferredTheme());

  document.addEventListener('DOMContentLoaded', function () {
    requestAnimationFrame(function () {
      document.body.classList.add('theme-transition');
    });

    document.querySelectorAll('.theme-toggle').forEach(function (btn) {
      btn.setAttribute('role', 'button');
      btn.setAttribute('tabindex', '0');
      btn.setAttribute('aria-label', 'Toggle dark and light mode');

      btn.addEventListener('click', function () {
        var current = root.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
        var next = current === 'dark' ? 'light' : 'dark';
        localStorage.setItem(STORAGE_KEY, next);
        applyTheme(next);
      });

      btn.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          btn.click();
        }
      });
    });
  });
})();