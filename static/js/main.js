// ============================================================
// main.js
// ============================================================
/**
 * Shared UI behaviors used across every page: flash-message auto-dismiss,
 * mobile navbar auto-close, scroll-reveal animation, Bootstrap tooltip
 * init, and a small postJSON() helper (CSRF-aware) reused by
 * history.js / admin.js for AJAX actions.
 */
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.alert[data-auto-dismiss]').forEach(function (alert) {
    setTimeout(function () {
      if (window.bootstrap && bootstrap.Alert) {
        bootstrap.Alert.getOrCreateInstance(alert).close();
      } else {
        alert.remove();
      }
    }, 5000);
  });

  var navCollapse = document.getElementById('mainNavbar');
  if (navCollapse) {
    navCollapse.querySelectorAll('.nav-link').forEach(function (link) {
      link.addEventListener('click', function () {
        if (navCollapse.classList.contains('show') && window.bootstrap) {
          bootstrap.Collapse.getOrCreateInstance(navCollapse).hide();
        }
      });
    });
  }

  var revealEls = document.querySelectorAll('.reveal-on-scroll');
  if (revealEls.length && 'IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('fade-up');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.15 });
    revealEls.forEach(function (el) { observer.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.classList.add('fade-up'); });
  }

  document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function (el) {
    if (window.bootstrap) new bootstrap.Tooltip(el);
  });
});

async function postJSON(url, data) {
  var meta = document.querySelector('meta[name="csrf-token"]');
  var csrfToken = meta ? meta.getAttribute('content') : '';

  var response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': csrfToken,
    },
    body: JSON.stringify(data || {}),
  });
  return response.json();
}