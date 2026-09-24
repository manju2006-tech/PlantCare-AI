// ============================================================
// admin.js
// ============================================================
document.addEventListener('DOMContentLoaded', function () {
  initAdminCharts();
  initUserActions();
});

function initAdminCharts() {
  var statsEl = document.getElementById('adminStatsData');
  if (!statsEl || !window.Chart) return;

  var stats;
  try {
    stats = JSON.parse(statsEl.textContent);
  } catch (err) {
    return;
  }

  var styles = getComputedStyle(document.documentElement);
  var primary = styles.getPropertyValue('--color-primary').trim() || '#1F7A4D';
  var disease = styles.getPropertyValue('--color-disease').trim() || '#D9502F';
  var healthy = styles.getPropertyValue('--color-healthy').trim() || '#1E9E52';
  var textColor = styles.getPropertyValue('--color-text-muted').trim() || '#52685B';
  var gridColor = styles.getPropertyValue('--color-border').trim() || 'rgba(0,0,0,0.1)';

  var splitCanvas = document.getElementById('adminSplitChart');
  if (splitCanvas) {
    new Chart(splitCanvas, {
      type: 'doughnut',
      data: {
        labels: ['Healthy', 'Diseased'],
        datasets: [{ data: [stats.healthy_scans, stats.diseased_scans], backgroundColor: [healthy, disease], borderWidth: 0 }],
      },
      options: { cutout: '70%', plugins: { legend: { position: 'bottom', labels: { color: textColor, boxWidth: 12, padding: 16 } } } },
    });
  }

  var dailyCanvas = document.getElementById('adminDailyChart');
  if (dailyCanvas) {
    new Chart(dailyCanvas, {
      type: 'line',
      data: {
        labels: stats.daily_counts.map(function (d) { return d.day; }),
        datasets: [{
          label: 'Scans', data: stats.daily_counts.map(function (d) { return d.count; }),
          borderColor: primary, backgroundColor: primary + '33', fill: true, tension: 0.35, pointBackgroundColor: primary,
        }],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: textColor }, grid: { color: gridColor } },
          y: { ticks: { color: textColor }, grid: { color: gridColor }, beginAtZero: true },
        },
      },
    });
  }

  var topCanvas = document.getElementById('adminTopDiseasesChart');
  if (topCanvas) {
    new Chart(topCanvas, {
      type: 'bar',
      data: {
        labels: stats.top_diseases.map(function (d) { return d.name; }),
        datasets: [{ label: 'Detections', data: stats.top_diseases.map(function (d) { return d.count; }), backgroundColor: disease, borderRadius: 6 }],
      },
      options: {
        indexAxis: 'y',
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: textColor }, grid: { color: gridColor }, beginAtZero: true },
          y: { ticks: { color: textColor }, grid: { display: false } },
        },
      },
    });
  }
}

function initUserActions() {
  document.querySelectorAll('[data-toggle-user]').forEach(function (btn) {
    btn.addEventListener('click', async function () {
      var id = this.getAttribute('data-toggle-user');
      var result = await postJSON('/admin/users/' + id + '/toggle', {});
      if (result.success) {
        var badge = document.querySelector('[data-status-badge="' + id + '"]');
        if (badge) {
          badge.textContent = result.is_active ? 'Active' : 'Inactive';
          badge.classList.toggle('badge-status-active', result.is_active);
          badge.classList.toggle('badge-status-inactive', !result.is_active);
        }
      } else {
        alert(result.error || 'Failed to update user status.');
      }
    });
  });

  document.querySelectorAll('[data-delete-user]').forEach(function (btn) {
    btn.addEventListener('click', async function () {
      var id = this.getAttribute('data-delete-user');
      if (!confirm('Delete this user and all of their scan history? This cannot be undone.')) return;

      var result = await postJSON('/admin/users/' + id + '/delete', {});
      if (result.success) {
        var row = document.querySelector('[data-user-row="' + id + '"]');
        if (row) row.remove();
      } else {
        alert(result.error || 'Failed to delete user.');
      }
    });
  });
}