// ============================================================
// dashboard.js
// ============================================================
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('[data-count-to]').forEach(function (el) {
    var target = parseInt(el.getAttribute('data-count-to'), 10) || 0;
    var duration = 900;
    var start = performance.now();

    function step(now) {
      var progress = Math.min((now - start) / duration, 1);
      el.textContent = Math.round(progress * target);
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  });

  var chartCanvas = document.getElementById('dashboardSplitChart');
  if (chartCanvas && window.Chart) {
    var healthy = parseInt(chartCanvas.dataset.healthy, 10) || 0;
    var diseased = parseInt(chartCanvas.dataset.diseased, 10) || 0;
    var styles = getComputedStyle(document.documentElement);
    var healthyColor = styles.getPropertyValue('--color-healthy').trim() || '#1E9E52';
    var diseaseColor = styles.getPropertyValue('--color-disease').trim() || '#D9502F';
    var textColor = styles.getPropertyValue('--color-text-muted').trim() || '#52685B';

    if (healthy === 0 && diseased === 0) return;

    new Chart(chartCanvas, {
      type: 'doughnut',
      data: {
        labels: ['Healthy', 'Diseased'],
        datasets: [{
          data: [healthy, diseased],
          backgroundColor: [healthyColor, diseaseColor],
          borderWidth: 0,
          hoverOffset: 6,
        }],
      },
      options: {
        cutout: '70%',
        plugins: {
          legend: { position: 'bottom', labels: { color: textColor, boxWidth: 12, padding: 16 } },
        },
      },
    });
  }
});