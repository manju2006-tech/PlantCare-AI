/**
 * Plant Health Analytics page: Chart.js visuals built from the
 * analytics dict embedded by analytics.html (#analyticsData), the same
 * pattern admin.js uses for the admin dashboard charts.
 */
document.addEventListener('DOMContentLoaded', function () {
  var dataEl = document.getElementById('analyticsData');
  if (!dataEl || !window.Chart) return;

  var analytics;
  try {
    analytics = JSON.parse(dataEl.textContent);
  } catch (err) {
    return;
  }

  if (!analytics.total_scans) return; // empty state - nothing to chart

  var styles = getComputedStyle(document.documentElement);
  var healthyColor = styles.getPropertyValue('--color-healthy').trim() || '#1E9E52';
  var diseaseColor = styles.getPropertyValue('--color-disease').trim() || '#D9502F';
  var textColor = styles.getPropertyValue('--color-text-muted').trim() || '#52685B';
  var gridColor = styles.getPropertyValue('--color-border').trim() || 'rgba(0,0,0,0.1)';

  // --- Healthy vs Diseased doughnut ---
  var splitCanvas = document.getElementById('analyticsSplitChart');
  if (splitCanvas) {
    new Chart(splitCanvas, {
      type: 'doughnut',
      data: {
        labels: ['Healthy', 'Diseased'],
        datasets: [{
          data: [analytics.healthy_count, analytics.diseased_count],
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

  // --- Disease distribution horizontal bar chart ---
  var distCanvas = document.getElementById('analyticsDistributionChart');
  if (distCanvas && analytics.disease_distribution && analytics.disease_distribution.length) {
    new Chart(distCanvas, {
      type: 'bar',
      data: {
        labels: analytics.disease_distribution.map(function (d) { return d.name; }),
        datasets: [{
          label: 'Detections',
          data: analytics.disease_distribution.map(function (d) { return d.count; }),
          backgroundColor: diseaseColor,
          borderRadius: 6,
        }],
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
});