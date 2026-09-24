// ============================================================
// history.js
// ============================================================
document.addEventListener('DOMContentLoaded', function () {
  var searchInput = document.getElementById('historySearch');
  var filterSelect = document.getElementById('historyFilter');
  var resultsGrid = document.getElementById('historyResults');
  var emptyState = document.getElementById('historyEmptyState');
  var paginationWrap = document.getElementById('historyPagination');

  if (!resultsGrid) return;

  var debounceTimer = null;

  function statusBadge(item) {
    return item.is_healthy
      ? '<span class="status-badge healthy"><i class="bi bi-check-circle-fill"></i> Healthy</span>'
      : '<span class="status-badge diseased"><i class="bi bi-exclamation-triangle-fill"></i> Diseased</span>';
  }

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str == null ? '' : str;
    return div.innerHTML;
  }

  function buildCard(item) {
    return (
      '<div class="col-md-6 col-lg-4 reveal-item" data-history-item="' + item.id + '">' +
        '<div class="glass-card glass-card--hover history-card">' +
          '<img src="/uploads/' + encodeURIComponent(item.image_filename) + '" alt="' + escapeHtml(item.plant_name) + '" class="history-thumb" loading="lazy">' +
          '<div class="history-meta">' +
            '<h6>' + escapeHtml(item.disease_name) + '</h6>' +
            '<div class="history-date">' + escapeHtml(item.plant_name) + ' &middot; ' + escapeHtml(item.created_at) + '</div>' +
            '<div class="mt-2 d-flex align-items-center gap-2 flex-wrap">' +
              statusBadge(item) +
              '<span class="font-mono text-muted-custom small">' + item.confidence + '%</span>' +
            '</div>' +
          '</div>' +
        '</div>' +
        '<div class="d-flex gap-2 mt-2">' +
          '<a href="/result/' + item.id + '" class="btn btn-plant-outline btn-sm flex-fill">View Report</a>' +
          '<button type="button" class="btn btn-danger-soft btn-sm" data-delete-id="' + item.id + '" aria-label="Delete this scan">' +
            '<i class="bi bi-trash3"></i>' +
          '</button>' +
        '</div>' +
      '</div>'
    );
  }

  function attachDeleteHandlers() {
    resultsGrid.querySelectorAll('[data-delete-id]').forEach(function (btn) {
      btn.addEventListener('click', async function () {
        var id = this.getAttribute('data-delete-id');
        if (!confirm('Delete this scan from your history? This cannot be undone.')) return;

        var result = await postJSON('/history/delete/' + id, {});
        if (result.success) {
          var card = resultsGrid.querySelector('[data-history-item="' + id + '"]');
          if (card) card.remove();
          if (!resultsGrid.children.length && emptyState) {
            emptyState.classList.remove('d-none');
          }
        } else {
          alert(result.error || 'Failed to delete this scan.');
        }
      });
    });
  }

  function renderPagination(data) {
    if (!paginationWrap) return;
    if (data.pages <= 1) {
      paginationWrap.innerHTML = '';
      return;
    }

    var html = '<nav aria-label="History pagination"><ul class="pagination justify-content-center">';
    for (var p = 1; p <= data.pages; p++) {
      html += '<li class="page-item ' + (p === data.page ? 'active' : '') + '">' +
                '<button type="button" class="page-link" data-page="' + p + '">' + p + '</button>' +
              '</li>';
    }
    html += '</ul></nav>';
    paginationWrap.innerHTML = html;

    paginationWrap.querySelectorAll('[data-page]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        fetchHistory(parseInt(btn.getAttribute('data-page'), 10));
      });
    });
  }

  async function fetchHistory(page) {
    var search = searchInput.value.trim();
    var filter = filterSelect.value;
    var params = new URLSearchParams({ search: search, filter: filter, page: page || 1 });

    var response = await fetch('/api/history?' + params.toString());
    var data = await response.json();
    if (!data.success) return;

    resultsGrid.innerHTML = data.items.map(buildCard).join('');
    attachDeleteHandlers();

    if (emptyState) emptyState.classList.toggle('d-none', data.items.length > 0);
    renderPagination(data);
  }

  searchInput.addEventListener('input', function () {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(function () { fetchHistory(1); }, 400);
  });

  filterSelect.addEventListener('change', function () { fetchHistory(1); });

  attachDeleteHandlers();
});