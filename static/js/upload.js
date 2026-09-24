// ============================================================
// upload.js
// ============================================================
/**
 * Upload page: drag-and-drop, client-side validation, image preview, and
 * an XMLHttpRequest-driven progress bar for the actual /predict call
 * (fetch() cannot report upload progress, hence XHR here specifically).
 */
document.addEventListener('DOMContentLoaded', function () {
  var dropzone = document.getElementById('dropzone');
  if (!dropzone) return;

  var fileInput = document.getElementById('fileInput');
  var previewWrap = document.getElementById('previewWrap');
  var previewImage = document.getElementById('previewImage');
  var uploadForm = document.getElementById('uploadForm');
  var analyzeBtn = document.getElementById('analyzeBtn');
  var progressWrap = document.getElementById('progressWrap');
  var progressBar = document.getElementById('progressBar');
  var progressLabel = document.getElementById('progressLabel');
  var errorAlert = document.getElementById('uploadError');
  var changeImageBtn = document.getElementById('changeImageBtn');

  var selectedFile = null;

  function showPreview(file) {
    selectedFile = file;
    var reader = new FileReader();
    reader.onload = function (e) {
      previewImage.src = e.target.result;
      dropzone.classList.add('d-none');
      previewWrap.classList.remove('d-none');
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    dropzone.classList.remove('d-none');
    previewWrap.classList.add('d-none');
    analyzeBtn.disabled = true;
    progressWrap.classList.add('d-none');
    progressBar.style.width = '0%';
    hideError();
  }

  function showError(message) {
    errorAlert.textContent = message;
    errorAlert.classList.remove('d-none');
  }

  function hideError() {
    errorAlert.classList.add('d-none');
    errorAlert.textContent = '';
  }

  function validateAndPreview(file) {
    hideError();
    var allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (allowedTypes.indexOf(file.type) === -1) {
      showError('Unsupported file type. Please upload a PNG or JPG image.');
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      showError('File is too large. Maximum size is 8 MB.');
      return;
    }
    showPreview(file);
  }

  dropzone.addEventListener('click', function () { fileInput.click(); });
  dropzone.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', function () {
    if (this.files && this.files[0]) validateAndPreview(this.files[0]);
  });

  ['dragenter', 'dragover'].forEach(function (evt) {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach(function (evt) {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });
  dropzone.addEventListener('drop', function (e) {
    var files = e.dataTransfer.files;
    if (files && files[0]) validateAndPreview(files[0]);
  });

  if (changeImageBtn) changeImageBtn.addEventListener('click', resetUpload);

  uploadForm.addEventListener('submit', function (e) {
    e.preventDefault();
    if (!selectedFile) {
      showError('Please choose an image first.');
      return;
    }
    hideError();

    var formData = new FormData();
    formData.append('image', selectedFile);

    var meta = document.querySelector('meta[name="csrf-token"]');
    var csrfToken = meta ? meta.getAttribute('content') : '';

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/predict', true);
    xhr.setRequestHeader('X-CSRFToken', csrfToken);

    analyzeBtn.disabled = true;
    progressWrap.classList.remove('d-none');
    progressBar.style.width = '0%';
    progressLabel.textContent = 'Uploading...';

    xhr.upload.onprogress = function (e) {
      if (e.lengthComputable) {
        var percent = Math.round((e.loaded / e.total) * 90);
        progressBar.style.width = percent + '%';
        progressLabel.textContent = 'Uploading... ' + percent + '%';
      }
    };

    xhr.upload.onload = function () {
      progressBar.style.width = '92%';
      progressLabel.textContent = 'Analyzing leaf with AI model...';
    };

    xhr.onload = function () {
      analyzeBtn.disabled = false;
      var response;
      try {
        response = JSON.parse(xhr.responseText);
      } catch (err) {
        showError('Unexpected server response. Please try again.');
        progressWrap.classList.add('d-none');
        return;
      }

      if (xhr.status >= 200 && xhr.status < 300 && response.success) {
        progressBar.style.width = '100%';
        progressLabel.textContent = 'Done! Redirecting...';
        setTimeout(function () { window.location.href = response.redirect_url; }, 500);
      } else {
        showError(response.error || 'Something went wrong analyzing your image.');
        progressWrap.classList.add('d-none');
      }
    };

    xhr.onerror = function () {
      analyzeBtn.disabled = false;
      progressWrap.classList.add('d-none');
      showError('Network error. Please check your connection and try again.');
    };

    xhr.send(formData);
  });
});