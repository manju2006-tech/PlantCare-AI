document.addEventListener('DOMContentLoaded', function () {
  var changePasswordForm = document.getElementById('changePasswordForm');
  var newPassword = document.getElementById('new_password');
  var confirmNewPassword = document.getElementById('confirm_new_password');
  var passwordMatchError = document.getElementById('passwordMatchError');

  if (changePasswordForm) {
    changePasswordForm.addEventListener('submit', function (e) {
      if (newPassword.value !== confirmNewPassword.value) {
        e.preventDefault();
        passwordMatchError.textContent = 'New passwords do not match.';
        passwordMatchError.classList.remove('d-none');
        confirmNewPassword.focus();
      } else {
        passwordMatchError.classList.add('d-none');
      }
    });
  }

  var deleteAccountForm = document.getElementById('deleteAccountForm');
  if (deleteAccountForm) {
    deleteAccountForm.addEventListener('submit', function (e) {
      var confirmed = confirm(
        'This will permanently delete your account and all scan history. This cannot be undone. Are you sure?'
      );
      if (!confirmed) {
        e.preventDefault();
      }
    });
  }
});