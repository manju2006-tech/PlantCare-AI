document.addEventListener("DOMContentLoaded", function () {
    const passwordInput = document.getElementById("password");
    const togglePassword = document.getElementById("togglePassword");

    if (!passwordInput || !togglePassword) {
        return;
    }

    const icon = togglePassword.querySelector("i");

    togglePassword.addEventListener("click", function () {
        if (passwordInput.type === "password") {
            passwordInput.type = "text";

            icon.classList.remove("bi-eye");
            icon.classList.add("bi-eye-slash");

            togglePassword.setAttribute("aria-label", "Hide password");
        } else {
            passwordInput.type = "password";

            icon.classList.remove("bi-eye-slash");
            icon.classList.add("bi-eye");

            togglePassword.setAttribute("aria-label", "Show password");
        }
    });
});