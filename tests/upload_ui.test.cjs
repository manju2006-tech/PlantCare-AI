const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function uploadPage(withBootstrap = true) {
  const elements = {};
  const ids = [
    'dropzone', 'fileInput', 'previewWrap', 'previewImage', 'uploadForm',
    'analyzeBtn', 'progressWrap', 'progressBar', 'progressLabel', 'uploadError',
    'changeImageBtn', 'unsupportedPlantModal', 'unsupportedPlantMessage',
    'retryUnsupportedPlantBtn',
  ];
  for (const id of ids) {
    const classes = new Set(['d-none']);
    elements[id] = {
      listeners: {}, style: {}, value: '', src: '', textContent: '',
      classList: {
        add: (name) => classes.add(name), remove: (name) => classes.delete(name),
        contains: (name) => classes.has(name),
      },
      addEventListener(name, callback) { this.listeners[name] = callback; },
      click() {},
    };
  }
  const requests = [];
  let popups = 0;
  const bootstrap = {
    Modal: { getOrCreateInstance: () => ({ show: () => { popups++; } }) },
  };
  const window = { location: {} };
  if (withBootstrap) window.bootstrap = bootstrap;
  const sandbox = {
    window, bootstrap,
    document: {
      getElementById: (id) => elements[id], querySelector: () => null,
      addEventListener: (name, callback) => callback(),
    },
    FileReader: class {
      readAsDataURL() { this.onload({ target: { result: 'data:image/jpeg;base64,test' } }); }
    },
    FormData: class { append() {} },
    XMLHttpRequest: class {
      constructor() { this.upload = {}; requests.push(this); }
      open() {}
      setRequestHeader() {}
      send() {}
    },
    setTimeout: (callback) => callback(),
  };
  vm.runInNewContext(fs.readFileSync(
    path.join(__dirname, '../static/js/upload.js'), 'utf8'
  ), sandbox);
  const choose = () => {
    elements.fileInput.files = [{ type: 'image/jpeg', size: 100 }];
    elements.fileInput.listeners.change.call(elements.fileInput);
  };
  const submit = () => elements.uploadForm.listeners.submit({ preventDefault() {} });
  const respond = (status, payload) => {
    const xhr = requests.at(-1);
    xhr.status = status;
    xhr.responseText = typeof payload === 'string' ? payload : JSON.stringify(payload);
    xhr.onload();
  };
  return { elements, requests, choose, submit, respond, window, popups: () => popups };
}

test('unsupported response opens popup and retry clears the rejected photo', () => {
  const page = uploadPage();
  page.choose();
  page.submit();
  page.respond(422, { success: false, unknown_plant: true, error: 'Plant not supported.' });
  assert.equal(page.popups(), 1);
  assert.equal(page.elements.unsupportedPlantMessage.textContent, 'Plant not supported.');
  assert.equal(page.elements.uploadError.classList.contains('d-none'), false);
  page.elements.retryUnsupportedPlantBtn.listeners.click();
  assert.equal(page.elements.previewImage.src, '');
  assert.equal(page.elements.analyzeBtn.disabled, true);
  assert.equal(page.elements.uploadError.classList.contains('d-none'), true);
  page.submit();
  assert.equal(page.requests.length, 1);
  assert.equal(page.elements.uploadError.textContent, 'Please choose an image first.');
});

test('availability errors display an error without claiming the plant is unsupported', () => {
  const page = uploadPage();
  page.choose(); page.submit();
  page.respond(503, { success: false, error: 'Verification unavailable.' });
  assert.equal(page.popups(), 0);
  assert.equal(page.elements.uploadError.textContent, 'Verification unavailable.');
});

test('unsupported responses still explain the error when Bootstrap is unavailable', () => {
  const page = uploadPage(false);
  page.choose(); page.submit();
  page.respond(422, { success: false, unknown_plant: true, error: 'Unable to identify.' });
  assert.equal(page.popups(), 0);
  assert.equal(page.elements.uploadError.textContent, 'Unable to identify.');
});

test('accepted uploads continue to the result page', () => {
  const page = uploadPage();
  page.choose(); page.submit();
  page.respond(200, { success: true, redirect_url: '/result/21' });
  assert.equal(page.popups(), 0);
  assert.equal(page.window.location.href, '/result/21');
});

test('a non-JSON server error allows another attempt', () => {
  const page = uploadPage();
  page.choose(); page.submit();
  page.respond(500, '<html>Server error</html>');
  assert.equal(page.elements.analyzeBtn.disabled, false);
  assert.equal(page.elements.uploadError.textContent, 'Unexpected server response. Please try again.');
});
