import '@tensorflow/tfjs-backend-wasm';
// in tfjs2, the webgl and cpu backends are also broken out separately
// but trying to combine these packages with tfjs 1 will cause problems
// import '@tensorflow/tfjs-backend-webgl';
// import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { loadGraphModel } from '@tensorflow/tfjs-converter';
import * as blazeface from '@tensorflow-models/blazeface';

// scale up the detection from blazeface to capture more context
const SCALE_FACTOR = 1.25;

const IMG_SIZE = [96, 96];
const SUB_FACTOR = 127.5;
const DIV_FACTOR = 127.5;
function preprocess(img, cx, cy, size, width, height) {
  return tf.tidy(() => {
    const top = cy - (size / 2);
    const left = cx - (size / 2);
    const boxes = [[
      top / height,
      left / width,
      (top + size) / height,
      (left + size) / width
    ]];
    const boxIndices = [0];
    return tf.image.cropAndResize(
        img.toFloat().expandDims(), boxes, boxIndices, IMG_SIZE)
        .sub(SUB_FACTOR)
        .div(DIV_FACTOR);
  })
}

async function setupCamera() {
  let video = document.getElementById('video');
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': { facingMode: 'user' },
  });
  video.srcObject = stream;
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
      return video;
    };
  });
}

async function inference(detector, model, inputElt, width, height, ctx) {
  let message = '';

  // typically we use tf.tidy to track and clean new tensors
  // but in async code we use tf.engine().startScope()/endScope()
  // https://stackoverflow.com/a/59934467/940196
  tf.engine().startScope();

  const input = tf.browser.fromPixels(inputElt);

  // gray conversion
  // const inputGray = inputRaw.mean(2);
  // const input = tf.stack([inputGray, inputGray, inputGray], 2);
  
  const detectorStart = performance.now();
  const detections = await detector.estimateFaces(input);
  const detectorDuration = performance.now() - detectorStart;
  message += 'detector: ' + detectorDuration.toFixed() + 'ms<br>';

  if (detections.length > 0) {
    const detection = detections[0];

    // raw detection
    const [x1,y1] = detection.topLeft;
    const [x2,y2] = detection.bottomRight;
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = 'rgba(255, 0, 0)';
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);

    // scaled and equal aspect detection
    const size = Math.max(x2-x1, y2-y1) * SCALE_FACTOR;
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    const x = cx - (size / 2);
    const y = cy - (size / 2);
    ctx.strokeStyle = 'rgba(0, 255, 0)';
    ctx.strokeRect(x, y, size, size);

    const batch = preprocess(input, cx, cy, size, width, height);
    const modelStart = performance.now();
    const prediction = model.predict(batch).arraySync()[0];
    const modelDuration = performance.now() - modelStart;
    message += 'model: ' + modelDuration.toFixed() + 'ms<br>';

    const labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt'];
    const blend = 0.5;
    labels.map((label, i) => {
      const elt = document.getElementById('amount-' + label);
      const prev = elt.value;
      const cur = prediction[i] * 100;
      elt.value = (prev * (1-blend)) + (cur * blend);
    });
  }

  message += 'Tensors: ' + tf.memory().numTensors;
  document.getElementById('debug').innerHTML = message;
  
  tf.engine().endScope();

  requestAnimationFrame(() => {
    inference(detector, model, inputElt, width, height, ctx);
  })
}

async function run() {
  // 'webgl', 'cpu', or 'wasm'
  await tf.setBackend('webgl');
  let video = await setupCamera();
  video.play();

  const inputElt = document.getElementById('video');
  const width = inputElt.videoWidth;
  const height = inputElt.videoHeight;

  const canvas = document.getElementById('output');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  const detector = await blazeface.load();
  const model = await loadGraphModel('mobilenetv2-ferplus-0.830/model.json');

  inference(detector, model, inputElt, width, height, ctx);

/*
  // some code for testing performance
  const n = 32;
  let high;
  let low;
  const input = tf.browser.fromPixels(inputElt);
  const batch = preprocess(input, cx, cy, size, width, height);
  const start = performance.now();
  for (let i = 0; i < n; i++) {
    const i_start = performance.now();
    prediction = model.predict(batch).arraySync()[0];
    const i_duration = performance.now() - i_start;
    if (!high || i_duration > high) {
      high = i_duration;
    }
    if (!low || i_duration < low) {
      low = i_duration;
    }
  }
  const duration = performance.now() - start;
  console.log(high.toFixed() + 'ms slow');
  console.log((duration / n).toFixed() + 'ms average');
  console.log(low.toFixed() + 'ms fast');
  */
}

document.addEventListener('DOMContentLoaded', (event) => {
  run();
})