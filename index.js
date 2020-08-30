// in tfjs2, the webgl and cpu backends are also broken out separately
// but trying to combine these packages with tfjs 1 will cause problems
// import '@tensorflow/tfjs-backend-webgl';
// import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';

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

function receiveInference(e, ctx, inputElt, worker) {
  console.debug('received inference')
  const {
    prediction,
    x, x1, x2, y, y1, y2,
    width, height, size,
    debugMessage
  } = e.data

  // if nothing detected, only do a little work and then return
  if (!prediction) {
    document.getElementById('debug').innerHTML = debugMessage;

    setTimeout(() => {
      requestInference(inputElt, inputElt.videoWidth, inputElt.videoHeight, worker);
    }, 0)
    return
  }

  // draw raw detection
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = 'rgba(255, 0, 0)';
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  // draw scaled and equal aspect detection
  ctx.strokeStyle = 'rgba(0, 255, 0)';
  ctx.strokeRect(x, y, size, size);

  const labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt'];
  const blend = 0.5;
  labels.map((label, i) => {
    const elt = document.getElementById('amount-' + label);
    const prev = elt.value;
    const cur = prediction[i] * 100;
    elt.value = (prev * (1 - blend)) + (cur * blend);
  });

  document.getElementById('debug').innerHTML = debugMessage;

  setTimeout(() =>
    requestInference(inputElt, inputElt.videoWidth, inputElt.videoHeight, worker)
    , 0)
}

// create this once for use inside requestInference()
const fromPixels2DContext = document.createElement('canvas').getContext('2d');

async function requestInference(inputElt, width, height, worker) {
  console.debug('request inference')

  fromPixels2DContext.canvas.width = width;
  fromPixels2DContext.canvas.height = height;
  fromPixels2DContext.drawImage(inputElt, 0, 0, width, height);
  const buffer = fromPixels2DContext.getImageData(0, 0, width, height).data.buffer;
  
  worker.postMessage({ inputArray: buffer, width, height }, [buffer])
}

async function run() {
  // kickoff worker
  const worker = startupWorker()

  let video = await setupCamera();
  video.play();

  const inputElt = document.getElementById('video');
  const canvas = document.getElementById('output');
  canvas.width = inputElt.videoWidth;
  canvas.height = inputElt.videoHeight;
  const ctx = canvas.getContext('2d');

  worker.addEventListener('message', (e) => receiveInference(e, ctx, inputElt, worker))

  // inits the worker
  worker.postMessage("prepare")
}

function startupWorker() {
  return new Worker("worker.js");
}

document.addEventListener('DOMContentLoaded', (event) => {
  run();
})
