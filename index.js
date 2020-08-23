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

async function requestInference(inputElt, width, height, worker) {
  console.debug('request inference')

  // typically we use tf.tidy to track and clean new tensors
  // but in async code we use tf.engine().startScope()/endScope()
  // https://stackoverflow.com/a/59934467/940196
  tf.engine().startScope();

  const input = tf.browser.fromPixels(inputElt);
  console.debug(input)

  const transferrableInput = await (await input.data()).buffer
  console.log(transferrableInput)

  // send to worker
  worker.postMessage({ inputArray: transferrableInput, width, height }, [transferrableInput])
  tf.engine().endScope();
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
