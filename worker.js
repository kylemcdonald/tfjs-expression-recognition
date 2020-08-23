import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-wasm';
import { loadGraphModel } from '@tensorflow/tfjs-converter';
import * as blazeface from '@tensorflow-models/blazeface';

// scale up the detection from blazeface to capture more context
const SCALE_FACTOR = 1.25;

const IMG_SIZE = [96, 96];
const SUB_FACTOR = 127.5;
const DIV_FACTOR = 127.5;

addEventListener('message', prepare, { once: true })

async function prepare() {
  const loadStart = performance.now();
  // 'webgl', 'cpu', or 'wasm'
  await tf.setBackend('webgl');
  const detector = await blazeface.load();
  const model = await loadGraphModel('mobilenetv2-ferplus-0.830/model.json');
  const loadDuration = performance.now() - loadStart;

  addEventListener('message', async (e) => {
    const printMessages = []

    const { inputArray, width, height } = e.data
    console.debug(inputArray)
    const int32 = new Int32Array(inputArray);
    console.debug(int32)
    tf.engine().startScope();
    const input = tf.tensor3d(int32, [height, width, 3])
    console.debug(input)
    const detectorStart = performance.now();
    const detections = await detector.estimateFaces(input);
    const detectorDuration = performance.now() - detectorStart;
    printMessages.push('detector: ' + detectorDuration.toFixed() + 'ms')
    console.debug(printMessages[printMessages.length - 1])


    if (detections.length > 0) {
      const detection = detections[0];

      // raw detection
      const [x1, y1] = detection.topLeft;
      const [x2, y2] = detection.bottomRight;

      // scaled and equal aspect detection
      const size = Math.max(x2 - x1, y2 - y1) * SCALE_FACTOR;
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const x = cx - (size / 2);
      const y = cy - (size / 2);

      const batch = preprocess(input, cx, cy, size, width, height);
      const modelStart = performance.now();
      const prediction = model.predict(batch).arraySync()[0];
      const modelDuration = performance.now() - modelStart;
      printMessages.push('model: ' + modelDuration.toFixed() + 'ms')
      console.debug(printMessages[printMessages.length - 1])
      printMessages.push('Tensors: ' + tf.memory().numTensors)
      console.debug(printMessages[printMessages.length - 1]);

      const debugMessage = printMessages.join('</br>')

      postMessage({
        prediction,
        x, x1, x2, y, y1, y2,
        width, height, size,
        debugMessage
      })
    }
    else {
      const debugMessage = printMessages.join('</br>')
      postMessage({
        debugMessage
      })
    }
  })

  postMessage({ debugMessage: "model load: " + loadDuration.toFixed() + 'ms' })
  tf.engine().endScope();
}



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

