//import {padSequences} from './sequence_utils';

const HOSTED_URLS = {
  model:
      'model_js/model.json',
  metadata:
      'model_js/metadata.json'
};

const examples = {
  'example1':
      'Alice was beginning to get very tired of sitting by her sister on the bank.',
  'example2':
      'Buda-Pesth seems a wonderful place.',
  'example3':
      'Scepticism was as much the result of knowledge, as knowledge is of scepticism.'      
};


const PAD_INDEX = 0;  // Index of the padding character.
//export const OOV_INDEX = 2;  // Index fo the OOV character.

/**
 * Pad and truncate all sequences to the same length
 *
 * @param {number[][]} sequences The sequences represented as an array of array
 *   of numbers.
 * @param {number} maxLen Maximum length. Sequences longer than `maxLen` will be
 *   truncated. Sequences shorter than `maxLen` will be padded.
 * @param {'pre'|'post'} padding Padding type.
 * @param {'pre'|'post'} truncating Truncation type.
 * @param {number} value Padding value.
 */
function padSequences(
    sequences, maxLen, padding = 'pre', truncating = 'pre', value = PAD_INDEX) {
  // TODO(cais): This perhaps should be refined and moved into tfjs-preproc.
  return sequences.map(seq => {
    // Perform truncation.
    if (seq.length > maxLen) {
      if (truncating === 'pre') {
        seq.splice(0, seq.length - maxLen);
      } else {
        seq.splice(maxLen, seq.length - maxLen);
      }
    }

    // Perform padding.
    if (seq.length < maxLen) {
      const pad = [];
      for (let i = 0; i < maxLen - seq.length; ++i) {
        pad.push(value);
      }
      if (padding === 'pre') {
        seq = pad.concat(seq);
      } else {
        seq = seq.concat(pad);
      }
    }

    return seq;
  });
}


function status(statusText) {
  console.log(statusText);
  document.getElementById('status').textContent = statusText;
}

function showMetadata(metadataJSON) {
  document.getElementById('vocabularySize').textContent =
      metadataJSON['vocabulary_size'];
  document.getElementById('maxLen').textContent =
      metadataJSON['max_len'];
  var label_string = " ";
  for (var x in metadataJSON['class_labels']) {
    label_string += x + ' ->  ' + metadataJSON['class_labels'][x] + ', '
  }
  document.getElementById('classLabels').textContent = label_string
}

function settextField(text, predict) {
  const textField = document.getElementById('text-entry');
  textField.value = text;
  doPredict(predict);
}

function setPredictFunction(predict) {
  const textField = document.getElementById('text-entry');
  textField.addEventListener('input', () => doPredict(predict));
}

function disableLoadModelButtons() {
  document.getElementById('load-model').style.display = 'none';
}

function doPredict(predict) {
  const textField = document.getElementById('text-entry');
  const result = predict(textField.value);
  score_string = "Class scores: ";
  for (var x in result.score) {
    score_string += x + ' ->  ' + result.score[x].toFixed(3) + ', '
  }
  //console.log(score_string);
  status(
      score_string + ' elapsed: ' + result.elapsed.toFixed(3) + ' ms)');
}

function prepUI(predict) {
  setPredictFunction(predict);
  const testExampleSelect = document.getElementById('example-select');
  testExampleSelect.addEventListener('change', () => {
    settextField(examples[testExampleSelect.value], predict);
  });
  settextField(examples['example1'], predict);
}

async function urlExists(url) {
  status('Testing url ' + url);
  try {
    const response = await fetch(url, {method: 'HEAD'});
    return response.ok;
  } catch (err) {
    return false;
  }
}

async function loadHostedPretrainedModel(url) {
  status('Loading pretrained model from ' + url);
  try {
    const model = await tf.loadLayersModel(url);
    status('Done loading pretrained model.');
    disableLoadModelButtons();
    return model;
  } catch (err) {
    console.error(err);
    status('Loading pretrained model failed.');
  }
}

async function loadHostedMetadata(url) {
  status('Loading metadata from ' + url);
  try {
    const metadataJson = await fetch(url);
    const metadata = await metadataJson.json();
    status('Done loading metadata.');
    return metadata;
  } catch (err) {
    console.error(err);
    status('Loading metadata failed.');
  }
}

class Classifier {

  async init(urls) {
    this.urls = urls;
    this.model = await loadHostedPretrainedModel(urls.model);
    await this.loadMetadata();
    return this;
  }

  async loadMetadata() {
    const metadata =
        await loadHostedMetadata(this.urls.metadata);
    showMetadata(metadata);
    this.maxLen = metadata['max_len'];
    console.log('maxLen = ' + this.maxLen);
    this.wordIndex = metadata['word_index']
  }

  predict(text) {
    // Convert to lower case and remove all punctuations.
    const inputText =
        text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
    // Look up word indices.
    const sequence = inputText.map(word => {
      let wordIndex = 0;
      if (word in this.wordIndex) {
        wordIndex = this.wordIndex[word];
      } 
      return wordIndex;
    });
    
    // Perform truncation and padding.
    const paddedSequence = padSequences([sequence], this.maxLen);
    const input = tf.tensor2d(paddedSequence, [1, this.maxLen]);
    
//     const inputBuffer = tf.buffer([1, this.maxLen], 'float32');
//     for (let i = 0; i < inputText.length; ++i) {
//       const word = inputText[i];
//       inputBuffer.set(this.wordIndex[word], 0, i);
//       //console.log(word, this.wordIndex[word], inputBuffer);
//     }
//     const input = inputBuffer.toTensor();
//     //console.log(input);

    status('Running inference');
    const beginMs = performance.now();
    const predictOut = this.model.predict(input);
    //console.log(predictOut.dataSync());
    const score = predictOut.dataSync();//[0];
    predictOut.dispose();
    const endMs = performance.now();

    return {score: score, elapsed: (endMs - beginMs)};
  }
};

async function setup() {
  if (await urlExists(HOSTED_URLS.model)) {
    status('Model available: ' + HOSTED_URLS.model);
    const button = document.getElementById('load-model');
    button.addEventListener('click', async () => {
      const predictor = await new Classifier().init(HOSTED_URLS);
      prepUI(x => predictor.predict(x));
    });
    button.style.display = 'inline-block';
  }

  status('Standing by.');
}

setup();
