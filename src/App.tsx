import logo from './logo.svg';
import './App.css';
import React, {useEffect, useState} from 'react';
import {Button, Container, createMuiTheme, IconButton, Snackbar, TextField, ThemeProvider} from '@material-ui/core';
import * as Icons from '@material-ui/icons';
import * as Colors from '@material-ui/core/colors';
import * as tf from '@tensorflow/tfjs';

const darkTheme = createMuiTheme({
  palette: {
    primary: {main: Colors.common.white},
    type: "dark"
  }
});

let model, metadata;

async function loadModel() {
  let flag = false;
  if (model === undefined) {
    model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json');
    flag = true;
  }
  if (metadata === undefined) {
    metadata = await (await fetch('https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json')).json();
  }
  return flag;
}

function padSequences(sequences, maxLen, padding = 'pre', truncating = 'pre', value = 0) {
  return sequences.map(seq => {
    if (seq.length > maxLen) {
      if (truncating === 'pre') {
        seq.splice(0, seq.length - maxLen);
      } else {
        seq.splice(maxLen, seq.length - maxLen);
      }
    }

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

function getSentimentScore(text) {
  const inputText = text.trim().toLowerCase().replace(/([.,!])/g, '').split(' ');
  // Convert the words to a sequence of word indices.
  const sequence = inputText.map(word => {
    let wordIndex = metadata.word_index[word] + metadata.index_from;
    if (wordIndex > metadata.vocabulary_size) {
      wordIndex = 2;
    }
    return wordIndex;
  });
  // Perform truncation and padding.
  const paddedSequence = padSequences([sequence], metadata.max_len);
  const input = tf.tensor2d(paddedSequence, [1, metadata.max_len]);

  const predictOut = model.predict(input);
  const score = predictOut.dataSync()[0];
  predictOut.dispose();

  return score;
}

function App() {
  let [text, setText] = useState('');
  let [snackOpen, setSnackOpen] = useState(false);
  let [message, setMessage] = useState('');

  function handleSnackClose(event, reason = '') {
    if (reason === 'clickaway') return;
    setSnackOpen(false);
  }

  function textUpdate(event: React.ChangeEvent<HTMLTextAreaElement>) {
    setText(event.target.value);
  }

  function onClick(_event: React.MouseEvent<HTMLButtonElement>) {
    if (text === '') {
      setMessage("Please enter a message.");
      setSnackOpen(true);
      return;
    }
    const SentimentThreshold = {
      Positive: 0.66,
      Neutral: 0.33,
      Negative: 0
    }

    if (model !== undefined) {
      setSnackOpen(false);
      const sentimentScore = getSentimentScore(text);
      let textSentiment = '';
      if (sentimentScore > SentimentThreshold.Positive) {
        textSentiment = 'positive';
      } else if (sentimentScore > SentimentThreshold.Neutral) {
        textSentiment = 'neutral';
      } else if (sentimentScore >= SentimentThreshold.Negative) {
        textSentiment = 'negative';
      }
      setMessage("That is a " + textSentiment + " remark! With a sentiment score of " + sentimentScore.toFixed(4)*100 + "%.");
      setSnackOpen(true);
    }
  }

  useEffect(() => {
    loadModel().then(r => {
      if (r) console.log('Model loaded! ', metadata);
    });
  });

  return (
    <div className="App">
      <header className="App-header">
        <ThemeProvider theme={darkTheme}>
          <Container maxWidth="md">
            <TextField variant="filled" multiline fullWidth onChange={textUpdate} value={text}/>
          </Container>
          <br/>
          <Button variant="outlined" onClick={onClick}>Process</Button>
          <img src={logo} className="App-logo" alt="logo" width="25%"/>
          <Snackbar
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'left',
            }}
            open={snackOpen}
            autoHideDuration={6000}
            onClose={handleSnackClose}
            message={message}
            action={
              <React.Fragment>
                <IconButton size="small" aria-label="close" color="inherit">
                  <Icons.Close fontSize="small" onClick={handleSnackClose}/>
                </IconButton>
              </React.Fragment>
            }
          />
        </ThemeProvider>
      </header>
    </div>
  );
}

export default App;
