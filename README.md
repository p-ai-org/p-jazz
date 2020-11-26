# Generating artificial jazz music using GPT2 and Seq2Seq LSTM

**Goal**: To create a machine learning model that can generate novel jazz piano

**TL;DR**: Check out some of our artificial jazz results here: [soundcloud.com/pjazz-marms](http://soundcloud.com/pjazz-marms)

### Data processing: 
The training data we used were MIDI files taken from [here](http://bushgrafts.com) that were first converted into piano roll images. To see the current MIDI files used to train the models of our project, see [here](https://drive.google.com/drive/folders/1Q30i6j2lURjpStxhIqay73pcjgTMHNay?usp=sharing).
### Training: 
We experimented with several different types of models, but eventually settled on two main routes: text generation (GPT2) and sequence2sequence LSTM.

#### Using [GPT2](https://openai.com/blog/better-language-models/): 

- The first attempt was taking each timestep and turning each bit (aka a note) into a word, a word-interpretation of the piano roll and assigned weights to it for rests. <PICTURE OF THE PIANO ROLL> <PICTURE OF BEING TURNED INTO GPT TEXT>
- We then tried to use an on-and-off method. For this method, we only indicated when a note started and when it ended.  As opposed to the every-timestep version where we were being very explicit about all the notes that are being played at any given time. <EXAMPLE OF ON/OFF TEXT>
- The learning was slow at first but there seems to be an improvement. Qualitatively, it just sounded more musical, partly because it played fewer notes and held them for longer. While we can't exactly look into the model and point out exactly why this model works better, we can reason that with the original pre-processing method, there's no concept of notes being held, only played at every timestep, so that likely resulted in it mimicking that, instead of holding out single notes. Also, on/off is just closer to representing what's actually happening in the music, so it's felt more intuitive to proceed with the training model in this way.
- We then tried using the model to train saxophone solo pieces as well as classical music pieces to see how this would change the outputs. We're still working on running tests on this method.
- We also tried to construct a model to separate melody and harmony to see how generating these separately would impact the output. We had a decent heuristic for separating the melody from the harmony, but our challenge was in choosing the right architecture to train both in parallel, which we weren't able to design in our limited time.

#### Using [Seq2Seq LSTM](https://towardsdatascience.com/how-to-implement-seq2seq-lstm-model-in-keras-shortcutnlp-6f355f3e5639)

- We started off with using a many-to-one LSTM model. We took the entire corpus of piano rolls as training data and trained a stack LSTM to predict the next timestep from (in our test runs) the last 24 timesteps.
- However, the loss function led it to just predict the previous timestep (so it looked like the last note played was held down for the entirety of the prediction), since technically that was not a wrong option the vast majority of the time. <NOTE BEING HELD DOWN FOREVER>
- After letting the many-to-one model run a little more, we found that the results it produces is not actually too bad as previously mentioned. Its predictions are just very slow rhythmically compared to what preceeds it (the training set). <NOTE NOT BEING HELD DOWN FOREVER>
- Then we decided to switch over to using the sequence-to-sequence encoder-decoder LSTM model. We hypothesised that it would make more sense, given the ordered, sequential nature of music. Typically, this model is used for translation-related tasks. The challenge in applying it to music piano rolls was then than how to cut up these "phrases" (a sequence).
- Our current algorithm takes note of when a pause (timesteps where no notes are being played) is sufficiently long enough (0.25 of a quarter note) and uses that as a marker of the "cut-off point". We did also set 2 quarter notes as the minimum length of a phrase and 8 quarter notes as the maximum. If the current phrase being tracked runs more than 8 quarter notes, we cut it off (before it meets a pause) and discard the remainder of that phrase and restart the new phrase tracker after the first pause it meets from the cutoff point.
- Although seq2seq isn't a perfect model either (because in theory there's no algorithmic process that converts from one phrase to another), we did it as an experiment to see if it could pick up patterns from the previous phrase.

### A few challenges we faced:
- One significant challenge was that the size of the data we used is comparatively small compared to an ideal size for a deep learning ML project.
- There was a major design question about the data we used: what kind of jazz music did we want to use (blues, swing, free jazz, etc)? Even after determining a specific style of jazz, should we use music from specific artists (who had similar styles)? That would further limit the available data we could use to train our models.
- Jazz as a genre is also tricky to analyze because there exists a huge variety of the styles it can be played in, and the use of syncopation and swing, fast runs vs the short runs makes it difficult to represent in code. We also faced some normalisation issues because of the chromaticism in jazz, that makes it hard to standardise the keys the music were played in.


_This project was partly done as a final project for a course with [Professor Mike Izbicki](http://izbicki.me) at Claremont McKenna College._
