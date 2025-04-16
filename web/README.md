# Orpheus TTS

## Overview

Orpheus TTS is an open-source text-to-speech system built on the Llama-3B backbone. Orpheus demonstrates the emergent capabilities of using LLMs for speech synthesis. Check out our [blog post](https://canopylabs.ai/model-releases) to learn more.

## Abilities

- **Human-Like Speech**: Natural intonation, emotion, and rhythm that is superior to SOTA closed source models
- **Zero-Shot Voice Cloning**: Clone voices without prior fine-tuning
- **Guided Emotion and Intonation**: Control speech and emotion characteristics with simple tags
- **Low Latency**: ~200ms streaming latency for realtime applications, reducible to ~100ms with input streaming

## Inference

```js
import { OrpheusModel } from "orpheus-speech";

// Load the model
const model_id = "onnx-community/orpheus-3b-0.1-ft-ONNX";
const orpheus = await OrpheusModel({
  model_name: model_id,
  dtype: "q4f16",
  device: "webgpu",
});

// Create a new audio stream
const prompt = "Hey there my name is Tara, <chuckle> and I'm a speech generation model that can sound like a person.";
const voice = "tara";
const stream = orpheus.generate_speech({ prompt, voice });

// Iterate over the audio chunks
for await (const chunk of stream) {
  console.log(chunk.audio.length);
}

// Save the final audio to a file
const result = stream.data;
result.save("output.wav");
```

### Prompting

Voice options (in order of perceived conversational realism): "tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe". You can also add the following emotive tags: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`.

Additionally, you can use regular LLM generation args like `temperature`, `top_p`, etc. We recommend setting `repetition_penalty>=1.1` for more stable generation. Increasing `repetition_penalty` and `temperature` makes the model speak faster.
