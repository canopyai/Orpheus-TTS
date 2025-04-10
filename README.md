# Orpheus TTS

#### Updates 🔥
- [4/2025] We release a [family of multilingual models](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba) in a research preview. We release a [training guide](https://canopylabs.ai/releases/orpheus_can_speak_any_language#training) that explains how we created these models in the hopes that even better versions in both the languages released and new languages are created. We welcome feedback and criticism as well as invite questions in this [discussion](https://github.com/canopyai/Orpheus-TTS/discussions/123) for feedback and questions.

## Overview
Orpheus TTS is a SOTA open-source text-to-speech system built on the Llama-3b backbone. Orpheus demonstrates the emergent capabilities of using LLMs for speech synthesis.

[Check out our original blog post](https://canopylabs.ai/model-releases)


https://github.com/user-attachments/assets/ce17dd3a-f866-4e67-86e4-0025e6e87b8a

## Abilities

- **Human-Like Speech**: Natural intonation, emotion, and rhythm that is superior to SOTA closed source models
- **Zero-Shot Voice Cloning**: Clone voices without prior fine-tuning
- **Guided Emotion and Intonation**: Control speech and emotion characteristics with simple tags
- **Low Latency**: ~200ms streaming latency for realtime applications, reducible to ~100ms with input streaming

## Models

We provide 2 models English models, and additionally we offer the data processing scripts and sample datasets to make it very straightforward to create your own finetune.

1. [**Finetuned Prod**](https://huggingface.co/canopylabs/orpheus-tts-0.1-finetune-prod) – A finetuned model for everyday TTS applications

2. [**Pretrained**](https://huggingface.co/canopylabs/orpheus-tts-0.1-pretrained) – Our base model trained on 100k+ hours of English speech data

We also offer a family of multilingual models in a research release.

1. [**Multlingual Family**](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba) - 7 pairs of pretrained and finetuned models.

### Inference

#### Simple setup on colab

We offer a standardised prompt format across languages, and these notebooks illustrate how to use our models in English.

1. [Colab For Tuned Model](https://colab.research.google.com/drive/1KhXT56UePPUHhqitJNUxq63k-pQomz3N?usp=sharing) (not streaming, see below for realtime streaming) – A finetuned model for everyday TTS applications.
2. [Colab For Pretrained Model](https://colab.research.google.com/drive/10v9MIEbZOr_3V8ZcPAIh8MN7q2LjcstS?usp=sharing) – This notebook is set up for conditioned generation but can be extended to a range of tasks.

#### Streaming Inference Example

1. Clone this repo
   ```bash
   git clone https://github.com/canopyai/Orpheus-TTS.git
   ```
2. Navigate and install packages
   ```bash
   cd Orpheus-TTS && pip install orpheus-speech # uses vllm under the hood for fast inference
   ```
   vllm pushed a slightly buggy version on March 18th so some bugs are being resolved by reverting to `pip install vllm==0.7.3` after `pip install orpheus-speech`
4. Run the example below:
   ```python
   from orpheus_tts import OrpheusModel
   import wave
   import time
   
   def main():
       model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
       prompt = """Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we're all connected 24/7 but somehow people feel more alone than ever. And don't even get me started on how it's messing with kids' self-esteem and mental health and whatnot."""
   
       start_time = time.monotonic()
       syn_tokens = model.generate_speech(
           prompt=prompt,
           voice="tara",
       )
   
       with wave.open("output.wav", "wb") as wf:
           wf.setnchannels(1)
           wf.setsampwidth(2)
           wf.setframerate(24000)
   
           total_frames = 0
           chunk_counter = 0
           for audio_chunk in syn_tokens:  # output streaming
               chunk_counter += 1
               frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
               total_frames += frame_count
               wf.writeframes(audio_chunk)
           duration = total_frames / wf.getframerate()
   
           end_time = time.monotonic()
           print(f"It took {end_time - start_time} seconds to generate {duration:.2f} seconds of audio")
   
   if __name__ == "__main__":
       main()
   ```
#### Additional Functionality

1. Watermark your audio: Use Silent Cipher to watermark your audio generation; see [Watermark Audio Implementation](additional_inference_options/watermark_audio) for implementation.

2. For No GPU inference using Llama cpp see implementation [documentation](additional_inference_options/no_gpu/README.md) for implementation example


#### Prompting

1. The `finetune-prod` models: for the primary model, your text prompt is formatted as `{name}: I went to the ...`. The options for name in order of conversational realism (subjective benchmarks) are "tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe" for English - each language has different voices [see voices here] (https://canopylabs.ai/releases/orpheus_can_speak_any_language#info)). Our python package does this formatting for you, and the notebook also prepends the appropriate string. You can additionally add the following emotive tags: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`. For multilingual, see this [post](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba) for supported tags.

2. The pretrained model: you can either generate speech just conditioned on text, or generate speech conditioned on one or more existing text-speech pairs in the prompt. Since this model hasn't been explicitly trained on the zero-shot voice cloning objective, the more text-speech pairs you pass in the prompt, the more reliably it will generate in the correct voice.


Additionally, use regular LLM generation args like `temperature`, `top_p`, etc. as you expect for a regular LLM. `repetition_penalty>=1.1`is required for stable generations. Increasing `repetition_penalty` and `temperature` makes the model speak faster.


## Finetune Model

Here is an overview of how to finetune your model on any text and speech.
This is a very simple process analogous to tuning an LLM using Trainer and Transformers.

You should start to see high quality results after ~50 examples but for best results, aim for 300 examples/speaker.

1. Your dataset should be a huggingface dataset in [this format](https://huggingface.co/datasets/canopylabs/zac-sample-dataset)
2. We prepare the data using [this notebook](https://colab.research.google.com/drive/1wg_CPCA-MzsWtsujwy-1Ovhv-tn8Q1nD?usp=sharing). This pushes an intermediate dataset to your Hugging Face account which you can can feed to the training script in finetune/train.py. Preprocessing should take less than 1 minute/thousand rows.
3. Modify the `finetune/config.yaml` file to include your dataset and training properties, and run the training script. You can additionally run any kind of huggingface compatible process like Lora to tune the model.
   ```bash
    pip install transformers datasets wandb trl flash_attn torch
    huggingface-cli login <enter your HF token>
    wandb login <wandb token>
    accelerate launch train.py
   ```
### Additional Resources
1. [PEFT finetuning with unsloth](https://github.com/unslothai/notebooks/pull/17/files)
   
## Pretrain Model

This is a very simple process analogous to training an LLM using Trainer and Transformers.

The base model provided is trained over 100k hours. I recommend not using synthetic data for training as it produces worse results when you try to finetune specific voices, probably because synthetic voices lack diversity and map to the same set of tokens when tokenised (i.e. lead to poor codebook utilisation).

We train the 3b model on sequences of length 8192 - we use the same dataset format for TTS finetuning for the <TTS-dataset> pretraining. We chain input_ids sequences together for more efficient training. The text dataset required is in the form described in this issue [#37 ](https://github.com/canopyai/Orpheus-TTS/issues/37). 

If you are doing extended training this model, i.e. for another language or style we recommend starting with finetuning only (no text dataset). The main idea behind the text dataset is discussed in the blog post. (tldr; doesn't forget too much semantic/reasoning ability so its able to better understand how to intone/express phrases when spoken, however most of the forgetting would happen very early on in the training i.e. <100000 rows), so unless you are doing very extended finetuning it may not make too much of a difference.

## Also Check out

While we can't verify these implementations are completely accurate/bug free, they have been recommended on a couple of forums, so we include them here:

1. [A lightweight client for running Orpheus TTS locally using LM Studio API](https://github.com/isaiahbjork/orpheus-tts-local)
2. [Open AI compatible Fast-API implementation](https://github.com/Lex-au/Orpheus-FastAPI)
3. [HuggingFace Space kindly set up by MohamedRashad](https://huggingface.co/spaces/MohamedRashad/Orpheus-TTS)
4. [Gradio WebUI that runs smoothly on WSL and CUDA](https://github.com/Saganaki22/OrpheusTTS-WebUI)


# Checklist

- [x] Release 3b pretrained model and finetuned models
- [ ] Release pretrained and finetuned models in sizes: 1b, 400m, 150m parameters
- [ ] Fix glitch in realtime streaming package that occasionally skips frames.
- [ ] Fix voice cloning Colab notebook implementation
