import {
  AutoModelForCausalLM,
  AutoTokenizer,
  BaseStreamer,
  SnacDecoderModel,
  Tensor,
  RawAudio,
} from "@huggingface/transformers";

import { AsyncQueue } from "./queue.ts";
import WORKLET from "./worklet.ts";

/**
 * The number of tokens used for each frame (7 = 4 + 2 + 1).
 * This value stays fixed.
 */
const FRAME_SIZE = 7;

/**
 * The sampling rate of the SNAC decoder.
 */
const SAMPLING_RATE = 24000;

/**
 * The number of (valid) frames to send to the SNAC decoder.
 */
const NUM_PROCESSING_FRAMES = 3;

/**
 * To ensure consistency (and avoid artifacts), we utilize a sliding window with bidirectional padding.
 * It is in our best interest to keep this value to a minimum to avoid duplicate processing.
 * Test results indicate that a padding of 1 frame on each side works well.
 */
const NUM_PADDING_FRAMES_LEFT = 1;
const NUM_PADDING_FRAMES_RIGHT = 1;

/**
 * Define the total number of frames in the buffer.
 * This includes the padding frames on both sides and the processing frames in the middle.
 */
const NUM_BUFFER_FRAMES =
  NUM_PADDING_FRAMES_LEFT + NUM_PROCESSING_FRAMES + NUM_PADDING_FRAMES_RIGHT;
const BUFFER_SIZE = FRAME_SIZE * NUM_BUFFER_FRAMES;

/**
 * The number of samples per frame.
 */
const SAMPLES_PER_FRAME = 2048;

const AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"] as const;

type AvailableVoice = (typeof AVAILABLE_VOICES)[number];
type SupportedDtype = "fp32" | "fp16" | "q8" | "q4" | "q4f16";
type SupportedDevice = "cpu" | "cuda" | "webgpu" | "wasm";
type QueueType = AsyncQueue<bigint | null>;

type GenerateSpeechOptions = {
  prompt: string;
  voice?: AvailableVoice;
};

class AudioStream implements AsyncGenerator<RawAudio, void, unknown> {
  private chunks: RawAudio[] = [];
  private stream: AsyncGenerator<RawAudio, void, unknown>;
  private samples = 0;
  private playInit: Promise<{ context: AudioContext; node: AudioWorkletNode }> | undefined;
  private done = false;

  constructor(stream: AsyncGenerator<RawAudio>) {
    this.stream = stream;

    // Eagerly initialize the AudioContext and load the worklet module.
    // This is done to avoid blocking the main thread when the user calls play().
    if (typeof AudioContext !== "undefined") {
      this.playInit = (async () => {
        // Initialize the AudioContext and load the worklet module.
        const context = new AudioContext({ sampleRate: SAMPLING_RATE });

        const blob = new Blob([`(${WORKLET.toString()})()`], {
          type: "application/javascript",
        });
        const url = URL.createObjectURL(blob);
        await context.audioWorklet.addModule(url);
        URL.revokeObjectURL(url);

        // Create an instance of the AudioWorkletNode and connect it to the context.
        const node = new AudioWorkletNode(context, "buffered-audio-worklet-processor");
        node.connect(context.destination);

        return { context, node };
      })();
    }
  }

  next() {
    return this.stream.next();
  }

  throw(e: any) {
    return this.stream.throw(e);
  }

  return() {
    return this.stream.return();
  }

  [Symbol.asyncIterator]() {
    const self = this;
    return (async function* () {
      for await (const chunk of self.stream) {
        self.chunks.push(chunk);
        self.samples += chunk.audio.length;
        yield chunk;
      }
      self.done = true;
    })();
  }

  [Symbol.asyncDispose]() {
    return this.stream[Symbol.asyncDispose]();
  }

  get data(): RawAudio {
    const merged = new Float32Array(this.samples);
    let offset = 0;
    for (const chunk of this.chunks) {
      merged.set(chunk.audio, offset);
      offset += chunk.audio.length;
    }
    return new RawAudio(merged, SAMPLING_RATE);
  }

  get buffered(): number {
    return this.chunks.reduce((acc, chunk) => acc + chunk.audio.length, 0);
  }

  async play({ playbackDelayMs = 0, initialBufferMs = 0 } = {}) {
    if (!this.playInit) {
      throw new Error("Unable to play audio in this environment.");
    }
    const { context, node } = await this.playInit;
    await context.resume();

    console.log("AudioWorkletNode connected");
    const startTime = performance.now();
    const unprocessed = [];
    // let firstChunk = true;
    // let minSamples = Math.floor(
    const minSamples = (initialBufferMs / 1000) * SAMPLING_RATE;
    for await (const chunk of this) {
      console.log(performance.now(), startTime, playbackDelayMs);
      console.log(this.samples, minSamples);
      if (performance.now() - startTime < playbackDelayMs || this.samples < minSamples) {
        console.log("buffering", this.samples, minSamples);
        // We aren't ready to start, or we haven't received enough audio data yet.
        unprocessed.push(chunk);
        continue;
      }
      // We have enough audio data to start playing.
      if (unprocessed.length > 0) {
        for (const prevChunk of unprocessed) {
          console.log("playing buffered chunk", prevChunk.audio.length);
          node.port.postMessage(prevChunk.audio);
        }
        unprocessed.length = 0;
      }

      console.log("playing chunk", chunk.audio.length);
      // Send the audio data to the AudioWorkletNode.
      node.port.postMessage(chunk.audio);
      // firstChunk = false;
    }
    // await context.suspend();
    // console.log("AudioWorkletNode disconnected");
    // node.disconnect();
    // context.close();
    // console.log("AudioContext closed");
  }
}

class OrpheusStreamer extends BaseStreamer {
  private _skip_prompt = false;
  private queue: QueueType;
  constructor(queue: QueueType) {
    super();
    this.queue = queue;
  }
  put(value: bigint[][]) {
    if (this._skip_prompt) {
      this._skip_prompt = false;
      return;
    }
    this.queue.push(value[0][0]);
  }
  end() {
    this.queue.push(/* Sentinel to indicate completion. */ null);
  }
}

export async function OrpheusModel({
  model_name,
  dtype,
  device,
}: {
  model_name: string;
  dtype?: SupportedDtype;
  device?: SupportedDevice;
}) {
  const tokenizer = await AutoTokenizer.from_pretrained(model_name);
  const llm = // null;
    await AutoModelForCausalLM.from_pretrained(model_name, {
      dtype,
      device,
    });

  const snac_model_id = "onnx-community/snac_24khz-ONNX";
  const snac_decoder = await SnacDecoderModel.from_pretrained(snac_model_id, {
    dtype: "fp32", // NB: Full-precision
    device,
  });

  function _format_prompt({ prompt, voice = "tara" }: GenerateSpeechOptions) {
    if (voice) {
      prompt = `${voice}: ${prompt}`;
    }
    return `<custom_token_3>${prompt}<|eot_id|><custom_token_4><custom_token_5><custom_token_1>`;
  }

  function redistribute_codes(codes: number[], codebook_size: number) {
    if (FRAME_SIZE !== 7) {
      throw new Error(`Expected FRAME_SIZE to be 7, but got ${FRAME_SIZE}`);
    }
    const layer1: number[] = [];
    const layer2: number[] = [];
    const layer3: number[] = [];
    const layers = [layer1, layer2, layer3, layer3, layer2, layer3, layer3];
    for (let i = 0; i < codes.length; i += FRAME_SIZE) {
      for (let j = 0; j < FRAME_SIZE; ++j) {
        layers[j].push(codes[i + j] - j * codebook_size);
      }
    }
    return [layer1, layer2, layer3];
  }

  async function* generate_tokens_async({
    prompt,
    voice = "tara",
    max_tokens = 1200,
    temperature = 0.6,
    repetition_penalty = 1.3,
    top_k = 50,
    do_sample = true,
    stop_token_ids = [128258],
    // TODO: Implement the following parameters
    // top_p=0.8,
  }: GenerateSpeechOptions & {
    max_tokens?: number;
    temperature?: number;
    repetition_penalty?: number;
    top_k?: number;
    do_sample?: boolean;
    stop_token_ids?: number[];
  }) {
    const prompt_string = _format_prompt({ prompt, voice });
    const inputs = tokenizer(prompt_string);

    // breaks with:
    // const tokens = [128000, 128556, 135321, 138033, 144122, 145829, 152326, 156371].map(BigInt);
    // const tokens = [128000, 128556, 135321, 138033, 144122, 145829, 152326, 156371, 129243, 135573, 137158, 144432, 147095, 151924, 154673, 130929, 133382, 139507, 143500, 146638, 150913, 153607, 129214, 132827, 139739, 140758, 147587, 150936, 154152, 129411, 132395, 139127, 143222, 145115, 151423, 153755, 132028, 134350, 139947, 142660, 146723, 151495, 154993, 128542, 135547, 137803, 144505, 147302, 150058, 154460, 131220, 134065, 136578, 141886, 147031, 150054, 156289, 132262, 134894, 140054, 141327, 146645, 149819, 154300, 131175, 133315, 137029, 142885, 145936, 148940, 153288, 129603, 135555, 139287, 143656, 145787, 150375, 155626, 131607, 133070, 137309, 141833, 146944, 152570, 153761, 129379, 134291, 138102, 140654, 144757, 152280, 156162, 130860, 132786, 137297, 143014, 146932, 151557, 156079, 132216, 135389, 136977, 143972, 146087, 152318, 153196, 128669, 133786, 140190, 141739, 147677, 151506, 156800, 130763, 135025, 140016, 140856, 148006, 150862, 154636, 130225, 133799, 140163, 143149, 146062, 150397, 155620, 132068, 136258, 137782, 142362, 145393, 151907, 153877, 129619, 133750, 139002, 144615, 146390, 150666, 154801, 128381, 133168, 137409, 143058, 148705, 149024, 153558, 131181, 136148, 136465, 143704, 146245, 151666, 153030, 130315, 135227, 139526, 142244, 147698, 150777, 155564, 128446, 135167, 137483, 142720, 146262, 150319, 153523, 130839, 136420, 138381, 144445, 148656, 148842, 155902, 129804, 135268, 139325, 143421, 144752, 151159, 155962, 130014, 132791, 138871, 140922, 148322, 149060, 154962, 128285, 136342, 138775, 141128, 148068, 150426, 153802, 128258].map(BigInt);
    // for (const token of tokens) {
    //   yield token;
    //   await new Promise(resolve => setTimeout(resolve, 5));
    // }
    // return tokens;

    // return;
    const queue = new AsyncQueue<bigint | null>();
    const streamer = new OrpheusStreamer(queue);
    const generated_ids = llm.generate({
      ...inputs,
      max_new_tokens: max_tokens,
      repetition_penalty,
      eos_token_id: stop_token_ids,
      temperature,
      top_k,
      do_sample,

      // Used to support streaming
      streamer,
    }) as Promise<Tensor>;

    const temp = [];
    while (true) {
      const value = await queue.shift();
      if (value === null) break;
      temp.push(value);
      yield value;
    }
    // console.log(temp.length, JSON.stringify(temp.map((t) => Number(t))));
    return await generated_ids;
  }

  async function convert_to_audio(multiframe: bigint[]) {
    if (multiframe.length < FRAME_SIZE) {
      return null;
    }

    const raw_codes = multiframe
      .slice(0, Math.floor(multiframe.length / FRAME_SIZE) * FRAME_SIZE)
      .map((t) => Number(t) - 128266);

    // @ts-expect-error ts(2339) Property 'codebook_size' does not exist on type 'PretrainedConfig'.
    const codebook_size = snac_decoder.config.codebook_size;

    const redistributed_codes = redistribute_codes(raw_codes, codebook_size);
    if (redistributed_codes.flat().some((v) => v < 0 || v >= codebook_size)) {
      return null;
    }

    const snac_model_inputs = Object.fromEntries(
      redistributed_codes.map((layer, i) => [
        `audio_codes.${i}`,
        new Tensor("int64", layer, [1, layer.length]),
      ]),
    );
    const { audio_values } = (await snac_decoder(snac_model_inputs)) as {
      audio_values: Tensor;
    };
    return audio_values.data as Float32Array;
  }

  return {
    async *generate_speech_stream({ prompt, voice }: GenerateSpeechOptions) {
      const buffer: bigint[] = [];
      let numProcessedTokens = 0;
      let prevAudioData: Float32Array | null = null;
      for await (const token of generate_tokens_async({
        prompt,
        voice,
      })) {
        if (token < 128266n) continue;
        buffer.push(token);

        if (buffer.length % (NUM_PROCESSING_FRAMES * FRAME_SIZE) === 0) {
          const audio = await convert_to_audio(buffer.slice(-BUFFER_SIZE));
          if (audio) {
            const chunk = new RawAudio(
              audio.slice(
                prevAudioData === null ? 0 : NUM_PADDING_FRAMES_LEFT * SAMPLES_PER_FRAME,
                -NUM_PADDING_FRAMES_RIGHT * SAMPLES_PER_FRAME,
              ),
              SAMPLING_RATE,
            );
            yield chunk;

            numProcessedTokens = buffer.length;
            prevAudioData = audio;
          }
        }
      }

      // Process any remaining tokens in the buffer
      if (numProcessedTokens >= buffer.length) {
        // We finished with exactly the right number of tokens, but since we overlap generations,
        // we still need to yield the end of the last audio chunk, if it exists.
        if (prevAudioData) {
          const chunk = new RawAudio(
            prevAudioData.slice(-NUM_PADDING_FRAMES_RIGHT * SAMPLES_PER_FRAME),
            SAMPLING_RATE,
          );
          yield chunk;
        }
      } else {
        // There are still tokens left in the buffer.
        const remaining = buffer.slice(
          numProcessedTokens -
            (prevAudioData ? NUM_PADDING_FRAMES_RIGHT * FRAME_SIZE : 0) - // Include the previous chunk's data if it exists
            NUM_PADDING_FRAMES_LEFT * FRAME_SIZE, // Add padding frames to avoid artifacts
        );
        const audio = await convert_to_audio(remaining);
        if (audio) {
          const chunk = new RawAudio(
            audio.slice(prevAudioData ? NUM_PADDING_FRAMES_LEFT * SAMPLES_PER_FRAME : 0),
            SAMPLING_RATE,
          );
          yield chunk;
        }
      }
    },
    generate_speech(options: GenerateSpeechOptions) {
      const stream = this.generate_speech_stream(options);
      return new AudioStream(stream);
    },
  };
}

export { RawAudio };
