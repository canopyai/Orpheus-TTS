export default () => {
  class BufferedAudioWorkletProcessor extends AudioWorkletProcessor {
    private bufferQueue: Float32Array[] = [];
    private currentChunkOffset: number = 0;

    constructor() {
      super();

      this.port.onmessage = (event: MessageEvent) => {
        const data = event.data;
        if (data instanceof Float32Array) {
          this.bufferQueue.push(data);
        }
      };
    }

    process(
      inputs: Float32Array[][],
      outputs: Float32Array[][],
      parameters: Record<string, Float32Array>,
    ): boolean {
      const channel = outputs[0][0];
      if (!channel) return true;

      const numSamples = channel.length;
      let outputIndex = 0;

      while (outputIndex < numSamples) {
        if (this.bufferQueue.length > 0) {
          const currentChunk = this.bufferQueue[0];
          const remainingSamples = currentChunk.length - this.currentChunkOffset;
          const samplesToCopy = Math.min(remainingSamples, numSamples - outputIndex);

          channel.set(
            currentChunk.subarray(this.currentChunkOffset, this.currentChunkOffset + samplesToCopy),
            outputIndex,
          );

          this.currentChunkOffset += samplesToCopy;
          outputIndex += samplesToCopy;

          // Remove the chunk if fully consumed.
          if (this.currentChunkOffset >= currentChunk.length) {
            this.bufferQueue.shift();
            this.currentChunkOffset = 0;
          }
        } else {
          // If no data is available, fill the rest of the buffer with silence.
          channel.fill(0, outputIndex);
          outputIndex = numSamples;
        }
      }
      return true;
    }
  }

  registerProcessor("buffered-audio-worklet-processor", BufferedAudioWorkletProcessor);
};
