from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import os


model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

snac_device = os.environ.get("SNAC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
model = model.to(snac_device)


def convert_to_audio(multiframe, count):
  frames = []
  if len(multiframe) < 7:
    return
  
  codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
  codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
  codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)

  num_frames = len(multiframe) // 7
  frame = multiframe[:num_frames*7]

  for j in range(num_frames):
    i = 7*j
    if codes_0.shape[0] == 0:
      codes_0 = torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)
    else:
      codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)])

    if codes_1.shape[0] == 0:
      
      codes_1 = torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)
      codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
    else:
      codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)])
      codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
    
    if codes_2.shape[0] == 0:
      codes_2 = torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])
    else:
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])

  codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
  # check that all tokens are between 0 and 4096 otherwise return *
  if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
    return

  with torch.inference_mode():
    audio_hat = model.decode(codes)
  
  audio_slice = audio_hat[:, :, 2048:4096]
  detached_audio = audio_slice.detach().cpu()
  audio_np = detached_audio.numpy()
  audio_int16 = (audio_np * 32767).astype(np.int16)
  audio_bytes = audio_int16.tobytes()
  return audio_bytes

def turn_token_into_id(token_string, index):
    # Strip whitespace
    token_string = token_string.strip()
    
    # Find the last token in the string
    last_token_start = token_string.rfind("<custom_token_")
    
    if last_token_start == -1:
        print("No token found in the string")
        return None
    
    # Extract the last token
    last_token = token_string[last_token_start:]
    
    # Process the last token
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    else:
        return None
  
    
async def tokens_decoder(token_gen):
    buffer = []
    count = 0
    async for token_sim in token_gen:       
        token = turn_token_into_id(token_sim, count)
        if token is None:
            pass
        else:
            if token > 0:
                buffer.append(token)
                count += 1

                if count % 7 == 0 and count > 27:
                    buffer_to_proc = buffer[-28:]
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples


# ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
def tokens_decoder_sync(syn_token_gen):

    audio_queue = queue.Queue()

    # Convert the synchronous token generator into an async generator.
    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        # tokens_decoder.tokens_decoder is assumed to be an async generator that processes tokens.
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)  # Sentinel

    def run_async():
        asyncio.run(async_producer())

    thread = threading.Thread(target=run_async)
    thread.start()

    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        yield audio

    thread.join()