# NeuTTS Nano

English text-to-speech with instant voice cloning, powered by [neuphonic/neutts-nano](https://huggingface.co/neuphonic/neutts-nano).

Ultra-compact (~117M active params) on-device TTS that clones a speaker from a few seconds of reference audio and synthesizes natural English speech at 24 kHz.

## Examples

### Preset voice

**Input**
- `text`: `My name is Andy. I'm 25 and I just moved to London. The underground is pretty confusing, but it gets me around in no time at all.`
- `voice`: `jo`

**Output:** 24 kHz WAV speech in the Jo reference voice.

### Custom voice clone

**Input**
- `text`: `This is a quick demo of NeuTTS Nano voice cloning on Replicate.`
- `ref_audio`: a clean 3–15s mono WAV of the target speaker
- `ref_text`: exact transcript of that reference clip

**Output:** WAV speech matching the uploaded speaker.

## Inputs

| Name | Type | Default | Description |
|---|---|---|---|
| `text` | string | required | English text to synthesize |
| `voice` | choice | `jo` | Built-in reference: `jo`, `dave`, `emily`, `paul`, `sophie`, `steven` |
| `ref_audio` | file | optional | Custom 3–15s reference WAV (overrides `voice`) |
| `ref_text` | string | `""` | Transcript of `ref_audio` (required for custom cloning) |
| `temperature` | float | `1.0` | Sampling temperature |
| `top_k` | int | `50` | Top-k sampling cutoff |
| `seed` | int | `-1` | Seed (`-1` = random) |

## Tips

- Prefer clean, continuous speech references (mono, 16–44 kHz, 3–15 seconds).
- For best cloning quality, keep `ref_text` faithful to the reference audio.
- Outputs are watermarked by NeuTTS / Perth by default.

## Links

- Weights: https://huggingface.co/neuphonic/neutts-nano
- Codec: https://huggingface.co/neuphonic/neucodec
- Source: https://github.com/nghimestudio/neu-tts
- Upstream: https://github.com/neuphonic/neutts
