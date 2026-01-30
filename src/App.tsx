import { useCallback, useEffect, useRef, useState } from "react";
import { defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { cachedFetch, safetensors, tokenizers } from "@jax-js/loaders";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { createStreamingPlayer } from "@/tts/audio";
import { playTTS } from "@/tts/inference";
import { fromSafetensors, type PocketTTS } from "@/tts/pocket-tts";

const WEIGHTS_URL =
  "https://huggingface.co/ekzhang/jax-js-models/resolve/main/kyutai-pocket-tts_b6369a24-fp16.safetensors";
const HF_URL_PREFIX =
  "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/fbf8280";
const DEFAULT_VOICE_URL = `${HF_URL_PREFIX}/embeddings/azelma.safetensors`;
const TOKENIZER_URL = `${HF_URL_PREFIX}/tokenizer.model`;

type SafetensorsFile = ReturnType<typeof safetensors.parse>;

function prepareTextPrompt(text: string): [string, number] {
  let normalized = text.trim();
  if (normalized === "") throw new Error("Prompt cannot be empty");
  normalized = normalized.replace(/\s+/g, " ");
  const numberOfWords = normalized.split(" ").length;
  let framesAfterEosGuess = 3;
  if (numberOfWords <= 4) {
    framesAfterEosGuess = 5;
  }

  normalized = normalized.replace(/^(\p{Ll})/u, (c) => c.toLocaleUpperCase());
  if (/[\p{L}\p{N}]$/u.test(normalized)) {
    normalized = normalized + ".";
  }
  if (normalized.split(" ").length < 5) {
    normalized = " ".repeat(8) + normalized;
  }

  return [normalized, framesAfterEosGuess];
}

function App() {
  const [prompt, setPrompt] = useState(
    "The sun is shining, and the birds are singing.",
  );
  const [isPlaying, setIsPlaying] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [webgpuSupported, setWebgpuSupported] = useState<boolean | null>(null);

  const weightsRef = useRef<SafetensorsFile | null>(null);
  const modelRef = useRef<PocketTTS | null>(null);
  const tokenizerRef = useRef<tokenizers.Unigram | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function checkWebGpu() {
      if (!navigator.gpu) {
        if (!cancelled) setWebgpuSupported(false);
        return;
      }

      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!cancelled) setWebgpuSupported(Boolean(adapter));
      } catch (checkError) {
        if (!cancelled) {
          setWebgpuSupported(false);
          setError(
            checkError instanceof Error
              ? checkError.message
              : "WebGPU check failed.",
          );
        }
      }
    }

    void checkWebGpu();
    return () => {
      cancelled = true;
    };
  }, []);

  const downloadWeights = useCallback(async () => {
    if (weightsRef.current) return weightsRef.current;
    setStatus("Downloading model weights...");
    const data = await cachedFetch(WEIGHTS_URL);
    const parsed = safetensors.parse(data);
    weightsRef.current = parsed;
    return parsed;
  }, []);

  const getModel = useCallback(async () => {
    if (modelRef.current) return modelRef.current;
    setStatus("Loading model...");
    const weights = await downloadWeights();
    const model = fromSafetensors(weights);
    modelRef.current = model;
    return model;
  }, [downloadWeights]);

  const getTokenizer = useCallback(async () => {
    if (tokenizerRef.current) return tokenizerRef.current;
    setStatus("Loading tokenizer...");
    const tokenizer = await tokenizers.loadSentencePiece(TOKENIZER_URL);
    tokenizerRef.current = tokenizer;
    return tokenizer;
  }, []);

  const handlePlay = useCallback(async () => {
    if (isPlaying) return;
    setError(null);
    setIsPlaying(true);

    try {
      setStatus("Preparing WebGPU...");
      const devices = await init();
      if (!devices.includes("webgpu")) {
        setWebgpuSupported(false);
        setError("WebGPU is not available on this device.");
        return;
      }
      defaultDevice("webgpu");

      const model = await getModel();
      const tokenizer = await getTokenizer();

      setStatus("Preparing prompt...");
      const [text, framesAfterEos] = prepareTextPrompt(prompt);
      const tokens = tokenizer.encode(text);

      setStatus("Loading voice and synthesizing audio...");
      const audioPrompt = safetensors.parse(
        await cachedFetch(DEFAULT_VOICE_URL),
      ).tensors.audio_prompt;
      const voiceEmbed = np
        .array(audioPrompt.data as Float32Array, {
          shape: audioPrompt.shape,
          dtype: np.float32,
        })
        .slice(0)
        .astype(np.float16);

      const tokensAr = np.array(tokens, { dtype: np.uint32 });
      let embeds = model.flowLM.conditionerEmbed.ref.slice(tokensAr);
      embeds = np.concatenate([voiceEmbed, embeds]);

      const player = createStreamingPlayer();
      try {
        await playTTS(player, tree.ref(model), embeds, {
          framesAfterEos,
        });
      } finally {
        await player.close();
      }
    } catch (playError) {
      setError(
        playError instanceof Error ? playError.message : "Playback failed.",
      );
    } finally {
      setStatus(null);
      setIsPlaying(false);
    }
  }, [getModel, getTokenizer, isPlaying, prompt]);

  const playDisabled =
    isPlaying || prompt.trim() === "" || webgpuSupported === false;

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-6 py-12">
        <header className="space-y-2">
          <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
            Pocket TTS Test
          </p>
          <h1 className="text-3xl font-semibold tracking-tight text-slate-900">
            Epub voice preview
          </h1>
          <p className="text-sm text-muted-foreground">
            This is a minimal smoke test for the Kyutai Pocket TTS model in
            jax-js.
          </p>
        </header>

        {webgpuSupported === false && (
          <Alert variant="destructive">
            <AlertTitle>WebGPU unavailable</AlertTitle>
            <AlertDescription>
              This device does not expose WebGPU, so in-browser TTS will not
              run. Try a newer Chromebook or Chrome with WebGPU enabled.
            </AlertDescription>
          </Alert>
        )}

        {error && (
          <Alert variant="destructive">
            <AlertTitle>Playback error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <section className="rounded-xl border bg-white p-5 shadow-sm">
          <div className="space-y-3">
            <label className="text-sm font-medium text-slate-700">
              Sample text
            </label>
            <Textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              className="min-h-[160px] resize-none"
              placeholder="Enter a short paragraph to test audio output."
            />
          </div>
          <div className="mt-4 flex items-center gap-3">
            <Button onClick={handlePlay} disabled={playDisabled}>
              {isPlaying ? "Playing..." : "Play"}
            </Button>
            {status && <p className="text-sm text-muted-foreground">{status}</p>}
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
