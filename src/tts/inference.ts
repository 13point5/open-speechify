import { numpy as np, random, tree } from "@jax-js/jax";

import type { AudioPlayer } from "./audio";
import {
  createFlowLMState,
  createMimiDecodeState,
  type PocketTTS,
  runFlowLMStep,
  runMimiDecode,
} from "./pocket-tts";

export interface PlayTTSOptions {
  framesAfterEos: number;
  seed: number | null;
  lsdDecodeSteps: number;
  temperature: number;
  noiseClamp: number | null;
}

export async function playTTS(
  player: AudioPlayer,
  model: PocketTTS,
  embeds: np.Array,
  {
    framesAfterEos = 0,
    seed = null,
    lsdDecodeSteps = 1,
    temperature = 0.7,
    noiseClamp = null,
  }: Partial<PlayTTSOptions> = {},
): Promise<void> {
  let sequence = model.flowLM.bosEmb.ref.reshape([1, -1]);
  let audioPromise: Promise<void> = Promise.resolve();

  if (seed === null) seed = Math.floor(Math.random() * 2 ** 32);
  let key = random.key(seed);

  try {
    let flowLMState = createFlowLMState(model.flowLM);
    let mimiState = createMimiDecodeState(model.mimi);
    let eosStep: number | null = null;

    for (let step = 0; step < 1000; step++) {
      let stepKey: np.Array;
      [key, stepKey] = random.split(key);
      const {
        latent,
        isEos,
        state: newFlowLMState,
      } = runFlowLMStep(
        tree.ref(model.flowLM),
        flowLMState,
        stepKey,
        step === 0 ? sequence.ref : sequence.ref.slice([-1]),
        step === 0 ? embeds.ref : null,
        flowLMState.kvCacheLen,
        lsdDecodeSteps,
        temperature,
        noiseClamp,
      );
      flowLMState = newFlowLMState;

      const isEosData = await isEos.data();
      if (isEosData[0] && eosStep === null) {
        eosStep = step;
      }
      if (eosStep !== null && step >= eosStep + framesAfterEos) {
        latent.dispose();
        break;
      }

      sequence = np.concatenate([sequence, latent]);

      let mimiInput = sequence.ref.slice([-1]);
      mimiInput = mimiInput
        .mul(model.flowLM.embStd.ref)
        .add(model.flowLM.embMean.ref);

      const [audio, newMimiState] = runMimiDecode(
        tree.ref(model.mimi),
        mimiState,
        mimiInput,
        step,
      );
      mimiState = newMimiState;

      const lastAudioPromise = audioPromise;
      audioPromise = (async () => {
        const audioPcm = (await np
          .clip(audio.slice(0), -1, 1)
          .astype(np.float32)
          .data()) as Float32Array;
        if (audioPcm.length !== 1920) {
          throw new Error(
            `expected 1920 audio samples, got ${audioPcm.length}`,
          );
        }
        await lastAudioPromise;
        await player.playChunk(audioPcm);
      })();
    }
  } finally {
    sequence.dispose();
    tree.dispose([embeds]);
    await audioPromise;
  }
}
