import { jit, lax, nn, numpy as np, random, tree } from "@jax-js/jax";
import { safetensors, WeightMapper } from "@jax-js/loaders";

export type KVCache = {
  key: np.Array;
  value: np.Array;
};

export function emptyKVCache(): KVCache {
  return {
    key: np.zeros([0], { dtype: np.float16 }),
    value: np.zeros([0], { dtype: np.float16 }),
  };
}

export type PocketTTS = {
  flowLM: FlowLMModel;
  mimi: MimiModel;
};

export type FlowLMModel = {
  bosEmb: np.Array;
  conditionerEmbed: np.Array;
  embMean: np.Array;
  embStd: np.Array;
  flowNet: SimpleMLPAdaLN;
  inputLinear: Linear;
  outNorm: LayerNorm;
  outEos: Linear;
  speakerProjWeight: np.Array;
  transformer: StreamingTransformerLayer[];
};

export type FlowLMState = {
  kvCaches: KVCache[];
  kvCacheLen: number;
};

export function createFlowLMState(model: FlowLMModel): FlowLMState {
  return {
    kvCaches: model.transformer.map(() => emptyKVCache()),
    kvCacheLen: 0,
  };
}

export function runFlowLMStep(
  {
    bosEmb,
    conditionerEmbed,
    embMean,
    embStd,
    flowNet,
    inputLinear,
    outNorm,
    outEos,
    speakerProjWeight,
    transformer,
  }: FlowLMModel,
  { kvCaches, kvCacheLen }: FlowLMState,
  key: np.Array,
  sequence: np.Array,
  embeds: np.Array | null,
  offset: number,
  lsdDecodeSteps: number = 1,
  temperature: number = 0.7,
  noiseClamp: number | null = null,
  eosThreshold: number = -4.0,
): { latent: np.Array; isEos: np.Array; state: FlowLMState } {
  bosEmb.dispose();
  conditionerEmbed.dispose();
  embMean.dispose();
  embStd.dispose();
  speakerProjWeight.dispose();

  const ldim = bosEmb.shape[0];

  let input = runLinear(inputLinear, sequence);

  if (embeds !== null) input = np.concatenate([embeds, input], 0);

  for (let i = 0; i < transformer.length; i++) {
    if (kvCacheLen > 0 && kvCaches[i].key.shape[0] === kvCacheLen) {
      const newCapacity = Math.ceil((kvCacheLen + 1) / 64) * 64;
      kvCaches[i].key = np.pad(kvCaches[i].key, {
        0: [0, newCapacity - kvCacheLen],
      });
      kvCaches[i].value = np.pad(kvCaches[i].value, {
        0: [0, newCapacity - kvCacheLen],
      });
    }
    const layer = transformer[i];
    [input, kvCaches[i]] = runStreamingTransformerLayer(
      layer,
      kvCaches[i],
      input,
      offset,
      kvCacheLen,
      { numHeads: 16 },
    );
  }
  kvCacheLen += input.shape[0];

  let transformerOut = runLayerNorm(outNorm, input);

  transformerOut = transformerOut.slice([-1]);

  const eosLogit = runLinear(outEos, transformerOut.ref);
  const isEos = np.greater(eosLogit, eosThreshold);

  const noiseShape = [1, ldim];
  const std = Math.sqrt(temperature);
  let noise = random.normal(key, noiseShape).mul(std);
  if (noiseClamp !== null) {
    noise = np.clip(noise, -noiseClamp, noiseClamp);
  }

  const conditionedFlow = (s: np.Array, t: np.Array, x: np.Array) =>
    runSimpleMLPAdaLN(tree.ref(flowNet), transformerOut.ref, s, t, x);
  const latent = lsdDecode(conditionedFlow, noise, lsdDecodeSteps);
  tree.dispose([flowNet, transformerOut]);

  return { latent, isEos, state: { kvCaches, kvCacheLen } };
}

export type SimpleMLPAdaLN = {
  timeEmbed: TimestepEmbedder[];
  condEmbed: Linear;
  inputProj: Linear;
  resBlocks: ResBlock[];
  finalLayer: {
    linear: Linear;
    adaLNModulation: [undefined, Linear];
  };
};

export const runSimpleMLPAdaLN = jit(function runSimpleMLPAdaLN(
  { timeEmbed, condEmbed, inputProj, resBlocks, finalLayer }: SimpleMLPAdaLN,
  c: np.Array,
  s: np.Array,
  t: np.Array,
  x: np.Array,
): np.Array {
  x = runLinear(inputProj, x);

  const sEmb = runTimestepEmbedder(timeEmbed[0], s);
  const tEmb = runTimestepEmbedder(timeEmbed[1], t);
  const tCombined = sEmb.add(tEmb).div(2);

  const cEmb = runLinear(condEmbed, c);
  const y = tCombined.add(cEmb);

  for (const block of resBlocks) {
    x = runResBlock(block, x, y.ref);
  }

  const [, finalAdaLNLinear] = finalLayer.adaLNModulation;
  const finalMod = runLinear(finalAdaLNLinear, nn.silu(y));
  const [shift, scale] = np.split(finalMod, 2, -1);

  x = runLayerNorm({}, x, 1e-6);
  x = modulate(x, shift, scale);
  x = runLinear(finalLayer.linear, x);

  return x;
});

export function runRope(
  q: np.Array,
  k: np.Array,
  offset: np.Array,
  maxPeriod: number = 10000,
): [np.Array, np.Array] {
  const [T, H, D] = q.shape;
  const halfD = D / 2;

  const ds = np.arange(halfD, undefined, undefined, { dtype: np.float32 });
  const freqs = np.exp(ds.mul((-Math.log(maxPeriod) * 2) / D));

  const ts = np.arange(T).add(offset).astype(np.float32).reshape([T, 1, 1]);

  const qReshaped = q.reshape([T, H, halfD, 2]);
  const kReshaped = k.reshape([T, H, halfD, 2]);

  let [qr, qi] = np.split(qReshaped, 2, -1);
  let [kr, ki] = np.split(kReshaped, 2, -1);
  qr = np.squeeze(qr, -1);
  qi = np.squeeze(qi, -1);
  kr = np.squeeze(kr, -1);
  ki = np.squeeze(ki, -1);

  const angles = freqs.mul(ts);
  const rotr = np.cos(angles.ref).astype(qr.dtype);
  const roti = np.sin(angles).astype(qr.dtype);

  const qor = qr.ref.mul(rotr.ref).sub(qi.ref.mul(roti.ref));
  const qoi = qr.mul(roti.ref).add(qi.mul(rotr.ref));
  const kor = kr.ref.mul(rotr.ref).sub(ki.ref.mul(roti.ref));
  const koi = kr.mul(roti).add(ki.mul(rotr));

  const qo = np.stack([qor, qoi], -1).reshape([T, H, D]);
  const ko = np.stack([kor, koi], -1).reshape([T, H, D]);

  return [qo, ko];
}

export type MimiStreamingMultiheadAttention = {
  outProj: Linear;
  inProj: Linear;
};

export function runMimiStreamingMultiheadAttention(
  { inProj, outProj }: MimiStreamingMultiheadAttention,
  kvCache: KVCache,
  query: np.Array,
  offset: np.Array,
  kvCacheLen: np.Array,
  context: number,
  numHeads: number,
  maxPeriod: number = 10000,
): [np.Array, KVCache] {
  const [T, embedDim] = query.shape;
  const headDim = embedDim / numHeads;

  const projected = runLinear(inProj, query);
  const qkv = projected.reshape([T, 3 * numHeads, headDim]);
  const [q_, k_, v] = np.split(qkv, 3, 1);
  const [q, k] = runRope(q_, k_, offset, maxPeriod);

  const isPrefill = kvCache.key.size === 0;
  let x: np.Array;
  if (isPrefill) {
    tree.dispose([kvCache, kvCacheLen]);
    x = nn.dotProductAttention(q, k.ref, v.ref, {
      isCausal: true,
      localWindowSize: context ? [context - 1, 0] : undefined,
    });
    kvCache = { key: k, value: v };
  } else {
    const capacity = kvCache.key.shape[0];
    const cacheMask = np
      .arange(capacity)
      .reshape([-1, 1, 1])
      .less(kvCacheLen.ref);
    kvCache.key = np.where(
      cacheMask.ref,
      kvCache.key,
      np.tile(k, [capacity / T, 1, 1]),
    );
    kvCache.value = np.where(
      cacheMask,
      kvCache.value,
      np.tile(v, [capacity / T, 1, 1]),
    );
    const maskDelta = np
      .arange(capacity)
      .sub(np.arange(T).reshape([T, 1]))
      .sub(kvCacheLen);
    const mask = context
      ? maskDelta.ref.lessEqual(0).mul(maskDelta.greater(-context))
      : maskDelta.lessEqual(0);
    x = nn.dotProductAttention(q, kvCache.key.ref, kvCache.value.ref, { mask });
  }
  x = x.reshape([T, embedDim]);
  x = runLinear(outProj, x);
  return [x, kvCache];
}

export type StreamingTransformerLayer = {
  selfAttn: MimiStreamingMultiheadAttention;
  norm1: LayerNorm;
  norm2: LayerNorm;
  linear1: Linear;
  linear2: Linear;
  layerScale1?: np.Array;
  layerScale2?: np.Array;
};

export const runStreamingTransformerLayer = jit(
  function runStreamingTransformerLayer(
    {
      selfAttn,
      norm1,
      norm2,
      linear1,
      linear2,
      layerScale1,
      layerScale2,
    }: StreamingTransformerLayer,
    kvCache: KVCache,
    x: np.Array,
    offset: np.Array,
    kvCacheLen: np.Array,
    {
      context = 0,
      numHeads,
      maxPeriod = 10000,
    }: { context?: number; numHeads: number; maxPeriod?: number },
  ): [np.Array, KVCache] {
    const xOrig = x.ref;
    x = runLayerNorm(norm1, x);
    let update: np.Array;
    [update, kvCache] = runMimiStreamingMultiheadAttention(
      selfAttn,
      kvCache,
      x,
      offset,
      kvCacheLen,
      context,
      numHeads,
      maxPeriod,
    );
    if (layerScale1) {
      update = update.mul(layerScale1);
    }
    x = xOrig.add(update);

    const xOrig2 = x.ref;
    x = runLayerNorm(norm2, x);
    let ffnOut = runLinear(linear1, x);
    ffnOut = nn.gelu(ffnOut, { approximate: false });
    ffnOut = runLinear(linear2, ffnOut);
    if (layerScale2) {
      ffnOut = ffnOut.mul(layerScale2);
    }
    x = xOrig2.add(ffnOut);

    return [x, kvCache];
  },
  { staticArgnums: [5, 6, 7] },
);

export type SEANetResnetBlock = {
  block: [undefined, StreamingConv1d, undefined, StreamingConv1d];
};

export function runSEANetResnetBlock(
  { block }: SEANetResnetBlock,
  states: (np.Array | null)[],
  x: np.Array,
): [np.Array, np.Array[]] {
  let v = x.ref;
  let stateIdx = 0;
  for (const layer of block) {
    if (layer === undefined) {
      v = nn.elu(v);
    } else {
      [v, states[stateIdx]] = runConv1d(layer.conv, states[stateIdx], v);
      stateIdx++;
    }
  }
  return [x.add(v), states as np.Array[]];
}

export type SEANetEncoder = {
  model: [
    StreamingConv1d,
    SEANetResnetBlock,
    undefined,
    StreamingConv1d,
    SEANetResnetBlock,
    undefined,
    StreamingConv1d,
    SEANetResnetBlock,
    undefined,
    StreamingConv1d,
    undefined,
    StreamingConv1d,
  ];
};

export function runSEANetEncoder(
  { model }: SEANetEncoder,
  x: np.Array,
): np.Array {
  const ratios = [4, 5, 6];

  x = np.expandDims(x, 0);
  [x] = runConv1d(model[0].conv, null, x);

  let idx = 1;
  for (let i = 0; i < 3; i++) {
    let states: any = [null, null];
    [x, states] = runSEANetResnetBlock(
      model[idx] as SEANetResnetBlock,
      states,
      x,
    );
    tree.dispose(states);
    idx++;
    x = nn.elu(x);
    idx++;
    const stride = ratios[i];
    [x] = runConv1d((model[idx] as StreamingConv1d).conv, null, x, stride);
    idx++;
  }

  x = nn.elu(x);
  [x] = runConv1d((model[11] as StreamingConv1d).conv, null, x);

  return x.slice(0);
}

export type SEANetDecoder = {
  model: [
    StreamingConv1d,
    undefined,
    StreamingConvTranspose1d,
    SEANetResnetBlock,
    undefined,
    StreamingConvTranspose1d,
    SEANetResnetBlock,
    undefined,
    StreamingConvTranspose1d,
    SEANetResnetBlock,
    undefined,
    StreamingConv1d,
  ];
};

export function createSEANetDecoderState({
  model,
}: SEANetDecoder): SEANetDecoderState {
  return {
    conv1: createConv1dState(model[0].conv),
    blocks: [
      {
        convtr: createConvTranspose1dState(model[2].convtr, 6),
        res: [
          createConv1dState(model[3].block[1].conv),
          createConv1dState(model[3].block[3].conv),
        ],
      },
      {
        convtr: createConvTranspose1dState(model[5].convtr, 5),
        res: [
          createConv1dState(model[6].block[1].conv),
          createConv1dState(model[6].block[3].conv),
        ],
      },
      {
        convtr: createConvTranspose1dState(model[8].convtr, 4),
        res: [
          createConv1dState(model[9].block[1].conv),
          createConv1dState(model[9].block[3].conv),
        ],
      },
    ],
    conv2: createConv1dState(model[11].conv),
  };
}

export const runSEANetDecoder = jit(function runSEANetDecoder(
  { model }: SEANetDecoder,
  state: SEANetDecoderState,
  x: np.Array,
): [np.Array, SEANetDecoderState] {
  const ratios = [6, 5, 4];

  x = np.expandDims(x, 0);
  [x, state.conv1] = runConv1d(model[0].conv, state.conv1, x);

  let idx = 1;
  for (let i = 0; i < 3; i++) {
    const blockState = state.blocks[i];
    x = nn.elu(x);
    idx++;
    const stride = ratios[i];
    [x, blockState.convtr] = runConvTranspose1d(
      (model[idx] as StreamingConvTranspose1d).convtr,
      blockState.convtr,
      x,
      stride,
    );
    idx++;
    [x, blockState.res] = runSEANetResnetBlock(
      model[idx] as SEANetResnetBlock,
      blockState.res,
      x,
    );
    idx++;
  }

  x = nn.elu(x);
  [x, state.conv2] = runConv1d(model[11].conv, state.conv2, x);

  return [x.slice(0), state];
});

export type MimiModel = {
  encoder: SEANetEncoder;
  decoder: SEANetDecoder;
  encoderTransformer: StreamingTransformerLayer[];
  decoderTransformer: StreamingTransformerLayer[];
  quantizer: {
    outputProj: { weight: np.Array };
  };
  downsample: StreamingConv1d;
  upsample: StreamingConvTranspose1d;
};

export function runMimiEncode(
  {
    encoder,
    encoderTransformer,
    decoder,
    decoderTransformer,
    quantizer,
    downsample,
    upsample,
  }: MimiModel,
  x: np.Array,
): np.Array {
  tree.dispose([decoder, decoderTransformer, quantizer, upsample]);
  x = runSEANetEncoder(encoder, x);

  x = x.transpose([1, 0]);
  const offset = np.array(0, { dtype: np.int32, device: x.device });
  for (const layer of encoderTransformer) {
    let kvCache = emptyKVCache();
    [x, kvCache] = runStreamingTransformerLayer(
      layer,
      kvCache,
      x,
      offset.ref,
      0,
      { context: 250, numHeads: 8 },
    );
    tree.dispose(kvCache);
  }
  offset.dispose();
  x = x.transpose([1, 0]);

  [x] = runConv1d(downsample.conv, null, x, 16);
  return x;
}

export type MimiDecodeState = {
  kvCaches: KVCache[];
  kvCacheLen: number;
  initialConvState: np.Array;
  seanetStates: SEANetDecoderState;
};

export type SEANetDecoderState = {
  conv1: np.Array;
  blocks: {
    convtr: np.Array;
    res: np.Array[];
  }[];
  conv2: np.Array;
};

export function createMimiDecodeState(mimi: MimiModel): MimiDecodeState {
  return {
    kvCaches: mimi.decoderTransformer.map(() => emptyKVCache()),
    kvCacheLen: 0,
    initialConvState: createConvTranspose1dState(mimi.upsample.convtr, 16, 512),
    seanetStates: createSEANetDecoderState(mimi.decoder),
  };
}

export function runMimiDecode(
  {
    encoder,
    encoderTransformer,
    decoder,
    decoderTransformer,
    quantizer,
    downsample,
    upsample,
  }: MimiModel,
  { kvCaches, kvCacheLen, initialConvState, seanetStates }: MimiDecodeState,
  latent: np.Array,
  offset: number,
): [np.Array, MimiDecodeState] {
  tree.dispose([encoder, encoderTransformer, downsample]);

  latent = np.expandDims(latent.transpose([1, 0]), 0);
  latent = lax.conv(latent, quantizer.outputProj.weight, [1], "VALID");

  let x: np.Array;
  [x, initialConvState] = runConvTranspose1d(
    upsample.convtr,
    initialConvState,
    latent,
    16,
    latent.shape[1],
  );
  x = x.slice(0);

  x = x.transpose([1, 0]);
  for (let i = 0; i < decoderTransformer.length; i++) {
    const layer = decoderTransformer[i];
    [x, kvCaches[i]] = runStreamingTransformerLayer(
      layer,
      kvCaches[i],
      x,
      offset * 16,
      kvCacheLen,
      { context: 250, numHeads: 8 },
    );
  }
  x = x.transpose([1, 0]);

  [x, seanetStates] = runSEANetDecoder(decoder, seanetStates, x);

  kvCacheLen += 16;
  if (kvCaches[0].key.shape[0] !== 272) {
    const padAmount = 272 - kvCaches[0].key.shape[0];
    for (const c of kvCaches) {
      c.key = np.pad(c.key, { 0: [0, padAmount] });
      c.value = np.pad(c.value, { 0: [0, padAmount] });
    }
  }
  if (kvCacheLen === 272) {
    kvCacheLen -= 16;
    for (const c of kvCaches) {
      c.key = np.pad(c.key.slice([16]), { 0: [0, 16] });
      c.value = np.pad(c.value.slice([16]), { 0: [0, 16] });
    }
  }

  return [
    x,
    {
      kvCaches,
      kvCacheLen: kvCacheLen,
      initialConvState,
      seanetStates,
    },
  ];
}

export function lsdDecode(
  flowNet: (s: np.Array, t: np.Array, x: np.Array) => np.Array,
  x0: np.Array,
  numSteps: number = 1,
): np.Array {
  let current = x0;
  for (let i = 0; i < numSteps; i++) {
    const s = i / numSteps;
    const t = (i + 1) / numSteps;
    const sArr = np.full(x0.shape.slice(0, -1).concat([1]), s);
    const tArr = np.full(x0.shape.slice(0, -1).concat([1]), t);
    const flowDir = flowNet(sArr, tArr, current.ref);
    current = current.add(flowDir.div(numSteps));
  }
  return current;
}

export type TimestepEmbedder = {
  mlp: [Linear, undefined, Linear, RMSNorm];
  freqs: np.Array;
};

export function runTimestepEmbedder(
  { mlp, freqs }: TimestepEmbedder,
  t: np.Array,
): np.Array {
  const [linear1, , linear2, rmsNorm] = mlp;
  const args = t.mul(freqs);
  const embedding = np.concatenate([np.cos(args.ref), np.sin(args)], -1);
  let x = runLinear(linear1, embedding);
  x = nn.silu(x);
  x = runLinear(linear2, x);
  x = runRMSNorm(rmsNorm, x);
  return x;
}

function modulate(x: np.Array, shift: np.Array, scale: np.Array): np.Array {
  return x.mul(scale.add(1)).add(shift);
}

export type ResBlock = {
  inLN: LayerNorm;
  mlp: [Linear, undefined, Linear];
  adaLNModulation: [undefined, Linear];
};

export function runResBlock(
  { inLN, mlp, adaLNModulation }: ResBlock,
  x: np.Array,
  y: np.Array,
): np.Array {
  const [, adaLNLinear] = adaLNModulation;
  const modulation = runLinear(adaLNLinear, nn.silu(y));
  const [shiftMlp, scaleMlp, gateMlp] = np.split(modulation, 3, -1);

  let h = runLayerNorm(inLN, x.ref, 1e-6);
  h = modulate(h, shiftMlp, scaleMlp);

  const [mlpLinear1, , mlpLinear2] = mlp;
  h = runLinear(mlpLinear1, h);
  h = nn.silu(h);
  h = runLinear(mlpLinear2, h);

  return x.add(gateMlp.mul(h));
}

export type Linear = {
  weight: np.Array;
  bias?: np.Array;
};

export function runLinear({ weight, bias }: Linear, x: np.Array): np.Array {
  x = np.dot(x, weight.transpose());
  if (bias) x = x.add(bias);
  return x;
}

export type LayerNorm = {
  weight: np.Array;
  bias: np.Array;
};

export const runLayerNorm = jit(
  function runLayerNorm(
    { weight, bias }: Partial<LayerNorm> = {},
    x: np.Array,
    eps: number = 1e-5,
  ) {
    const dtype = x.dtype;
    x = x.astype(np.float32);
    const mean = x.ref.mean(-1, { keepdims: true });
    const var_ = np.var_(x.ref, -1, {
      mean: mean.ref,
      correction: 0,
      keepdims: true,
    });
    x = x.sub(mean).div(np.sqrt(var_.add(eps)));
    if (weight) {
      x = x.mul(weight).add(bias!);
    }
    return x.astype(dtype);
  },
  { staticArgnums: [2] },
);

export type RMSNorm = {
  alpha: np.Array;
};

export function runRMSNorm(
  { alpha }: RMSNorm,
  x: np.Array,
  eps: number = 1e-5,
) {
  const dtype = x.dtype;
  x = x.astype(np.float32);
  const var_ = np.var_(x.ref, -1, { correction: 0, keepdims: true });
  x = x.mul(alpha).div(np.sqrt(var_.add(eps)));
  return x.astype(dtype);
}

export type Conv1d = {
  weight: np.Array;
  bias?: np.Array;
};

export function createConv1dState(
  { weight }: Conv1d,
  stride: number = 1,
): np.Array {
  return np.zeros(
    [
      1,
      weight.shape[1],
      weight.shape[2] - stride,
    ],
    { dtype: np.float16 },
  );
}

export function runConv1d(
  { weight, bias }: Conv1d,
  state: np.Array | null,
  x: np.Array,
  stride: number = 1,
): [np.Array, np.Array] {
  state ??= createConv1dState({ weight }, stride);
  x = np.concatenate([state, x], 2);
  state = x.ref.slice([], [], [x.shape[2] - state.shape[2]]);
  let y = lax.conv(x, weight, [stride], "VALID");
  if (bias) y = y.add(np.expandDims(bias, -1));
  return [y, state];
}

export type ConvTranspose1d = {
  weight: np.Array;
  bias?: np.Array;
};

export function createConvTranspose1dState(
  { weight }: ConvTranspose1d,
  stride: number = 1,
  groups: number = 1,
): np.Array {
  return np.zeros(
    [
      1,
      weight.shape[1] * groups,
      weight.shape[2] - stride,
    ],
    { dtype: np.float16 },
  );
}

export function runConvTranspose1d(
  { weight, bias }: ConvTranspose1d,
  state: np.Array | null,
  x: np.Array,
  stride: number = 1,
  groups: number = 1,
): [np.Array, np.Array] {
  state ??= createConvTranspose1dState({ weight }, stride);
  const [cIn, cOut, kernelSize] = weight.shape;
  weight = np.flip(weight, -1);
  if (groups > 1) {
    weight = weight
      .reshape([groups, cIn / groups, cOut, kernelSize])
      .transpose([0, 2, 1, 3])
      .reshape([cOut * groups, cIn / groups, kernelSize]);
  } else {
    weight = weight.transpose([1, 0, 2]);
  }

  let y = lax.convGeneralDilated(
    x,
    weight,
    [1],
    [[kernelSize - 1, kernelSize - 1]],
    {
      lhsDilation: [stride],
      featureGroupCount: groups,
    },
  );
  y = y.add(np.pad(state, { 2: [0, y.shape[2] - state.shape[2]] }));
  [y, state] = np.split(y, [y.shape[2] - state.shape[2]], 2);
  if (bias) y = y.add(np.expandDims(bias, -1));
  return [y, state];
}

export type StreamingConv1d = {
  conv: Conv1d;
};

export type StreamingConvTranspose1d = {
  convtr: ConvTranspose1d;
};

const weightMapper = new WeightMapper({
  prefix: {
    "flow_lm.": "flowLM.",
    "mimi.decoder_transformer.transformer.layers": "mimi.decoderTransformer",
    "mimi.encoder_transformer.transformer.layers": "mimi.encoderTransformer",
  },
  suffix: {
    ".conditioner.embed.weight": ".conditionerEmbed",
    ".layer_scale_1.scale": ".layerScale1",
    ".layer_scale_2.scale": ".layerScale2",
  },
  substring: {
    ".conv.conv.": ".conv.",
    ".convtr.convtr.": ".convtr.",
    ".in_ln.": ".inLN.",
    ".transformer.layers.": ".transformer.",
  },
  autoCamelCase: true,
});

export function fromSafetensors(file: safetensors.File): PocketTTS {
  const mappedWeights = weightMapper.mapObject(file.tensors);
  const hydrated: Record<string, np.Array> = {};
  for (const [key, value] of Object.entries(mappedWeights)) {
    if (value.dtype === "F16") {
      hydrated[key] = np.array(value.data as Uint16Array, {
        dtype: np.float16,
        shape: value.shape,
      });
    } else {
      throw new Error(`Unexpected dtype ${value.dtype} for weight ${key}`);
    }
  }
  return safetensors.toNested(hydrated);
}
