import type { ChangeEvent } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ePub from "epubjs";
import { defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { cachedFetch, safetensors, tokenizers } from "@jax-js/loaders";
import { Pause, Play, UploadCloud } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import type { AudioPlayer } from "@/tts/audio";
import { createStreamingPlayer } from "@/tts/audio";
import { playTTS } from "@/tts/inference";
import { fromSafetensors, type PocketTTS } from "@/tts/pocket-tts";

const WEIGHTS_URL =
  "https://huggingface.co/ekzhang/jax-js-models/resolve/main/kyutai-pocket-tts_b6369a24-fp16.safetensors";
const HF_URL_PREFIX =
  "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/fbf8280";
const DEFAULT_VOICE_URL = `${HF_URL_PREFIX}/embeddings/azelma.safetensors`;
const TOKENIZER_URL = `${HF_URL_PREFIX}/tokenizer.model`;

const BLOCK_SELECTOR =
  "p,li,blockquote,figcaption,dd,dt,td,th,h1,h2,h3,h4,h5,h6,pre";
const HIGHLIGHT_STYLES = {
  fill: "#fde047",
  "fill-opacity": "0.35",
  "mix-blend-mode": "multiply",
};
const HOVER_STYLES = {
  fill: "#fef9c3",
  "fill-opacity": "0.35",
  "mix-blend-mode": "multiply",
};

type EpubRendition = {
  display: (target?: string) => Promise<void>;
  flow: (flowMode: string) => void;
  destroy: () => void;
  themes: {
    default: (theme: Record<string, Record<string, string>>) => void;
  };
  hooks?: {
    content?: {
      register: (callback: (contents: any) => void) => void;
    };
  };
  annotations?: {
    highlight: (
      cfiRange: string,
      data?: Record<string, unknown>,
      cb?: (event: unknown) => void,
      className?: string,
      styles?: Record<string, string>,
    ) => unknown;
    remove: (cfiRange: string, type?: string) => void;
  };
  on?: (event: string, handler: (...args: any[]) => void) => void;
};

type SentenceEntry = {
  id: string;
  text: string;
  cfiRange: string;
  sectionIndex: number;
  order: number;
};

type TocItem = {
  id?: string;
  href: string;
  label: string;
  subitems: TocItem[];
};

type TextMapEntry = {
  node: Text;
  start: number;
  end: number;
};

const segmenterCache = new Map<string, Intl.Segmenter>();

function getSegmenter(language: string | null) {
  if (typeof Intl === "undefined" || !("Segmenter" in Intl)) return null;
  const lang = language?.trim() || "en";
  if (!segmenterCache.has(lang)) {
    segmenterCache.set(lang, new Intl.Segmenter(lang, { granularity: "sentence" }));
  }
  return segmenterCache.get(lang) ?? null;
}

function splitIntoSentences(text: string, language: string | null) {
  if (!text) return [] as { text: string; start: number; end: number }[];
  const segmenter = getSegmenter(language);
  if (segmenter) {
    return Array.from(segmenter.segment(text)).map((segment) => ({
      text: segment.segment,
      start: segment.index,
      end: segment.index + segment.segment.length,
    }));
  }

  const matches = Array.from(text.matchAll(/[^.!?]+[.!?]+|[^.!?]+$/g));
  return matches.map((match) => ({
    text: match[0],
    start: match.index ?? 0,
    end: (match.index ?? 0) + match[0].length,
  }));
}

function normalizeSentence(text: string) {
  return text.replace(/\s+/g, " ").trim();
}

function trimOffsets(text: string, start: number, end: number) {
  let s = start;
  let e = end;
  while (s < e && /\s/.test(text[s])) s += 1;
  while (e > s && /\s/.test(text[e - 1])) e -= 1;
  return { start: s, end: e };
}

function buildTextMap(doc: Document, block: Element) {
  const walker = doc.createTreeWalker(block, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => {
      const parent = node.parentElement;
      if (!parent) return NodeFilter.FILTER_REJECT;
      if (parent.closest("script, style, pre, code, kbd, samp")) {
        return NodeFilter.FILTER_REJECT;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  const map: TextMapEntry[] = [];
  let fullText = "";
  let offset = 0;

  while (walker.nextNode()) {
    const node = walker.currentNode as Text;
    const value = node.nodeValue ?? "";
    if (value === "") continue;
    const start = offset;
    const end = offset + value.length;
    map.push({ node, start, end });
    fullText += value;
    offset = end;
  }

  return { fullText, map };
}

function positionForOffset(map: TextMapEntry[], offset: number) {
  for (const entry of map) {
    if (offset >= entry.start && offset <= entry.end) {
      return {
        node: entry.node,
        offset: Math.max(0, Math.min(entry.node.nodeValue?.length ?? 0, offset - entry.start)),
      };
    }
  }
  return null;
}

function rangeFromOffsets(doc: Document, map: TextMapEntry[], start: number, end: number) {
  const startPos = positionForOffset(map, start);
  const endPos = positionForOffset(map, end);
  if (!startPos || !endPos) return null;
  const range = doc.createRange();
  range.setStart(startPos.node, startPos.offset);
  range.setEnd(endPos.node, endPos.offset);
  return range;
}

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

export function EpubReader() {
  const viewerRef = useRef<HTMLDivElement | null>(null);
  const bookRef = useRef<any>(null);
  const renditionRef = useRef<EpubRendition | null>(null);

  const processedDocsRef = useRef<WeakSet<Document>>(new WeakSet());
  const sentencesBySectionRef = useRef<Map<number, SentenceEntry[]>>(new Map());
  const sentenceListRef = useRef<SentenceEntry[]>([]);
  const sentenceIndexRef = useRef<Map<string, number>>(new Map());
  const sentenceByIdRef = useRef<Map<string, SentenceEntry>>(new Map());
  const sentenceIdCounterRef = useRef(0);
  const activeHighlightRef = useRef<string | null>(null);
  const hoverHighlightRef = useRef<string | null>(null);
  const hoverSentenceIdRef = useRef<string | null>(null);
  const interactiveDocsRef = useRef<WeakSet<Document>>(new WeakSet());
  const currentSectionIndexRef = useRef<number | null>(null);

  const weightsRef = useRef<ReturnType<typeof safetensors.parse> | null>(null);
  const modelRef = useRef<PocketTTS | null>(null);
  const tokenizerRef = useRef<tokenizers.Unigram | null>(null);
  const voiceEmbedRef = useRef<np.Array | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const playbackIndexRef = useRef(0);
  const stopRequestedRef = useRef(false);
  const isPausedRef = useRef(false);
  const resumeResolverRef = useRef<(() => void) | null>(null);
  const stopResolverRef = useRef<(() => void) | null>(null);
  const webgpuInitializedRef = useRef(false);

  const [fileName, setFileName] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sentenceCount, setSentenceCount] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [ttsStatus, setTtsStatus] = useState<string | null>(null);
  const [ttsError, setTtsError] = useState<string | null>(null);
  const [webgpuSupported, setWebgpuSupported] = useState<boolean | null>(null);
  const [toc, setToc] = useState<TocItem[]>([]);
  const [activeChapterHref, setActiveChapterHref] = useState<string | null>(null);
  const isPlayingRef = useRef(false);

  useEffect(() => {
    isPausedRef.current = isPaused;
  }, [isPaused]);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

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
      } catch {
        if (!cancelled) setWebgpuSupported(false);
      }
    }

    void checkWebGpu();
    return () => {
      cancelled = true;
    };
  }, []);

  const rebuildSentenceList = useCallback(() => {
    const sections = Array.from(sentencesBySectionRef.current.entries()).sort(
      ([a], [b]) => a - b,
    );
    const all: SentenceEntry[] = [];
    const indexMap = new Map<string, number>();
    for (const [, entries] of sections) {
      const sorted = entries.sort((a, b) => a.order - b.order);
      for (const entry of sorted) {
        indexMap.set(entry.id, all.length);
        all.push(entry);
      }
    }
    sentenceListRef.current = all;
    sentenceIndexRef.current = indexMap;
    setSentenceCount(all.length);
  }, []);

  const clearHighlight = useCallback(() => {
    const rendition = renditionRef.current;
    if (!rendition?.annotations || !activeHighlightRef.current) return;
    console.log("🧹 Clearing active highlight:", activeHighlightRef.current);
    rendition.annotations.remove(activeHighlightRef.current, "highlight");
    activeHighlightRef.current = null;
  }, []);

  const clearHoverHighlight = useCallback(() => {
    const rendition = renditionRef.current;
    if (!rendition?.annotations || !hoverHighlightRef.current) return;
    rendition.annotations.remove(hoverHighlightRef.current, "highlight");
    hoverHighlightRef.current = null;
    hoverSentenceIdRef.current = null;
  }, []);

  const resetSentenceIndex = useCallback(
    (nextSectionIndex: number | null = null) => {
      sentencesBySectionRef.current.clear();
      sentenceListRef.current = [];
      sentenceIndexRef.current = new Map();
      sentenceByIdRef.current.clear();
      sentenceIdCounterRef.current = 0;
      playbackIndexRef.current = 0;
      interactiveDocsRef.current = new WeakSet();
      currentSectionIndexRef.current = nextSectionIndex;
      setSentenceCount(0);
      clearHighlight();
      clearHoverHighlight();
    },
    [clearHighlight, clearHoverHighlight],
  );

  const highlightSentence = useCallback((entry: SentenceEntry | null) => {
    const rendition = renditionRef.current;
    if (!rendition?.annotations) return;
    if (activeHighlightRef.current) {
      console.log("🧹 Removing previous highlight:", activeHighlightRef.current);
      rendition.annotations.remove(activeHighlightRef.current, "highlight");
    }
    if (entry) {
      console.log("✨ Highlighting sentence:", entry.text.substring(0, 50) + "...", "cfi:", entry.cfiRange);
      rendition.annotations.highlight(
        entry.cfiRange,
        { id: entry.id },
        undefined,
        "tts-highlight",
        HIGHLIGHT_STYLES,
      );
      activeHighlightRef.current = entry.cfiRange;
    } else {
      console.log("❌ No entry to highlight");
      activeHighlightRef.current = null;
    }
  }, []);


  const hoverSentence = useCallback((entry: SentenceEntry | null) => {
    const rendition = renditionRef.current;
    if (!rendition?.annotations) return;
    if (entry && hoverSentenceIdRef.current === entry.id) {
      return;
    }
    if (!entry && !hoverSentenceIdRef.current) {
      return;
    }
    if (hoverHighlightRef.current) {
      rendition.annotations.remove(hoverHighlightRef.current, "highlight");
    }
    if (entry) {
      rendition.annotations.highlight(
        entry.cfiRange,
        { id: entry.id, hover: true },
        undefined,
        "tts-hover",
        HOVER_STYLES,
      );
      hoverHighlightRef.current = entry.cfiRange;
      hoverSentenceIdRef.current = entry.id;
    } else {
      hoverHighlightRef.current = null;
      hoverSentenceIdRef.current = null;
    }
  }, []);


  const stopPlayback = useCallback(() => {
    console.log("🛑 stopPlayback called");
    stopRequestedRef.current = true;
    isPlayingRef.current = false;
    if (resumeResolverRef.current) {
      resumeResolverRef.current();
      resumeResolverRef.current = null;
    }
    if (stopResolverRef.current) {
      stopResolverRef.current();
      stopResolverRef.current = null;
    }
    if (playerRef.current?.context?.state === "running") {
      void playerRef.current.context.suspend();
    }
    clearHighlight();
    clearHoverHighlight();
    setIsPlaying(false);
    setIsPaused(false);
    setTtsStatus(null);
  }, [clearHighlight]);

  const clearBook = useCallback(() => {
    stopPlayback();
    if (renditionRef.current) {
      renditionRef.current.destroy();
      renditionRef.current = null;
    }
    if (bookRef.current?.destroy) {
      bookRef.current.destroy();
      bookRef.current = null;
    }
    if (viewerRef.current) {
      viewerRef.current.innerHTML = "";
    }
    processedDocsRef.current = new WeakSet();
    setTtsError(null);
    resetSentenceIndex(null);
    setToc([]);
    setActiveChapterHref(null);
  }, [resetSentenceIndex, stopPlayback]);

  useEffect(() => {
    return () => {
      clearBook();
    };
  }, [clearBook]);

  const ensureWebGpuReady = useCallback(async () => {
    if (webgpuSupported === false) {
      throw new Error("WebGPU is not available on this device.");
    }
    if (!webgpuInitializedRef.current) {
      const devices = await init();
      if (!devices.includes("webgpu")) {
        setWebgpuSupported(false);
        throw new Error("WebGPU is not available on this device.");
      }
      defaultDevice("webgpu");
      webgpuInitializedRef.current = true;
      setWebgpuSupported(true);
    }
  }, [webgpuSupported]);

  const downloadWeights = useCallback(async () => {
    if (weightsRef.current) return weightsRef.current;
    setTtsStatus("Downloading model weights...");
    const data = await cachedFetch(WEIGHTS_URL);
    const parsed = safetensors.parse(data);
    weightsRef.current = parsed;
    return parsed;
  }, []);

  const getModel = useCallback(async () => {
    if (modelRef.current) return modelRef.current;
    setTtsStatus("Loading TTS model...");
    const weights = await downloadWeights();
    const model = fromSafetensors(weights);
    modelRef.current = model;
    return model;
  }, [downloadWeights]);

  const getTokenizer = useCallback(async () => {
    if (tokenizerRef.current) return tokenizerRef.current;
    setTtsStatus("Loading tokenizer...");
    const tokenizer = await tokenizers.loadSentencePiece(TOKENIZER_URL);
    tokenizerRef.current = tokenizer;
    return tokenizer;
  }, []);

  const getVoiceEmbed = useCallback(async () => {
    if (voiceEmbedRef.current) return voiceEmbedRef.current;
    setTtsStatus("Loading voice preset...");
    const audioPrompt = safetensors.parse(
      await cachedFetch(DEFAULT_VOICE_URL),
    ).tensors.audio_prompt;
    const audioPromptData = audioPrompt.data as Float32Array<ArrayBuffer>;
    const voiceEmbed = np
      .array(audioPromptData, {
        shape: audioPrompt.shape,
        dtype: np.float32,
      })
      .slice(0)
      .astype(np.float16);
    voiceEmbedRef.current = voiceEmbed;
    return voiceEmbed;
  }, []);

  const synthesizeSentence = useCallback(
    async (text: string, player: AudioPlayer) => {
      const model = await getModel();
      const tokenizer = await getTokenizer();
      const voiceEmbed = await getVoiceEmbed();
      const [prompt, framesAfterEos] = prepareTextPrompt(text);
      const tokens = tokenizer.encode(prompt);

      const tokensAr = np.array(tokens, { dtype: np.uint32 });
      const textEmbeds = model.flowLM.conditionerEmbed.ref.slice(tokensAr);
      const embeds = np.concatenate([voiceEmbed.ref, textEmbeds]);

      await playTTS(player, tree.ref(model), embeds, { framesAfterEos });
    },
    [getModel, getTokenizer, getVoiceEmbed],
  );

  const ensureSentenceVisible = useCallback(async (entry: SentenceEntry) => {
    const rendition = renditionRef.current;
    if (!rendition) return;
    try {
      await rendition.display(entry.cfiRange);
    } catch {
      // ignore scroll errors
    }
  }, []);

  const playQueue = useCallback(async () => {
    console.log("🎵 playQueue called, isPlayingRef.current:", isPlayingRef.current);
    if (isPlayingRef.current) {
      console.log("❌ Already playing, returning");
      return;
    }
    const sentences = sentenceListRef.current;
    console.log("📝 Available sentences:", sentences.length);
    if (!sentences.length) {
      console.log("❌ No sentences available");
      setTtsError("No sentences found yet. Try scrolling to load more.");
      return;
    }

    stopRequestedRef.current = false;
    isPlayingRef.current = true;
    setIsPlaying(true);
    setIsPaused(false);
    setTtsError(null);
    setTtsStatus("Preparing TTS...");

    try {
      await ensureWebGpuReady();
      const player =
        playerRef.current ?? createStreamingPlayer({ autoResume: false });
      playerRef.current = player;
      if (player.context.state === "suspended") {
        await player.context.resume();
      }

      const stopPromise = new Promise<void>((resolve) => {
        stopResolverRef.current = resolve;
      });

      if (playbackIndexRef.current >= sentences.length) {
        playbackIndexRef.current = 0;
      }

      for (
        let index = playbackIndexRef.current;
        index < sentences.length;
        index += 1
      ) {
        if (stopRequestedRef.current) break;
        const entry = sentences[index];
        playbackIndexRef.current = index;
        await ensureSentenceVisible(entry);
        clearHoverHighlight();
        highlightSentence(entry);
        setTtsStatus(`Reading ${index + 1} of ${sentences.length}`);
        await synthesizeSentence(entry.text, player);
        await Promise.race([player.waitForEnd(), stopPromise]);

        if (stopRequestedRef.current) break;
        if (isPausedRef.current) {
          setTtsStatus("Paused");
          await new Promise<void>((resolve) => {
            resumeResolverRef.current = resolve;
          });
          resumeResolverRef.current = null;
        }
      }
    } catch (playError) {
      setTtsError(
        playError instanceof Error ? playError.message : "Playback failed.",
      );
    } finally {
      stopResolverRef.current = null;
      isPlayingRef.current = false;
      setIsPlaying(false);
      setIsPaused(false);
      setTtsStatus(null);
    }
  }, [clearHoverHighlight, ensureSentenceVisible, ensureWebGpuReady, highlightSentence, synthesizeSentence]);



  const attachSentenceInteractivity = useCallback(
    (contents: any) => {
      console.log("🔗 attachSentenceInteractivity called");
      const doc = contents?.document as Document | undefined;
      if (!doc) {
        console.log("❌ No document found");
        return;
      }
      const rendition = renditionRef.current;
      if (!rendition?.annotations) {
        console.log("❌ No annotations available");
        return;
      }
      if (interactiveDocsRef.current.has(doc)) {
        console.log("⏭️ Document already processed");
        return;
      }
      interactiveDocsRef.current.add(doc);
      console.log("✅ Document added to interactive set");

      const sectionIndex =
        typeof contents.sectionIndex === "number" ? contents.sectionIndex : 0;
      const entries = sentencesBySectionRef.current.get(sectionIndex) ?? [];
      console.log(`📖 Section ${sectionIndex}: found ${entries.length} sentences`);

      const iframe = doc.defaultView?.frameElement as HTMLElement | null;
      const container = iframe?.parentElement as HTMLElement | null;
      if (!iframe || !container) {
        console.log("❌ No iframe or container found");
        return;
      }
      console.log("✅ Found iframe and container");

      const startPlaybackFromSentence = (selectedId: string | null, source: string) => {
        console.log("🖱️ Sentence click source:", source, "id:", selectedId);
        if (!selectedId) {
          console.log("❌ No sentenceId");
          return;
        }

        const index = sentenceIndexRef.current.get(selectedId);
        if (index === undefined) {
          console.log("❌ No index found for sentenceId:", selectedId);
          return;
        }

        console.log("📍 Starting playback from index:", index, "sentenceId:", selectedId);
        console.log("🗂️ sentenceIndexRef contents:", Array.from(sentenceIndexRef.current.entries()));
        console.log("🎯 Current playbackIndexRef before:", playbackIndexRef.current);

        playbackIndexRef.current = index;
        console.log("🎯 Current playbackIndexRef after:", playbackIndexRef.current);

        stopRequestedRef.current = true;
        if (stopResolverRef.current) {
          console.log("🛑 Resolving stop promise");
          stopResolverRef.current();
          stopResolverRef.current = null;
        }

        if (playerRef.current?.context?.state === "running") {
          console.log("🔇 Suspending audio context");
          void playerRef.current.context.suspend();
        }
        clearHighlight();
        clearHoverHighlight();

        setIsPlaying(false);
        setIsPaused(false);
        setTtsStatus(null);

        console.log("🚀 Starting new playback queue after timeout");
        setTimeout(() => {
          console.log("⏰ Timeout triggered, calling playQueue()");
          void playQueue();
        }, 100);
      };

      console.log(`🎯 Processing ${entries.length} sentences for interactivity`);
      for (const entry of entries) {
        if (container.querySelector(`[data-id='${entry.id}']`)) continue;
        console.log(`🏷️ Adding hit detection for sentence: ${entry.id} - "${entry.text.substring(0, 30)}..."`);
        rendition.annotations.highlight(
          entry.cfiRange,
          { id: entry.id, hit: true },
          undefined,
          "tts-hit",
          {
            fill: "#000000",
            "fill-opacity": "0",
          },
        );
      }

      const marks = container.querySelectorAll("[ref='tts-hit']");
      console.log(`🎯 Found ${marks.length} tts-hit marks to add click listeners`);
      marks.forEach((mark) => {
        const markEl = mark as HTMLElement;
        if (markEl.dataset.sentenceBound === "true") return;
        markEl.dataset.sentenceBound = "true";
        markEl.style.cursor = "pointer";
        const sentenceId = markEl.dataset.id;

        console.log(`👂 Adding click listener to sentence: ${sentenceId}`);
        markEl.addEventListener("click", (e) => {
          console.log("🖱️ Sentence clicked (mark):", { sentenceId, e });
          startPlaybackFromSentence(sentenceId ?? null, "mark");
        });
      });

      let rafId = 0;
      let lastHoveredId: string | null = null;

      const rectContains = (rect: DOMRect, x: number, y: number) => {
        const offset = iframe.getBoundingClientRect();
        const top = rect.top - offset.top;
        const left = rect.left - offset.left;
        const bottom = top + rect.height;
        const right = left + rect.width;
        return top <= y && left <= x && bottom > y && right > x;
      };

      const findHoverId = (x: number, y: number) => {
        const hitMarks = Array.from(
          container.querySelectorAll("[ref='tts-hit']"),
        ) as HTMLElement[];
        for (const markEl of hitMarks) {
          const id = markEl.dataset.id;
          if (!id) continue;
          const rect = markEl.getBoundingClientRect();
          if (!rectContains(rect, x, y)) continue;
          const rects = markEl.getClientRects();
          for (let i = 0; i < rects.length; i += 1) {
            if (rectContains(rects[i], x, y)) {
              return id;
            }
          }
        }
        return null;
      };

      const handleClick = (event: MouseEvent) => {
        const clickedId = findHoverId(event.clientX, event.clientY);
        console.log("🖱️ Document click detected:", {
          clickedId,
          clientX: event.clientX,
          clientY: event.clientY,
        });
        startPlaybackFromSentence(clickedId, "document");
      };

      const handleMove = (event: MouseEvent) => {
        if (rafId) return;
        const { clientX, clientY } = event;
        rafId = window.requestAnimationFrame(() => {
          rafId = 0;
          const hoverId = findHoverId(clientX, clientY);
          if (hoverId === lastHoveredId) return;
          lastHoveredId = hoverId;
          if (!hoverId) {
            hoverSentence(null);
            return;
          }
          const entry = sentenceByIdRef.current.get(hoverId) ?? null;
          hoverSentence(entry);
        });
      };

      const handleLeave = () => {
        lastHoveredId = null;
        hoverSentence(null);
      };

      console.log("🖱️ Adding mouse event listeners to document");
      console.log("🖱️ Adding click listener to document (capture)");
      doc.addEventListener("mousemove", handleMove);
      doc.addEventListener("mouseleave", handleLeave);
      doc.addEventListener("click", handleClick, true);
    },
    [hoverSentence, playQueue, stopPlayback],
  );

  const handleTogglePlayback = useCallback(async () => {
    if (!isPlaying) {
      void playQueue();
      return;
    }

    if (!playerRef.current) return;

    if (isPaused) {
      setIsPaused(false);
      setTtsStatus("Resuming...");
      await playerRef.current.context.resume();
      if (resumeResolverRef.current) {
        resumeResolverRef.current();
      }
    } else {
      setIsPaused(true);
      setTtsStatus("Paused");
      await playerRef.current.context.suspend();
    }
  }, [isPaused, isPlaying, playQueue]);


  const indexSentences = useCallback(
    (contents: any) => {
      const doc = contents?.document as Document | undefined;
      const root = (contents?.content || doc?.body) as HTMLElement | null;
      if (!doc || !root) return;
      if (processedDocsRef.current.has(doc)) return;
      processedDocsRef.current.add(doc);

      const sectionIndex =
        typeof contents.sectionIndex === "number" ? contents.sectionIndex : 0;
      const language =
        doc.documentElement.lang ||
        bookRef.current?.package?.metadata?.language ||
        null;

      if (currentSectionIndexRef.current !== sectionIndex) {
        resetSentenceIndex(sectionIndex);
      }

      const blocks = Array.from(root.querySelectorAll(BLOCK_SELECTOR)).filter(
        (element) => {
          if (element.closest("nav, aside")) return false;
          const parentBlock = element.parentElement?.closest(BLOCK_SELECTOR);
          return !parentBlock;
        },
      );

      const entries: SentenceEntry[] = [];
      let order = 0;

      for (const block of blocks) {
        const { fullText, map } = buildTextMap(doc, block);
        if (!fullText.trim()) continue;

        const segments = splitIntoSentences(fullText, language);
        for (const segment of segments) {
          const trimmed = trimOffsets(fullText, segment.start, segment.end);
          if (trimmed.start >= trimmed.end) continue;

          const range = rangeFromOffsets(doc, map, trimmed.start, trimmed.end);
          if (!range || range.collapsed) continue;

          const normalized = normalizeSentence(segment.text);
          if (!normalized) continue;

          let cfiRange: string;
          try {
            cfiRange = contents.cfiFromRange(range);
          } catch {
            continue;
          }
          const entry: SentenceEntry = {
            id: `sentence-${sectionIndex}-${sentenceIdCounterRef.current++}`,
            text: normalized,
            cfiRange,
            sectionIndex,
            order,
          };
          entries.push(entry);
          sentenceByIdRef.current.set(entry.id, entry);
          order += 1;
        }
      }

      if (entries.length > 0) {
        sentencesBySectionRef.current.set(sectionIndex, entries);
        rebuildSentenceList();
      }
    },
    [rebuildSentenceList, resetSentenceIndex],
  );

  const handleFileChange = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      setError(null);
      setFileName(null);
      setActiveChapterHref(null);
      setToc([]);
      setIsLoading(true);
      clearBook();

      try {
        const buffer = await file.arrayBuffer();
        const book = ePub(buffer);
        bookRef.current = book;

        if (!viewerRef.current) {
          throw new Error("Reader container not available.");
        }

        const rendition = book.renderTo(viewerRef.current, {
          width: "100%",
          height: "100%",
          flow: "scrolled-doc",
          manager: "default",
          allowScriptedContent: true,
        }) as EpubRendition;
        renditionRef.current = rendition;
        rendition.flow("scrolled-doc");
        rendition.hooks?.content?.register((contents: any) => {
          console.log("🔗 Content hook triggered");
          indexSentences(contents);
          attachSentenceInteractivity(contents);
        });
        rendition.themes.default({
          body: {
            "font-family":
              "Geist, ui-sans-serif, system-ui, -apple-system, sans-serif",
            "line-height": "1.75",
            color: "#0f172a",
            background: "transparent",
          },
          p: {
            margin: "0 0 1em 0",
          },
          a: {
            color: "#2563eb",
          },
        });

        await rendition.display();

        const navigation = await book.loaded?.navigation;
        const tocItems = (navigation?.toc ?? []) as TocItem[];
        setToc(tocItems);

        rendition.on?.("relocated", (location: any) => {
          const href = location?.start?.href as string | undefined;
          if (href) {
            setActiveChapterHref(href.split("#")[0]);
          }
        });

        setFileName(file.name);
      } catch (loadError) {
        setError(
          loadError instanceof Error ? loadError.message : "Failed to load EPUB.",
        );
      } finally {
        setIsLoading(false);
      }
    },
    [attachSentenceInteractivity, clearBook, indexSentences],
  );

  const handleClear = useCallback(() => {
    clearBook();
    setFileName(null);
    setError(null);
  }, [clearBook]);

  const handleChapterClick = useCallback(
    (href: string) => {
      const rendition = renditionRef.current;
      if (!rendition) return;
      stopPlayback();
      resetSentenceIndex(null);
      void rendition.display(href);
      setActiveChapterHref(href.split("#")[0]);
    },
    [resetSentenceIndex, stopPlayback],
  );

  const renderTocItems = (items: TocItem[], depth = 0) => {
    if (!items.length) return null;
    return (
      <ul className={depth === 0 ? "space-y-1" : "mt-1 space-y-1"}>
        {items.map((item) => {
          const label = item.label || "Untitled";
          const href = item.href || "";
          const baseHref = href.split("#")[0];
          const isActive = activeChapterHref === baseHref;
          return (
            <li key={`${href}-${label}-${depth}`}>
              <button
                type="button"
                onClick={() => handleChapterClick(href)}
                className={`flex w-full items-start gap-2 rounded-md px-2 py-1 text-left text-sm transition-colors ${
                  isActive
                    ? "bg-slate-100 text-slate-900"
                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                }`}
                disabled={!href}
                style={{ paddingLeft: `${8 + depth * 12}px` }}
              >
                <span className="block leading-snug">{label}</span>
              </button>
              {item.subitems?.length > 0 && renderTocItems(item.subitems, depth + 1)}
            </li>
          );
        })}
      </ul>
    );
  };

  const playbackLabel = useMemo(() => {
    if (isPlaying && !isPaused) return "Pause";
    if (isPlaying && isPaused) return "Resume";
    return "Play";
  }, [isPaused, isPlaying]);

  const showReaderHeader = !activeChapterHref;

  return (
    <div className="h-screen overflow-hidden bg-slate-50">
      <div className="mx-auto flex h-full w-full max-w-7xl flex-col gap-4 px-6 py-4">
        <div className="rounded-xl border bg-white px-4 py-3 shadow-sm">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-900 text-xs font-semibold text-white">
                OS
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-900">
                  Open Speech Reader
                </p>
                <p className="text-xs text-muted-foreground">
                  {fileName ?? "No EPUB loaded"}
                </p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <label
                className={`inline-flex cursor-pointer items-center gap-2 rounded-md border-2 border-dashed px-3 py-2 text-xs font-medium transition-colors ${
                  isLoading
                    ? "cursor-not-allowed border-slate-200 bg-slate-50 text-slate-400"
                    : "border-slate-200 bg-slate-50/60 text-slate-700 hover:border-slate-300 hover:bg-slate-50"
                }`}
                aria-disabled={isLoading}
              >
                <input
                  type="file"
                  accept=".epub,application/epub+zip"
                  onChange={handleFileChange}
                  disabled={isLoading}
                  className="sr-only"
                />
                <UploadCloud className="h-4 w-4" />
                {fileName ? "Replace EPUB" : "Upload EPUB"}
              </label>
              <Button
                variant="outline"
                onClick={handleClear}
                disabled={!fileName || isLoading}
              >
                Clear
              </Button>
              <Button
                onClick={handleTogglePlayback}
                disabled={
                  isLoading ||
                  !fileName ||
                  sentenceCount === 0 ||
                  webgpuSupported === false
                }
              >
                {isPlaying && !isPaused ? (
                  <Pause className="mr-2 h-4 w-4" />
                ) : (
                  <Play className="mr-2 h-4 w-4" />
                )}
                {playbackLabel}
              </Button>
            </div>
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span>{sentenceCount} sentences indexed</span>
            {ttsStatus && <span>· {ttsStatus}</span>}
            {webgpuSupported === false && <span>· WebGPU unavailable</span>}
            <span>· Hover a sentence to preview, click to start</span>
          </div>
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertTitle>EPUB load error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {ttsError && (
          <Alert variant="destructive">
            <AlertTitle>TTS error</AlertTitle>
            <AlertDescription>{ttsError}</AlertDescription>
          </Alert>
        )}

        <div className="grid flex-1 min-h-0 gap-4 lg:grid-cols-[240px_1fr]">
          <aside className="flex min-h-0 flex-col rounded-xl border bg-white p-4 shadow-sm">
            <div className="mb-2 flex items-center justify-between">
              <p className="text-sm font-medium text-slate-700">Chapters</p>
            </div>
            <div className="flex-1 min-h-0 overflow-y-auto pr-1 text-sm">
              {toc.length ? (
                renderTocItems(toc)
              ) : (
                <p className="text-xs text-muted-foreground">
                  No chapters found.
                </p>
              )}
            </div>
          </aside>

          <section className="flex min-h-0 flex-col rounded-2xl border bg-white shadow-sm">
            {showReaderHeader && (
              <div className="border-b px-5 py-3">
                <p className="text-sm font-medium text-slate-700">Reader</p>
                <p className="text-xs text-muted-foreground">
                  Scroll to read. Hover a sentence to preview, click to start there.
                </p>
              </div>
            )}
            <div className={`flex-1 min-h-0 ${showReaderHeader ? "p-6" : "p-0"}`}>
              <div
                className={`relative h-full overflow-hidden bg-slate-50 ${
                  showReaderHeader ? "rounded-lg" : "rounded-2xl"
                }`}
              >
                <div ref={viewerRef} className="absolute inset-0" />
                {!fileName && !isLoading && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="rounded-lg border border-dashed bg-slate-50 px-4 py-6 text-center text-sm text-muted-foreground">
                      Upload an EPUB to start reading.
                    </div>
                  </div>
                )}
                {isLoading && (
                  <div className="absolute inset-0 flex items-center justify-center text-sm text-muted-foreground">
                    Loading EPUB...
                  </div>
                )}
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
