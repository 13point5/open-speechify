import type { ChangeEvent } from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import ePub from "epubjs";

import { UploadCloud } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";

type EpubRendition = {
  display: () => Promise<void>;
  flow: (flowMode: string) => void;
  destroy: () => void;
  themes: {
    default: (theme: Record<string, Record<string, string>>) => void;
  };
};

export function EpubReader() {
  const viewerRef = useRef<HTMLDivElement | null>(null);
  const bookRef = useRef<any>(null);
  const renditionRef = useRef<EpubRendition | null>(null);

  const [fileName, setFileName] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const clearBook = useCallback(() => {
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
  }, []);

  useEffect(() => {
    return () => {
      clearBook();
    };
  }, [clearBook]);

  const handleFileChange = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      setError(null);
      setFileName(null);
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
          flow: "scrolled",
          manager: "continuous",
          allowScriptedContent: true,
        }) as EpubRendition;
        renditionRef.current = rendition;
        rendition.flow("scrolled");
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

        const manager = (rendition as { manager?: any }).manager;
        if (manager) {
          manager.isVisible = () => true;
          manager.trim = () => Promise.resolve();
        }
        setFileName(file.name);
      } catch (loadError) {
        setError(
          loadError instanceof Error ? loadError.message : "Failed to load EPUB.",
        );
      } finally {
        setIsLoading(false);
      }
    },
    [clearBook],
  );

  const handleClear = useCallback(() => {
    clearBook();
    setFileName(null);
    setError(null);
  }, [clearBook]);

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-6 px-6 py-10">
        <header className="space-y-2">
          <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
            Epub Reader
          </p>
          <h1 className="text-3xl font-semibold tracking-tight text-slate-900">
            Continuous reading view
          </h1>
          <p className="text-sm text-muted-foreground">
            Upload an EPUB file to render it in a scrollable reading surface.
          </p>
        </header>

        {error && (
          <Alert variant="destructive">
            <AlertTitle>EPUB load error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <section className="rounded-xl border bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-sm font-medium text-slate-700">Upload EPUB</p>
                <p className="text-xs text-muted-foreground">
                  Choose an .epub file to start reading.
                </p>
              </div>
              <Button
                variant="outline"
                onClick={handleClear}
                disabled={!fileName || isLoading}
              >
                Clear
              </Button>
            </div>

            <label
              className={`group flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-8 text-center transition-colors ${
                isLoading
                  ? "cursor-not-allowed border-slate-200 bg-slate-50 text-slate-400"
                  : "border-slate-200 bg-slate-50/60 text-slate-600 hover:border-slate-300 hover:bg-slate-50"
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
              <span className="inline-flex items-center justify-center rounded-full bg-white p-3 shadow-sm ring-1 ring-slate-200">
                <UploadCloud className="h-5 w-5" />
              </span>
              <span className="mt-3 text-sm font-medium text-slate-700">
                Drop your EPUB here
              </span>
              <span className="mt-1 text-xs text-muted-foreground">
                or click to browse
              </span>
              {fileName && (
                <span className="mt-3 text-xs text-slate-500">
                  Loaded: {fileName}
                </span>
              )}
            </label>
          </div>
        </section>

        <section className="rounded-2xl border bg-white shadow-sm">
          <div className="border-b px-5 py-3">
            <p className="text-sm font-medium text-slate-700">
              Reader
            </p>
            <p className="text-xs text-muted-foreground">
              Scroll to read. The content will fill this panel.
            </p>
          </div>
          <div className="p-6">
            <div
              ref={viewerRef}
              className="relative h-[70vh] rounded-lg bg-slate-50/70"
            />
            {!fileName && !isLoading && (
              <div className="mt-6 rounded-lg border border-dashed bg-slate-50 px-4 py-6 text-center text-sm text-muted-foreground">
                Drop in an EPUB file to begin reading.
              </div>
            )}
            {isLoading && (
              <div className="mt-6 text-sm text-muted-foreground">
                Loading EPUB...
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
