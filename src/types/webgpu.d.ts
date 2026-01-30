export {};

declare global {
  interface Navigator {
    gpu?: {
      requestAdapter: () => Promise<unknown>;
    };
  }
}
