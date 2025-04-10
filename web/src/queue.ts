export class AsyncQueue<T> {
  private _queue: T[];
  private _resolvers: ((value: T) => void)[];

  constructor() {
    this._queue = [];
    this._resolvers = [];
  }

  // Add an item to the queue
  push(item: T): void {
    if (this._resolvers.length) {
      // If there's a waiting consumer, immediately resolve it
      const resolve = this._resolvers.shift();
      if (resolve) {
        resolve(item);
      }
    } else {
      this._queue.push(item);
    }
  }

  // Retrieve an item from the queue; if empty, wait until one is available
  shift(): Promise<T> {
    if (this._queue.length) {
      return Promise.resolve(this._queue.shift() as T);
    }
    return new Promise((resolve) => {
      this._resolvers.push(resolve);
    });
  }
}
