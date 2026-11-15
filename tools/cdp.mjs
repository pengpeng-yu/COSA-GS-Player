export function connectCdp(endpoint) {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(endpoint);
    const pending = new Map();
    const events = [];
    let nextId = 0;
    const openingTimeout = setTimeout(() => {
      socket.close();
      reject(new Error('CDP WebSocket connection timed out'));
    }, 10000);
    socket.addEventListener(
      'error',
      () => {
        clearTimeout(openingTimeout);
        reject(new Error('CDP WebSocket connection failed'));
      },
      { once: true },
    );
    socket.addEventListener('message', ({ data }) => {
      const message = JSON.parse(data);
      if (message.id !== undefined) {
        const call = pending.get(message.id);
        if (!call) return;
        pending.delete(message.id);
        clearTimeout(call.timer);
        if (message.error) call.reject(new Error(JSON.stringify(message.error)));
        else call.resolve(message.result);
      } else {
        events.push(message);
      }
    });
    socket.addEventListener('close', () => {
      for (const call of pending.values()) {
        clearTimeout(call.timer);
        call.reject(new Error('CDP socket closed'));
      }
      pending.clear();
    });
    socket.addEventListener(
      'open',
      () => {
        clearTimeout(openingTimeout);
        resolve({
          events,
          close: () => socket.close(),
          send(method, params = {}, sessionId) {
            if (socket.readyState !== WebSocket.OPEN) {
              return Promise.reject(new Error('CDP socket is not open: ' + method));
            }
            const id = ++nextId;
            return new Promise((resolveCall, rejectCall) => {
              const timer = setTimeout(() => {
                pending.delete(id);
                rejectCall(new Error('CDP command timed out: ' + method));
              }, 300000);
              pending.set(id, { resolve: resolveCall, reject: rejectCall, timer });
              socket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
            });
          },
        });
      },
      { once: true },
    );
  });
}

export function evaluator(cdp, sessionId) {
  return async expression => {
    const result = await cdp.send(
      'Runtime.evaluate',
      { expression, awaitPromise: true, returnByValue: true },
      sessionId,
    );
    if (result.exceptionDetails)
      throw new Error(
        result.exceptionDetails.exception?.description ?? JSON.stringify(result.exceptionDetails),
      );
    return result.result.value;
  };
}
export function browserErrors(cdp, sessionId) {
  return cdp.events
    .filter(
      e =>
        e.sessionId === sessionId &&
        (e.method === 'Runtime.exceptionThrown' ||
          (e.method === 'Runtime.consoleAPICalled' && e.params.type === 'error')),
    )
    .map(e => e.params);
}
