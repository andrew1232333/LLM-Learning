"""
Train one model on several machines at once by averaging their weights every so often.

Every machine trains its own copy on its own random batches for --sync_every seconds.
Then each sends how far its weights moved since the last average, the server averages
those moves (weighted by how many steps each machine did, so a faster machine counts
for more), and everyone applies the same average, so all copies are identical again.
Repeat. This is "local SGD", the core idea of DiLoCo.

Ordinary data parallelism averages gradients after every step, which needs a network
about as fast as the GPU. Averaging every minute or so needs very little: the model
crosses the network once per round instead of once per step, so Wi-Fi is plenty.

One machine runs the server (--serve) as well as training; the others connect to it.

  halo:  python train.py ... --sync_every 60 --serve --workers 2
  Mac:   python train.py ... --sync_every 60 --sync_addr 192.168.4.103:29500
"""

import json
import threading
import time
from multiprocessing.connection import Client, Listener

import torch
from torch.nn.utils import parameters_to_vector

KEY = b"tiny-llm"  # both ends must use the same key; keep the port on your own network


def to_bf16_bytes(vec):
    # Weight *moves* are small numbers, so bfloat16 keeps them accurate enough and halves
    # what crosses the network. (The weights themselves stay float32 everywhere.)
    return vec.to(torch.bfloat16).cpu().view(torch.int16).numpy().tobytes()


def from_bf16_bytes(data):
    return torch.frombuffer(bytearray(data), dtype=torch.int16).view(torch.bfloat16).float()


def send(conn, info, data):
    conn.send_bytes(json.dumps(info).encode())
    conn.send_bytes(data)


def recv(conn):
    return json.loads(conn.recv_bytes()), conn.recv_bytes()


def serve(port, workers):
    """The averaging server. Runs in a background thread on one machine."""
    with Listener(("0.0.0.0", port), authkey=KEY) as listener:
        conns = [listener.accept() for _ in range(workers)]
        # Everyone starts from the serving machine's weights, and it gets rank 0.
        pairs = sorted(((c, recv(c)) for c in conns), key=lambda p: not p[1][0]["main"])
        conns = [c for c, _ in pairs]
        (info, start) = pairs[0][1]
        step = info["step"]
        for rank, c in enumerate(conns):
            send(c, {"rank": rank, "step": step}, start)
        try:
            while True:
                msgs = [recv(c) for c in conns]
                steps = [info["steps"] for info, _ in msgs]
                avg = sum(from_bf16_bytes(d) * s for (_, d), s in zip(msgs, steps)) / max(1, sum(steps))
                step += sum(steps)
                reply = to_bf16_bytes(avg)
                for c in conns:
                    send(c, {"step": step}, reply)
        except EOFError:
            pass  # training finished and the workers hung up


class Sync:
    def __init__(self, model, args, step):
        host, port = args.sync_addr.rsplit(":", 1)
        if args.serve:
            threading.Thread(target=serve, args=(int(port), args.workers), daemon=True).start()
        for attempt in range(600):  # the server may not be up yet, or Wi-Fi may blip: retry for ~10 minutes
            try:
                self.conn = Client((host, int(port)), authkey=KEY)
                break
            except OSError as e:
                if attempt % 30 == 0:
                    print(f"Can't reach {args.sync_addr} yet ({e}), retrying...")
                time.sleep(1)
        else:
            raise SystemExit(f"Couldn't reach the sync server at {args.sync_addr}")
        print(f"Connected to {args.sync_addr}, waiting for {args.workers} machines...")
        weights = parameters_to_vector(model.parameters()).detach()
        send(self.conn, {"main": args.serve, "step": step}, weights.cpu().numpy().tobytes())
        info, start = recv(self.conn)
        self.rank, self.step = info["rank"], info["step"]
        # The anchor is where every copy agreed the weights were at the last average.
        self.anchor = torch.frombuffer(bytearray(start), dtype=torch.float32).to(weights.device)
        self.load(model)
        self.every, self.steps, self.last = args.sync_every, 0, time.time()
        print(f"All {args.workers} machines connected. This one is rank {self.rank}.")

    def load(self, model):
        with torch.no_grad():
            i = 0
            for p in model.parameters():
                p.copy_(self.anchor[i:i + p.numel()].view_as(p))
                i += p.numel()

    def due(self):
        return time.time() - self.last >= self.every

    def __call__(self, model):
        """Average with the other machines. Returns the total steps done by all of them."""
        t0 = time.time()
        moved = parameters_to_vector(model.parameters()).detach() - self.anchor
        send(self.conn, {"steps": self.steps}, to_bf16_bytes(moved))
        info, avg = recv(self.conn)
        self.anchor += from_bf16_bytes(avg).to(self.anchor.device)
        self.load(model)
        # Every machine should print the same weight norm: proof the copies agree again.
        print(f"sync  | {self.steps} steps here, {info['step']} in total | "
              f"weight norm {self.anchor.norm():.4f} | {time.time() - t0:.1f}s")
        self.steps, self.last = 0, time.time()
        return info["step"]
