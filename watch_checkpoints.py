#!/usr/bin/env python3
"""Watch two CARL PPO checkpoints play through a camera-only web viewer."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
import time
import webbrowser

import torch

from carl_ppo import CarlEnv
from carl_rlbot.policy import Policy, load_policy


CAR_OFFSET = (13.8757, 0.0, 20.755)
CHECKPOINT_METADATA = {}


class SpectatorState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.reset = threading.Event()
        self.sequence = 0
        self.frame = None
        self.pending_match = None

    def publish(self, frame: dict) -> None:
        with self.condition:
            self.sequence += 1
            self.frame = frame
            self.condition.notify_all()

    def select_match(
        self, blue_path: Path, orange_path: Path, sample_actions: bool
    ) -> None:
        with self.condition:
            self.pending_match = (blue_path, orange_path, sample_actions)

    def take_match(self):
        with self.condition:
            match = self.pending_match
            self.pending_match = None
            return match


def legacy_observation(raw: torch.Tensor) -> torch.Tensor:
    n_sim = raw.shape[0]
    ball = raw[:, None, :9].expand(-1, 2, -1).clone()
    order = torch.tensor(((0, 1), (1, 0)), device=raw.device)
    cars = raw[:, 9:].view(n_sim, 2, 21)[:, order].clone()

    ball[:, 1, 0:2].neg_()
    ball[:, 1, 3:5].neg_()
    ball[:, 1, 6:8].neg_()
    cars[:, 1, :, 0:2].neg_()
    cars[:, 1, :, 3:5].neg_()
    cars[:, 1, :, 6:8].neg_()
    cars[:, 1, :, 9:11].neg_()
    cars[:, 1, :, 12:14].neg_()

    ball[..., 0:3] /= 6000.0
    ball[..., 3:6] /= 6000.0
    ball[..., 6:9] /= 6.0
    cars[..., 0:3] /= 6000.0
    cars[..., 3:6] /= 2300.0
    cars[..., 6:9] /= 6.0
    cars[..., 15] /= 100.0
    return torch.cat((ball, cars.flatten(2)), dim=-1)


def checkpoint_id(path: Path) -> str:
    return path.stem.removeprefix("policy_")


def discover_checkpoints(directory: Path) -> tuple[Path, Path]:
    paths = sorted(
        directory.rglob("policy_*.pt"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    if len(paths) < 2:
        raise FileNotFoundError(f"need two policy checkpoints under {directory}")

    newest = load_policy(paths[0])
    for path in paths[1:]:
        candidate = load_policy(path)
        if (
            candidate.observation_size == newest.observation_size
            and candidate.expected_cars == newest.expected_cars
        ):
            return paths[0], path
    raise ValueError("no second checkpoint is compatible with the newest checkpoint")


def list_checkpoints(directory: Path) -> list[dict]:
    checkpoints = []
    paths = sorted(
        directory.rglob("policy_*.pt"),
        key=lambda item: (item.stat().st_mtime_ns, item.name),
        reverse=True,
    )
    live_paths = set(paths)
    for stale_path in CHECKPOINT_METADATA.keys() - live_paths:
        del CHECKPOINT_METADATA[stale_path]
    for path in paths:
        modified = path.stat().st_mtime_ns
        cached = CHECKPOINT_METADATA.get(path)
        if cached is None or cached[0] != modified:
            try:
                policy = load_policy(path)
            except (KeyError, RuntimeError, ValueError):
                continue
            metadata = {
                "path": path.relative_to(directory).as_posix(),
                "label": path.relative_to(directory).as_posix(),
                "observation_size": policy.observation_size,
                "tick_skip": policy.tick_skip,
                "deep_head": policy.deep_head,
                "modified": modified,
            }
            CHECKPOINT_METADATA[path] = (modified, metadata)
        checkpoints.append(CHECKPOINT_METADATA[path][1])
    return checkpoints


def resolve_checkpoint(directory: Path, value: str) -> Path:
    root = directory.resolve()
    path = (root / value).resolve()
    if root not in path.parents or not path.is_file() or not path.match("policy_*.pt"):
        raise ValueError("invalid checkpoint path")
    return path


def load_pair(blue_path: Path, orange_path: Path) -> tuple[Policy, Policy]:
    blue = load_policy(blue_path).to("cuda")
    orange = load_policy(orange_path).to("cuda")
    if blue.observation_size != orange.observation_size:
        raise ValueError("checkpoints use different observation layouts")
    if blue.expected_cars != 2 or orange.expected_cars != 2:
        raise ValueError("spectator currently requires 1v1 checkpoints")
    if blue.tick_skip != orange.tick_skip:
        raise ValueError("checkpoints use different policy frequencies")
    return blue, orange


def vector(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values]


def render_frame(
    raw: torch.Tensor,
    blue_path: Path,
    orange_path: Path,
    blue_score: int,
    orange_score: int,
    game: int,
    tick: int,
    sample_actions: bool,
) -> dict:
    raw = raw[0].detach().cpu()
    ball = raw[:9]
    car_values = raw[9:].view(2, 21)
    cars = []
    for team, car in enumerate(car_values):
        forward = car[9:12]
        up = car[12:15]
        right = torch.linalg.cross(up, forward)
        center = (
            car[:3]
            + forward * CAR_OFFSET[0]
            + right * CAR_OFFSET[1]
            + up * CAR_OFFSET[2]
        )
        cars.append(
            {
                "team": team,
                "pos": vector(center),
                "fwd": vector(forward),
                "rgt": vector(right),
                "up": vector(up),
                "boost": float(car[15]),
                "boosting": bool(car[20]),
                "demoed": bool(car[17]),
            }
        )
    return {
        "tick": tick,
        "game": game,
        "sample_actions": sample_actions,
        "blue": {"checkpoint": checkpoint_id(blue_path), "score": blue_score},
        "orange": {
            "checkpoint": checkpoint_id(orange_path),
            "score": orange_score,
        },
        "cars": cars,
        "ball": {"pos": vector(ball[:3])},
    }


def simulate(
    state: SpectatorState,
    blue_path: Path,
    orange_path: Path,
    tick_skip: int | None,
    max_ticks: int,
    sample_actions: bool,
    seed: int,
) -> None:
    try:
        configured_tick_skip = tick_skip
        blue, orange = load_pair(blue_path, orange_path)
        tick_skip = configured_tick_skip or blue.tick_skip
        env = CarlEnv(1, 1, 1, seed, max_ticks, 1.0, tick_skip)
        observations = env.reset()

        def initial_hidden() -> list[torch.Tensor]:
            return [
                torch.zeros(1, 1, blue.hidden_size, device="cuda"),
                torch.zeros(1, 1, orange.hidden_size, device="cuda"),
            ]

        hidden = initial_hidden()
        blue_score = orange_score = 0
        game = 1
        tick = 0
        next_step = time.perf_counter()

        while not state.stop.is_set():
            pending_match = state.take_match()
            if pending_match is not None:
                next_blue_path, next_orange_path, next_sample = pending_match
                try:
                    next_blue, next_orange = load_pair(next_blue_path, next_orange_path)
                    next_tick_skip = configured_tick_skip or next_blue.tick_skip
                    next_env = CarlEnv(1, 1, 1, seed, max_ticks, 1.0, next_tick_skip)
                    next_observations = next_env.reset()
                except Exception as error:
                    state.publish({"error": f"{type(error).__name__}: {error}"})
                else:
                    blue_path, orange_path = next_blue_path, next_orange_path
                    blue, orange = next_blue, next_orange
                    sample_actions = next_sample
                    tick_skip = next_tick_skip
                    env = next_env
                    observations = next_observations
                    hidden = initial_hidden()
                    blue_score = orange_score = 0
                    game += 1
                    tick = 0
                    next_step = time.perf_counter()

            if state.reset.is_set():
                state.reset.clear()
                observations = env.reset()
                hidden = initial_hidden()
                blue_score = orange_score = 0
                game += 1
                tick = 0

            raw = env.raw_state()
            if blue.observation_layout == "legacy":
                policy_observations = legacy_observation(raw)
            else:
                policy_observations = observations.view(1, 2, -1)

            with torch.inference_mode():
                blue_action, hidden[0] = blue.act(
                    policy_observations[:, 0], hidden[0], sample=sample_actions
                )
                orange_action, hidden[1] = orange.act(
                    policy_observations[:, 1], hidden[1], sample=sample_actions
                )
            actions = torch.stack((blue_action, orange_action), dim=1).flatten(0, 1)
            env_step = env.step(actions)
            observations = env_step.observation
            tick += tick_skip

            goal = int(env_step.info["goal"].view(1, 2)[0, 0].item())
            if goal > 0:
                blue_score += goal
            elif goal < 0:
                orange_score -= goal

            raw = env.raw_state()
            state.publish(
                render_frame(
                    raw,
                    blue_path,
                    orange_path,
                    blue_score,
                    orange_score,
                    game,
                    tick,
                    sample_actions,
                )
            )

            if env_step.done.any():
                hidden = initial_hidden()
                blue_score = orange_score = 0
                game += 1
                tick = 0

            next_step += tick_skip / 120.0
            delay = next_step - time.perf_counter()
            if delay > 0:
                state.stop.wait(delay)
            else:
                next_step = time.perf_counter()
    except Exception as error:
        state.publish({"error": f"{type(error).__name__}: {error}"})


def make_handler(
    state: SpectatorState,
    frontend: Path,
    arena: Path,
    checkpoint_dir: Path,
):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path == "/api/reset":
                state.reset.set()
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            if self.path != "/api/match":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 8192:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                blue_path = resolve_checkpoint(checkpoint_dir, payload["blue"])
                orange_path = resolve_checkpoint(checkpoint_dir, payload["orange"])
                sample_actions = bool(payload.get("sample_actions", False))
                blue_policy = load_policy(blue_path)
                orange_policy = load_policy(orange_path)
                if (
                    blue_policy.observation_size != orange_policy.observation_size
                    or blue_policy.expected_cars != orange_policy.expected_cars
                    or blue_policy.tick_skip != orange_policy.tick_skip
                ):
                    raise ValueError("selected checkpoints are incompatible")
            except (
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                self.send_error(HTTPStatus.BAD_REQUEST, str(error))
                return
            state.select_match(blue_path, orange_path, sample_actions)
            self.send_response(HTTPStatus.ACCEPTED)
            self.end_headers()

        def do_GET(self) -> None:
            if self.path == "/api/checkpoints":
                payload = json.dumps(list_checkpoints(checkpoint_dir)).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path == "/api/stream":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                sequence = 0
                try:
                    while True:
                        with state.condition:
                            state.condition.wait_for(
                                lambda: state.sequence > sequence, timeout=10.0
                            )
                            if state.sequence == sequence:
                                self.wfile.write(b": keepalive\n\n")
                                self.wfile.flush()
                                continue
                            sequence = state.sequence
                            payload = json.dumps(state.frame, separators=(",", ":"))
                        self.wfile.write(f"data: {payload}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            path = arena if self.path == "/arena.obj" else frontend / "index.html"
            if self.path == "/app.js":
                path = frontend / "app.js"
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(path.read_bytes())

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=root / "checkpoints" / "carl_gru_ppo",
    )
    parser.add_argument("--blue", type=Path)
    parser.add_argument("--orange", type=Path)
    parser.add_argument("--tick-skip", type=int)
    parser.add_argument("--max-ticks", type=int, default=4096)
    parser.add_argument("--sample-actions", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if (args.blue is None) != (args.orange is None):
        parser.error("--blue and --orange must be provided together")
    if args.tick_skip is not None and args.tick_skip < 1:
        parser.error("tick and episode lengths must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CARL checkpoint playback requires CUDA")
    if args.blue is None:
        blue_path, orange_path = discover_checkpoints(args.checkpoint_dir)
    else:
        blue_path, orange_path = args.blue.resolve(), args.orange.resolve()

    carl_root = Path(__file__).resolve().parent.parent / "CARL"
    frontend = carl_root / "web" / "checkpoint"
    arena = carl_root / "assets" / "arena.obj"
    if not frontend.is_dir() or not arena.is_file():
        raise FileNotFoundError(f"CARL spectator assets not found under {carl_root}")

    state = SpectatorState()
    thread = threading.Thread(
        target=simulate,
        args=(
            state,
            blue_path,
            orange_path,
            args.tick_skip,
            args.max_ticks,
            args.sample_actions,
            args.seed,
        ),
        daemon=True,
    )
    thread.start()

    url = f"http://{args.host}:{args.port}"
    print(f"Blue:   {blue_path}")
    print(f"Orange: {orange_path}")
    print(f"Viewer: {url}")
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(state, frontend, arena, args.checkpoint_dir),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop.set()
        thread.join(timeout=5.0)
        server.server_close()


if __name__ == "__main__":
    main()
