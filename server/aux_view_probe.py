"""보조 카메라 렌더 비용 가설 검증(2-6): 보이는 메시를 끄거나 삼각형을 줄이면 렌더+JPEG 시간이 얼마나 줄어드는가.

실행(CPU만 쓴다. 서버·브라우저 없이 이 파일 하나로 돈다):
    cd ~/handover && MUJOCO_GL=egl nice -n 19 server/.venv/bin/python -m server.aux_view_probe --out <폴더> [--part singles|multi|all] [--reverse]

한 시점 160x120(옆에서 105°/-10°)을 아래 조건에서 재고(singles), 두 시점(옆에서 + 비스듬히 45°/-25°)을 따로 그릴 때와
한 번에 그릴 때를 비교한다(multi).
  V0   보이는 메시 그대로(기준)       V0r  mujoco.Renderer.render 경로(aux_view.py가 쓰는 길, 참고)   V0-640  V0를 640x480으로
  V1a  메시 끄고 충돌 상자만(group 0,1,3)      V1b  V1a + 그리퍼 충돌 메시(group 4)
  V2-100/50/10/2  보이는 메시 삼각형을 그 비율로 줄인 모델(순수 numpy 꼭짓점 묶기)
  V3   로봇을 전부 끈 빈 장면(바닥, 블록, 목표만) = render+readback 고정비용
multi: S2 = Renderer(GL 컨텍스트) 2개를 따로 / M1 = 컨텍스트 하나, 버퍼 하나에 viewport 둘(읽기와 JPEG는 한 번)
시간 분해(ms): update = mj_forward + mjv_updateScene, render_read = mjr_render + mjr_readPixels(+뒤집기), jpeg = PIL 인코딩.
  GL은 비동기라 render와 readback을 정확히 나눌 수 없다. 그래서 합쳐서 보고, 고정비용은 V3에서 직접 잰다.
줄인 메시는 이 프로세스의 메모리에만 있다. third_party의 STL은 읽기만 하고 수정하지 않는다. 학습·녹화 코드(StateRenderer 등)는 쓰지 않는다.
저장: --out 폴더에 결과 JSON과 시뮬레이션 화면 PNG뿐이다(웹캠·얼굴 없음).
줄인 모델의 모양 변화: 컴파일러가 메시를 다시 무게중심에 맞추지만 그 차이는 geom 위치에 같이 더해져서 월드 위치는 그대로다(mesh_pos가 달라져도 화면은 안 어긋난다).
진짜 변화는 삼각형을 줄이면서 모양이 조금 깎이는 것이다. world_bbox_shift_mm()가 보이는 메시마다 월드 좌표 꼭짓점 상자의 이동(mm)을 재서 줄임 비율별로 보고한다.
"""

import os

# mujoco를 import하기 전에 정해야 한다.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import gl_context  # noqa: E402
from PIL import Image  # noqa: E402

from server import aux_view_config as ac  # noqa: E402
from server import config  # noqa: E402
from server.aux_mesh import (  # noqa: E402, F401  (STL_RECORD 등은 이 파일의 테스트가 probe.* 이름으로 쓴다)
    STL_RECORD,
    build_model,
    cluster_decimate,
    decimate_to_fraction,
    drawn_mesh_faces,
    read_binary_stl,
    visual_mesh_names,
)
from server.render import set_render_flags  # noqa: E402
from server.seeds import block_qpos_for_seed  # noqa: E402
from server.view import clamp_view, make_free_camera  # noqa: E402

# 카메라(제안). side는 aux_view_config.py의 값(= config.VIEW_PRESETS["side"]). 채택하면 oblique도 aux_view_config.py로 옮긴다.
VIEW_SIDE = ac.AUX_CAMERA_VIEW
VIEW_OBLIQUE = {"azimuth_deg": 45.0, "elevation_deg": -25.0, "distance_m": 0.8}

VIEW_W, VIEW_H = 160, 120  # px, 한 시점의 크기(기준)
FRACTIONS = (1.0, 0.5, 0.1, 0.02)
# MjvOption.geomgroup 값(0~5). 기본 보기는 0,1,2.
GROUPS_DEFAULT = (1, 1, 1, 0, 0, 0)
GROUPS_COLLISION = (1, 1, 0, 1, 0, 0)
GROUPS_COLLISION_PLUS_JAW = (1, 1, 0, 1, 1, 0)
GROUPS_NO_ROBOT = (1, 1, 0, 0, 0, 0)

# 메시 줄이기와 줄인 모델 만들기는 server/aux_mesh.py로 옮겼다(aux_view.py가 쓴다). 아래 이름은 같은 함수다(이 파일의 테스트와 측정이 그대로 쓴다).


def _visual_world_bboxes(model: mujoco.MjModel, qpos: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    data = mujoco.MjData(model)
    data.qpos[:13] = qpos
    mujoco.mj_forward(model, data)
    boxes = {}
    for i in range(model.ngeom):
        if model.geom_group[i] == 2 and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[i])
            start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            world = data.geom_xpos[i] + model.mesh_vert[start : start + count].astype(float) @ data.geom_xmat[i].reshape(3, 3).T
            boxes[i] = (world.min(axis=0), world.max(axis=0))
    return boxes


def world_bbox_shift_mm(base: mujoco.MjModel, model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, float]:
    """보이는 메시 geom마다 월드 좌표 꼭짓점 상자가 원본 대비 얼마나 움직였나(mm): 중심 이동의 최대와 가장자리 이동의 최대."""
    a, b = _visual_world_bboxes(base, qpos), _visual_world_bboxes(model, qpos)
    center = max(np.abs((b[i][0] + b[i][1]) / 2 - (a[i][0] + a[i][1]) / 2).max() for i in a)
    edge = max(max(np.abs(b[i][0] - a[i][0]).max(), np.abs(b[i][1] - a[i][1]).max()) for i in a)
    return {"center_max_mm": round(float(center) * 1000, 3), "edge_max_mm": round(float(edge) * 1000, 3)}


# --- 렌더 ---


class GLRenderer:
    """mujoco.Renderer와 같은 순서로 GL 컨텍스트를 만들지만 viewport 여러 개를 한 버퍼에 그릴 수 있게 저수준 함수(mjr_render)를 직접 쓴다."""

    def __init__(self, model: mujoco.MjModel, width_px: int, height_px: int) -> None:
        self.model, self.width_px, self.height_px = model, width_px, height_px
        self.data = mujoco.MjData(model)
        self._gl = gl_context.GLContext(width_px, height_px)
        self._gl.make_current()
        self._ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._ctx)
        self.scene = mujoco.MjvScene(model, maxgeom=10000)  # set_render_flags가 읽는 이름
        self.option = mujoco.MjvOption()
        self._out = np.empty((height_px, width_px, 3), dtype=np.uint8)

    def close(self) -> None:
        self._ctx.free()
        self._gl.free()

    def frame(self, qpos: np.ndarray, cameras: list, groups, view_w_px: int) -> tuple[np.ndarray, dict]:
        """카메라 하나가 view_w_px 너비 viewport 하나씩. 반환: (이미지 (H, W, 3), 단계별 ms)."""
        self._gl.make_current()
        self.option.geomgroup[:] = groups
        t0 = time.perf_counter()
        self.data.qpos[:13] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        update_s = time.perf_counter() - t0
        render_read_s = 0.0
        for i, camera in enumerate(cameras):
            t = time.perf_counter()
            mujoco.mjv_updateScene(self.model, self.data, self.option, None, camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
            set_render_flags(self, config.RENDER_SHADOW, config.RENDER_REFLECTION)
            update_s += time.perf_counter() - t
            t = time.perf_counter()
            mujoco.mjr_render(mujoco.MjrRect(i * view_w_px, 0, view_w_px, self.height_px), self.scene, self._ctx)
            render_read_s += time.perf_counter() - t
        t = time.perf_counter()
        mujoco.mjr_readPixels(self._out, None, mujoco.MjrRect(0, 0, self.width_px, self.height_px), self._ctx)
        image = np.flipud(self._out).copy()
        render_read_s += time.perf_counter() - t
        return image, {"update": update_s * 1000, "render_read": render_read_s * 1000}


def encode_jpeg(image: np.ndarray) -> tuple[bytes, float]:
    t = time.perf_counter()
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=ac.AUX_JPEG_QUALITY)
    return buf.getvalue(), (time.perf_counter() - t) * 1000


def qpos_poses() -> list[np.ndarray]:
    """고정 자세 묶음: home, 시드 몇 개의 블록 위치, 팔을 움직인 자세(test_aux_isolation.py와 같은 값)."""
    home = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH)).key("home").qpos.copy()
    poses = [home.copy()]
    for seed in (0, 1, 7, 1000):
        q = home.copy()
        q[6:13] = block_qpos_for_seed(seed)
        poses.append(q)
    for pan, lift in ((0.4, -1.2), (-0.5, -1.0)):
        q = home.copy()
        q[0], q[1] = pan, lift
        poses.append(q)
    return poses


def camera_of(view: dict) -> mujoco.MjvCamera:
    return make_free_camera(clamp_view(**view), config.VIEW_LOOKAT_M)


def pct(values: list[float], q: float) -> float:
    return round(float(np.percentile(values, q)), 2)


def summarize(stages: dict[str, list[float]], jpeg_bytes: list[int], cpu_s: float, wall_s: float, faces: int) -> dict:
    total = [sum(parts) for parts in zip(*(stages[k] for k in ("update", "render_read", "jpeg")), strict=True)]
    out = {name: {"p50": pct(v, 50), "p95": pct(v, 95)} for name, v in stages.items()}
    out["total"] = {"p50": pct(total, 50), "p95": pct(total, 95)}
    out.update(cpu_cores=round(cpu_s / wall_s, 2), jpeg_kb_mean=round(float(np.mean(jpeg_bytes)) / 1024, 2), mesh_faces_drawn=faces, frames=len(total))
    return out


def measure_shared(rend: GLRenderer, cameras: list, groups, poses: list, n: int, warmup: int, view_w: int = VIEW_W) -> dict:
    """GL 컨텍스트 하나, viewport len(cameras)개(각 너비 view_w px)를 한 버퍼에 그리고 읽기와 JPEG는 한 번."""
    stages = {"update": [], "render_read": [], "jpeg": []}
    sizes: list[int] = []
    cpu0, wall0 = time.process_time(), time.monotonic()
    for k in range(warmup + n):
        image, parts = rend.frame(poses[k % len(poses)], cameras, groups, view_w)
        jpeg, jpeg_ms = encode_jpeg(image)
        if k >= warmup:
            stages["update"].append(parts["update"])
            stages["render_read"].append(parts["render_read"])
            stages["jpeg"].append(jpeg_ms)
            sizes.append(len(jpeg))
        if k == warmup - 1:
            cpu0, wall0 = time.process_time(), time.monotonic()  # 워밍업 뒤부터 CPU를 센다
    return summarize(stages, sizes, time.process_time() - cpu0, time.monotonic() - wall0, drawn_mesh_faces(rend.model, groups))


def measure_separate(rends: list[GLRenderer], cameras: list, groups, poses: list, n: int, warmup: int) -> dict:
    """Renderer(GL 컨텍스트)가 시점마다 따로: 시점 하나당 한 장을 그리고 JPEG도 따로 한 번씩(aux 스레드 두 개를 한 스레드에서 번갈아 돌린 것과 같다)."""
    stages = {"update": [], "render_read": [], "jpeg": []}
    sizes: list[int] = []
    cpu0, wall0 = time.process_time(), time.monotonic()
    for k in range(warmup + n):
        acc = {"update": 0.0, "render_read": 0.0, "jpeg": 0.0}
        total_bytes = 0
        for rend, camera in zip(rends, cameras, strict=True):
            image, parts = rend.frame(poses[k % len(poses)], [camera], groups, VIEW_W)
            jpeg, jpeg_ms = encode_jpeg(image)
            acc["update"] += parts["update"]
            acc["render_read"] += parts["render_read"]
            acc["jpeg"] += jpeg_ms
            total_bytes += len(jpeg)
        if k >= warmup:
            for name, value in acc.items():
                stages[name].append(value)
            sizes.append(total_bytes)
        if k == warmup - 1:
            cpu0, wall0 = time.process_time(), time.monotonic()
    return summarize(stages, sizes, time.process_time() - cpu0, time.monotonic() - wall0, drawn_mesh_faces(rends[0].model, groups))


def measure_reference(model: mujoco.MjModel, camera, poses: list, n: int, warmup: int) -> dict:
    """aux_view.py가 쓰는 길(mujoco.Renderer.update_scene + render)과 같은 순서로 잰다. 저수준 경로와 비슷한지 확인하는 참고 값."""
    data = mujoco.MjData(model)
    stages = {"update": [], "render_read": [], "jpeg": []}
    sizes: list[int] = []
    with mujoco.Renderer(model, VIEW_H, VIEW_W) as renderer:
        cpu0, wall0 = time.process_time(), time.monotonic()
        for k in range(warmup + n):
            t0 = time.perf_counter()
            data.qpos[:13] = poses[k % len(poses)]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            set_render_flags(renderer, config.RENDER_SHADOW, config.RENDER_REFLECTION)
            t1 = time.perf_counter()
            pixels = renderer.render()
            t2 = time.perf_counter()
            jpeg, jpeg_ms = encode_jpeg(pixels)
            if k >= warmup:
                stages["update"].append((t1 - t0) * 1000)
                stages["render_read"].append((t2 - t1) * 1000)
                stages["jpeg"].append(jpeg_ms)
                sizes.append(len(jpeg))
            if k == warmup - 1:
                cpu0, wall0 = time.process_time(), time.monotonic()
    return summarize(stages, sizes, time.process_time() - cpu0, time.monotonic() - wall0, drawn_mesh_faces(model, GROUPS_DEFAULT))


def render_image(model: mujoco.MjModel, camera, qpos: np.ndarray, groups, width_px: int, height_px: int) -> np.ndarray:
    rend = GLRenderer(model, width_px, height_px)
    try:
        rend.frame(qpos, [camera], groups, width_px)  # 첫 렌더링은 버린다(docs/07)
        image, _ = rend.frame(qpos, [camera], groups, width_px)
    finally:
        rend.close()
    return image


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """두 이미지(0~255)의 픽셀 평균 절대 차이. 눈으로 보는 닮음이 아니라 거친 숫자다."""
    return round(float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()), 2)


# --- 실행 ---


def fmt_row(label: str, r: dict) -> str:
    return (
        f"{label:<34} update {r['update']['p50']:6.2f}/{r['update']['p95']:6.2f}  render_read {r['render_read']['p50']:7.2f}/{r['render_read']['p95']:7.2f}  "
        f"jpeg {r['jpeg']['p50']:5.2f}/{r['jpeg']['p95']:5.2f}  합계 {r['total']['p50']:7.2f}/{r['total']['p95']:7.2f} ms  "
        f"CPU {r['cpu_cores']:.2f}코어  JPEG {r['jpeg_kb_mean']:.1f}KB  메시삼각형 {r['mesh_faces_drawn']:,}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=("singles", "multi", "all"), default="all", help="5분이 넘지 않게 나눠 돌릴 수 있다")
    parser.add_argument("--n", type=int, default=60, help="조건마다 재는 프레임 수")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reverse", action="store_true", help="조건 순서를 거꾸로(순서 효과 확인용)")
    parser.add_argument("--max-load", type=float, default=50.0)
    parser.add_argument("--out", type=Path, default=None, help="결과 JSON과 PNG를 쓸 폴더(기본: 임시 폴더)")
    args = parser.parse_args()

    load_1min = os.getloadavg()[0]
    if load_1min >= args.max_load:
        print(f"load average {load_1min:.1f} >= {args.max_load:g}: 서버가 바빠서 측정하지 않는다.", file=sys.stderr)
        return 2
    os.nice(19)
    out_dir = args.out or Path(tempfile.mkdtemp(prefix="aux_probe_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    poses = qpos_poses()
    side, oblique = camera_of(VIEW_SIDE), camera_of(VIEW_OBLIQUE)
    base = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
    stl_cache: dict = {}
    models = {"V0": base}
    build_s = {}
    for fraction in FRACTIONS:
        t = time.monotonic()
        models[f"V2-{round(fraction * 100)}"] = build_model(config.SCENE_PATH, fraction, stl_cache)
        build_s[f"V2-{round(fraction * 100)}"] = round(time.monotonic() - t, 2)
    face_counts = {name: int(m.mesh_facenum.sum()) for name, m in models.items()}
    visual_faces = {name: drawn_mesh_faces(m, GROUPS_DEFAULT) for name, m in models.items()}
    shifts = {name: world_bbox_shift_mm(base, m, poses[5]) for name, m in models.items() if name != "V0"}
    print(f"load {load_1min:.2f}. 모델 만들기(s): {build_s}. 보이는 메시 삼각형: {visual_faces}")
    print(f"원본 대비 월드 좌표 꼭짓점 상자 이동(mm): {shifts}")

    results: dict[str, dict] = {}
    n, warmup = args.n, args.warmup

    def run_single(label, model, groups, width=VIEW_W, height=VIEW_H):
        rend = GLRenderer(model, width, height)
        try:
            results[label] = measure_shared(rend, [side], groups, poses, n, warmup, view_w=width)
        finally:
            rend.close()
        print(fmt_row(label, results[label]), flush=True)

    def run_reference():
        label = "V0r Renderer 경로(참고)"
        results[label] = measure_reference(models["V0"], side, poses, n, warmup)
        print(fmt_row(label, results[label]), flush=True)

    singles = [
        ("V0", lambda: run_single("V0 원본 메시", models["V0"], GROUPS_DEFAULT)),
        ("V0r", run_reference),
        ("V0-640", lambda: run_single("V0-640 640x480", models["V0"], GROUPS_DEFAULT, 640, 480)),
        ("V1a", lambda: run_single("V1a 충돌 상자만", models["V0"], GROUPS_COLLISION)),
        ("V1b", lambda: run_single("V1b 충돌 상자+그리퍼 메시", models["V0"], GROUPS_COLLISION_PLUS_JAW)),
        *[(f"V2-{p}", (lambda p=p: run_single(f"V2-{p} 메시 {p}%", models[f"V2-{p}"], GROUPS_DEFAULT))) for p in (100, 50, 10, 2)],
        ("V3", lambda: run_single("V3 빈 장면(로봇 끔)", models["V0"], GROUPS_NO_ROBOT)),
    ]

    def run_multi(name, model, groups):
        both = [side, oblique]
        rends = [GLRenderer(model, VIEW_W, VIEW_H) for _ in both]
        try:
            results[f"S2 {name}"] = measure_separate(rends, both, groups, poses, n, warmup)
        finally:
            for r in rends:
                r.close()
        print(fmt_row(f"S2 {name}", results[f"S2 {name}"]), flush=True)
        rend = GLRenderer(model, VIEW_W * 2, VIEW_H)
        try:
            results[f"M1 {name}"] = measure_shared(rend, both, groups, poses, n, warmup)
        finally:
            rend.close()
        print(fmt_row(f"M1 {name}", results[f"M1 {name}"]), flush=True)

    multis = [
        ("V0", lambda: run_multi("V0", models["V0"], GROUPS_DEFAULT)),
        ("V1a", lambda: run_multi("V1a", models["V0"], GROUPS_COLLISION)),
        *[(f"V2-{p}", (lambda p=p: run_multi(f"V2-{p}", models[f"V2-{p}"], GROUPS_DEFAULT))) for p in (50, 10, 2)],
    ]

    plan = []
    if args.part in ("singles", "all"):
        plan += [fn for _, fn in singles]
    if args.part in ("multi", "all"):
        plan += [fn for _, fn in multis]
    if args.reverse:
        plan.reverse()
    print("\n각 칸: p50/p95 ms (update, render_read, jpeg, 합계는 세 단계의 합)")
    for fn in plan:
        fn()

    # PNG(시뮬레이션 화면만): 줄임 비율별, 카메라 두 개, 160x120과 640x480. 같은 자세와 같은 카메라의 원본 대비 평균 절대 차이도 낸다.
    pose = poses[5]  # 팔을 움직인 자세
    pngs: dict[str, dict] = {}
    reference_small: dict[str, np.ndarray] = {}
    variants = [("V0", "V0", GROUPS_DEFAULT), ("V1a", "V0", GROUPS_COLLISION), ("V1b", "V0", GROUPS_COLLISION_PLUS_JAW)]
    variants += [(f"V2-{p}", f"V2-{p}", GROUPS_DEFAULT) for p in (100, 50, 10, 2)]
    for tag, model_key, groups in variants:
        for cam_name, camera in (("side", side), ("oblique", oblique)):
            small = render_image(models[model_key], camera, pose, groups, VIEW_W, VIEW_H)
            Image.fromarray(small).save(out_dir / f"{tag}_{cam_name}_160x120.png")
            entry = {"png_160x120": str(out_dir / f"{tag}_{cam_name}_160x120.png")}
            if tag.startswith("V2") or tag in ("V0", "V1a"):
                big = render_image(models[model_key], camera, pose, groups, 640, 480)
                Image.fromarray(big).save(out_dir / f"{tag}_{cam_name}_640x480.png")
                entry["png_640x480"] = str(out_dir / f"{tag}_{cam_name}_640x480.png")
            if tag == "V0":
                reference_small[cam_name] = small
            elif cam_name in reference_small:
                entry["mean_abs_diff_vs_V0_160x120"] = mean_abs_diff(small, reference_small[cam_name])
            pngs[f"{tag}_{cam_name}"] = entry
    print("\nPNG 폴더:", out_dir)
    print("원본 대비 평균 절대 차이(160x120, 0~255):", {k: v["mean_abs_diff_vs_V0_160x120"] for k, v in pngs.items() if "mean_abs_diff_vs_V0_160x120" in v})

    elapsed = round(time.monotonic() - started, 1)
    payload = {
        "part": args.part, "reverse": args.reverse, "n": n, "warmup": warmup, "load_1min_at_start": load_1min, "elapsed_s": elapsed,
        "views": {"side": VIEW_SIDE, "oblique": VIEW_OBLIQUE}, "view_px": [VIEW_W, VIEW_H], "jpeg_quality": ac.AUX_JPEG_QUALITY,
        "model_build_s": build_s, "mesh_faces_total": face_counts, "visual_faces": visual_faces, "world_bbox_shift_mm": shifts,
        "results": results, "images": pngs,
    }
    path = out_dir / f"aux_probe_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"결과 파일: {path}  (걸린 시간 {elapsed} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
