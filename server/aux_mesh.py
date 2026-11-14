"""보조 카메라(2-6)용 줄인 로봇 모델 만들기. 순수 numpy(새 패키지 없음). aux_view.py가 시작할 때 한 번 부르고, aux_view_probe.py(측정 도구)도 쓴다.

보조 화면은 로봇의 보이는 메시(삼각형 약 40만 개)를 그리느라 렌더가 약 50 ms 걸렸다(프로브 측정). 삼각형을 10%로 줄이면 약 10 ms다.
- 줄이기는 꼭짓점 묶기(vertex clustering): 꼭짓점을 격자 칸으로 묶어 평균 위치 하나로 바꾸고 찌그러진 삼각형을 지운다.
- 줄인 모델은 이 프로세스 메모리 안에만 있다(MjSpec). third_party의 STL은 읽기만 하고 수정하지 않는다. 시뮬레이션(SimWorker)과 학습용 StateRenderer는 원래 장면을 그대로 쓴다.
- 컴파일러가 줄인 메시를 다시 무게중심에 맞추지만 그 차이는 geom 위치에 같이 더해져서 월드 위치는 그대로다. 변하는 것은 깎이는 모양뿐이다(10%에서 가장자리 최대 약 2 mm, 프로브 측정).
- 이 파일은 학습·녹화·AUTO·평가 코드가 import하지 않는다(server/tests/test_aux_isolation.py, test_aux_mesh.py).
"""

from pathlib import Path

import mujoco
import numpy as np

STL_RECORD = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])  # 바이너리 STL 삼각형 하나 = 50바이트


def read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """바이너리 STL을 읽어 (꼭짓점 (n, 3) m, 삼각형 (m, 3) 꼭짓점 번호)로 돌려준다. 같은 위치 꼭짓점은 하나로 합친다. 파일은 읽기만 한다."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        header = f.read(84)
    count = int.from_bytes(header[80:84], "little")
    if size != 84 + 50 * count:
        raise ValueError(f"{path.name}: 바이너리 STL이 아니다(크기 {size}, 삼각형 {count}개 기준 {84 + 50 * count})")
    raw = np.fromfile(path, dtype=STL_RECORD, offset=84)
    corners = raw["vertices"].reshape(-1, 3).astype(np.float64)
    verts, inverse = np.unique(corners, axis=0, return_inverse=True)
    return verts, inverse.reshape(-1, 3).astype(np.int64)


def cluster_decimate(verts: np.ndarray, faces: np.ndarray, cell_m: float) -> tuple[np.ndarray, np.ndarray]:
    """꼭짓점을 한 변 cell_m(m)인 격자 칸으로 묶어 삼각형을 줄인다(vertex clustering). 칸 안 꼭짓점은 평균 위치 하나가 된다.

    찌그러져 없어진 삼각형(꼭짓점 둘 이상이 같은 칸)과 같은 세 꼭짓점의 중복 삼각형은 지운다. 쓰지 않는 꼭짓점도 지운다.
    같은 입력이면 항상 같은 결과다(무작위 없음).
    """
    if cell_m <= 0:
        raise ValueError("cell_m must be > 0")
    cells = np.floor((verts - verts.min(axis=0)) / cell_m).astype(np.int64)
    dims = cells.max(axis=0) + 1
    key = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
    _, cluster_of, counts = np.unique(key, return_inverse=True, return_counts=True)
    cluster_of = cluster_of.reshape(-1)
    sums = np.stack([np.bincount(cluster_of, weights=verts[:, axis], minlength=len(counts)) for axis in range(3)], axis=1)
    new_verts = sums / counts[:, None]

    new_faces = cluster_of[faces]
    a, b, c = new_faces[:, 0], new_faces[:, 1], new_faces[:, 2]
    new_faces = new_faces[(a != b) & (b != c) & (a != c)]
    n = len(new_verts)
    ordered = np.sort(new_faces, axis=1)
    face_key = (ordered[:, 0] * n + ordered[:, 1]) * n + ordered[:, 2]
    _, first = np.unique(face_key, return_index=True)
    new_faces = new_faces[np.sort(first)]  # 처음 나온 방향(감는 방향)을 남긴다

    used, remap = np.unique(new_faces, return_inverse=True)
    return new_verts[used], remap.reshape(-1, 3).astype(np.int64)


def decimate_to_fraction(
    verts: np.ndarray, faces: np.ndarray, fraction: float, tolerance: float = 0.08
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """삼각형 수가 원래의 fraction(0~1) 안팎이 되도록 칸 크기를 이분법으로 찾는다. fraction >= 1이면 그대로(칸 크기 None)."""
    if fraction >= 1.0:
        return verts, faces, None
    target = max(4, int(round(len(faces) * fraction)))
    extent = float((verts.max(axis=0) - verts.min(axis=0)).max())
    lo, hi = extent * 1e-4, extent  # 칸이 너무 작으면 삼각형이 많이 남고, 크면 적게 남는다
    best = None
    for _ in range(24):
        mid = float(np.sqrt(lo * hi))
        new_verts, new_faces = cluster_decimate(verts, faces, mid)
        best = (new_verts, new_faces, mid)
        count = len(new_faces)
        if abs(count - target) <= tolerance * target:
            break
        if count > target:
            lo = mid
        else:
            hi = mid
    return best


def visual_mesh_names(model: mujoco.MjModel) -> list[str]:
    """보이는(group 2) geom이 쓰는 메시 이름."""
    ids = {int(model.geom_dataid[i]) for i in range(model.ngeom) if model.geom_group[i] == 2 and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH}
    return sorted(model.mesh(i).name for i in ids)


def build_model(scene_path: Path, fraction: float, stl_cache: dict | None = None) -> mujoco.MjModel:
    """보이는 메시의 삼각형을 fraction 비율로 줄인 aux 전용 모델(메모리에만 있다). fraction이 1이어도 같은 경로로 만든다.

    충돌용 메시(group 4)와 상자, 장면의 바닥·블록·목표는 그대로다. 모델의 오프스크린 버퍼 크기 같은 장면 설정도 그대로 이어진다.
    """
    stl_cache = {} if stl_cache is None else stl_cache
    spec = mujoco.MjSpec.from_file(str(scene_path))
    mesh_dir = Path(scene_path).parent / spec.meshdir
    names = set(visual_mesh_names(mujoco.MjModel.from_xml_path(str(scene_path))))
    for mesh in spec.meshes:
        if mesh.name not in names:
            continue
        path = mesh_dir / mesh.file
        if path not in stl_cache:
            stl_cache[path] = read_binary_stl(path)
        verts, faces, _ = decimate_to_fraction(*stl_cache[path], fraction)
        mesh.file = ""
        mesh.uservert = verts.ravel()
        mesh.userface = faces.ravel().astype(np.int32)
    return spec.compile()


def drawn_mesh_faces(model: mujoco.MjModel, groups) -> int:
    """geomgroup 설정에서 보이는 메시 geom이 그릴 삼각형 수(상자·구 등 기본 도형은 세지 않는다: 몇 백 개 수준)."""
    total = 0
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and groups[int(model.geom_group[i])]:
            total += int(model.mesh_facenum[int(model.geom_dataid[i])])
    return total
