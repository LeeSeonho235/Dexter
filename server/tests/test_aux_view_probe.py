"""server/aux_view_probe.py: 메시 줄이기(순수 numpy), STL 읽기, 줄인 모델 만들기, 저수준 렌더 경로의 확인. 시간 측정 자체는 여기서 하지 않는다."""

import hashlib

import mujoco
import numpy as np
import pytest
from test_aux_isolation import NOT_IMPORTING_AUX, SERVER_DIR, imported_and_used_names

from server import aux_view_probe as probe
from server import config

REAL_STL = config.REPO_ROOT / "third_party" / "mujoco_menagerie" / "robotstudio_so101" / "assets" / "sts3215_03a_v1.stl"


def grid_mesh(n: int) -> tuple[np.ndarray, np.ndarray]:
    """n x n 칸 격자 면(삼각형 2n^2개), 살짝 굽어 있다."""
    xs = np.linspace(0.0, 1.0, n + 1)
    verts = np.array([(x, y, 0.05 * np.sin(4 * x) * y) for y in xs for x in xs])
    faces = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            faces += [(a, a + 1, a + n + 1), (a + 1, a + n + 2, a + n + 1)]
    return verts, np.array(faces, dtype=np.int64)


def write_stl(path, verts, faces):
    record = np.zeros(len(faces), dtype=probe.STL_RECORD)
    record["vertices"] = verts[faces].astype(np.float32)
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + np.uint32(len(faces)).tobytes())
        record.tofile(f)


def test_read_binary_stl_merges_shared_vertices(tmp_path):
    verts, faces = grid_mesh(4)
    path = tmp_path / "grid.stl"
    write_stl(path, verts, faces)
    read_verts, read_faces = probe.read_binary_stl(path)
    assert read_faces.shape == (32, 3) and len(read_verts) == 25  # 꼭짓점이 합쳐진다
    assert np.allclose(np.sort(read_verts[read_faces].reshape(-1, 3), axis=0), np.sort(verts[faces].astype(np.float32).reshape(-1, 3), axis=0))


def test_read_binary_stl_rejects_non_binary(tmp_path):
    path = tmp_path / "ascii.stl"
    path.write_text("solid x\nendsolid x\n")
    with pytest.raises(ValueError):
        probe.read_binary_stl(path)


def test_cluster_decimate_reduces_and_stays_valid():
    verts, faces = grid_mesh(30)
    new_verts, new_faces = probe.cluster_decimate(verts, faces, 0.1)
    assert 0 < len(new_faces) < len(faces) / 4
    assert new_faces.min() == 0 and new_faces.max() == len(new_verts) - 1  # 쓰지 않는 꼭짓점이 없다
    a, b, c = new_faces.T
    assert np.all((a != b) & (b != c) & (a != c))  # 찌그러진 삼각형이 없다
    assert len(np.unique(np.sort(new_faces, axis=1), axis=0)) == len(new_faces)  # 중복 삼각형이 없다
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    assert np.all(new_verts.min(axis=0) >= lo - 1e-9) and np.all(new_verts.max(axis=0) <= hi + 1e-9)
    assert np.all(new_verts.min(axis=0) - lo <= 0.1) and np.all(hi - new_verts.max(axis=0) <= 0.1)  # 칸 크기 안에서 범위 보존


def test_cluster_decimate_is_deterministic_and_tiny_cells_keep_everything():
    verts, faces = grid_mesh(20)
    first = probe.cluster_decimate(verts, faces, 0.07)
    second = probe.cluster_decimate(verts, faces, 0.07)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    same_verts, same_faces = probe.cluster_decimate(verts, faces, 1e-6)
    assert len(same_faces) == len(faces) and len(same_verts) == len(verts)
    with pytest.raises(ValueError):
        probe.cluster_decimate(verts, faces, 0.0)


def test_decimate_to_fraction_hits_target_and_is_monotonic():
    verts, faces = grid_mesh(60)  # 7200 삼각형
    same = probe.decimate_to_fraction(verts, faces, 1.0)
    assert same[0] is verts and same[1] is faces and same[2] is None
    counts = {}
    for fraction in (0.5, 0.1, 0.02):
        _, new_faces, cell = probe.decimate_to_fraction(verts, faces, fraction)
        counts[fraction] = len(new_faces)
        assert abs(len(new_faces) - len(faces) * fraction) <= 0.15 * len(faces) * fraction, (fraction, len(new_faces))
        assert cell > 0
    assert counts[0.5] > counts[0.1] > counts[0.02] >= 4


@pytest.mark.skipif(not REAL_STL.exists(), reason="third_party 메시가 이 컴퓨터에 없음")
def test_real_stl_is_read_without_modifying_the_file():
    before = hashlib.sha256(REAL_STL.read_bytes()).hexdigest()
    verts, faces = probe.read_binary_stl(REAL_STL)
    assert len(faces) == 19080 and len(verts) < 3 * len(faces)
    probe.decimate_to_fraction(verts, faces, 0.1)
    assert hashlib.sha256(REAL_STL.read_bytes()).hexdigest() == before


@pytest.fixture(scope="module")
def base_model():
    return mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))


@pytest.fixture(scope="module")
def cache():
    return {}


def test_build_model_at_100_percent_matches_the_original(base_model, cache):
    model = probe.build_model(config.SCENE_PATH, 1.0, cache)
    assert (model.nmesh, model.ngeom, model.nq) == (base_model.nmesh, base_model.ngeom, base_model.nq)
    assert int(model.mesh_facenum.sum()) == int(base_model.mesh_facenum.sum())
    assert np.abs(model.mesh_pos - base_model.mesh_pos).max() < 1e-9  # 컴파일러의 다시 맞추기가 원본과 같다
    assert (model.vis.global_.offwidth, model.vis.global_.offheight) == (640, 480)


def test_build_model_reduces_only_visual_meshes(base_model, cache):
    model = probe.build_model(config.SCENE_PATH, 0.1, cache)
    assert (model.nmesh, model.ngeom) == (base_model.nmesh, base_model.ngeom)
    visual_before = probe.drawn_mesh_faces(base_model, probe.GROUPS_DEFAULT)
    visual_after = probe.drawn_mesh_faces(model, probe.GROUPS_DEFAULT)
    assert visual_after < 0.2 * visual_before
    # 충돌 그룹(3, 4)의 메시와 상자는 그대로
    assert probe.drawn_mesh_faces(model, (0, 0, 0, 0, 1, 0)) == probe.drawn_mesh_faces(base_model, (0, 0, 0, 0, 1, 0))
    assert np.array_equal(model.geom_size[model.geom_group == 3], base_model.geom_size[base_model.geom_group == 3])


def test_world_position_is_kept_when_meshes_are_reduced(base_model, cache, pose):
    """mesh_pos(컴파일러의 무게중심 다시 맞추기)가 달라져도 월드 위치는 같다. 변하는 것은 삼각형을 줄이며 깎이는 모양뿐이다.
    기준값은 2026-11-13 측정(중심 이동 / 가장자리 이동 최대): 100% 0 / 0 mm, 50% 0.004 / 0.008, 10% 0.72 / 2.0, 2% 2.8 / 7.5 mm."""
    full = probe.world_bbox_shift_mm(base_model, probe.build_model(config.SCENE_PATH, 1.0, cache), pose)
    assert full == {"center_max_mm": 0.0, "edge_max_mm": 0.0}
    half = probe.world_bbox_shift_mm(base_model, probe.build_model(config.SCENE_PATH, 0.5, cache), pose)
    assert half["edge_max_mm"] < 0.1
    tenth_model = probe.build_model(config.SCENE_PATH, 0.1, cache)
    tenth = probe.world_bbox_shift_mm(base_model, tenth_model, pose)
    assert tenth["center_max_mm"] < 1.5 and tenth["edge_max_mm"] < 3.0
    assert np.abs(tenth_model.mesh_pos - base_model.mesh_pos).max() > 1e-4  # mesh_pos는 실제로 달라졌는데도 월드 위치는 거의 같다(위 값)


def test_hidden_groups_change_the_drawn_face_count(base_model):
    default = probe.drawn_mesh_faces(base_model, probe.GROUPS_DEFAULT)
    assert default > 300_000
    assert probe.drawn_mesh_faces(base_model, probe.GROUPS_COLLISION) == 0  # 충돌 그룹 3에는 메시가 없다
    assert 0 < probe.drawn_mesh_faces(base_model, probe.GROUPS_COLLISION_PLUS_JAW) < 20_000
    assert probe.drawn_mesh_faces(base_model, probe.GROUPS_NO_ROBOT) == 0


@pytest.fixture(scope="module")
def pose():
    return probe.qpos_poses()[5]


def test_low_level_render_matches_the_renderer_path(base_model, pose):
    """저수준 경로(mjr_render 직접)가 aux_view.py가 쓰는 mujoco.Renderer 경로와 같은 그림을 낸다."""
    side = probe.camera_of(probe.VIEW_SIDE)
    ours = probe.render_image(base_model, side, pose, probe.GROUPS_DEFAULT, probe.VIEW_W, probe.VIEW_H)
    data = mujoco.MjData(base_model)
    data.qpos[:13] = pose
    mujoco.mj_forward(base_model, data)
    with mujoco.Renderer(base_model, probe.VIEW_H, probe.VIEW_W) as renderer:
        renderer.update_scene(data, camera=side)
        probe.set_render_flags(renderer, config.RENDER_SHADOW, config.RENDER_REFLECTION)
        renderer.render()  # 첫 렌더링은 버린다
        renderer.update_scene(data, camera=side)
        probe.set_render_flags(renderer, config.RENDER_SHADOW, config.RENDER_REFLECTION)
        reference = renderer.render()
    assert ours.shape == reference.shape == (probe.VIEW_H, probe.VIEW_W, 3)
    assert probe.mean_abs_diff(ours, reference) < 1.0
    assert ours.std() > 5


def test_two_viewports_in_one_buffer_match_two_separate_single_views(base_model, pose):
    side, oblique = probe.camera_of(probe.VIEW_SIDE), probe.camera_of(probe.VIEW_OBLIQUE)
    rend = probe.GLRenderer(base_model, probe.VIEW_W * 2, probe.VIEW_H)
    try:
        rend.frame(pose, [side, oblique], probe.GROUPS_DEFAULT, probe.VIEW_W)  # 첫 렌더링은 버린다
        both, parts = rend.frame(pose, [side, oblique], probe.GROUPS_DEFAULT, probe.VIEW_W)
    finally:
        rend.close()
    assert both.shape == (probe.VIEW_H, probe.VIEW_W * 2, 3)
    assert set(parts) == {"update", "render_read"} and all(v >= 0 for v in parts.values())
    left, right = both[:, : probe.VIEW_W], both[:, probe.VIEW_W :]
    assert probe.mean_abs_diff(left, right) > 5  # 두 시점은 서로 다른 그림이다
    for half, camera in ((left, side), (right, oblique)):
        single = probe.render_image(base_model, camera, pose, probe.GROUPS_DEFAULT, probe.VIEW_W, probe.VIEW_H)
        assert probe.mean_abs_diff(half, single) < 1.0


def test_probe_is_not_imported_by_training_recording_auto_or_eval_code():
    for filename in NOT_IMPORTING_AUX + ["app.py", "sim_worker.py", "control.py"]:
        path = SERVER_DIR / filename
        if path.exists():
            modules, _ = imported_and_used_names(path)
            assert not {m for m in modules if "aux_view_probe" in m}, filename
