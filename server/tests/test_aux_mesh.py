"""server/aux_mesh.py: 줄인 모델 만들기가 원본 파일과 원본 모델을 건드리지 않고, 학습·녹화·AUTO·평가 코드와 이어지지 않는다.
메시 줄이기 함수 자체의 동작 테스트는 test_aux_view_probe.py가 aux_view_probe.* 이름으로 확인한다(같은 함수라는 것은 아래에서 확인)."""

import hashlib

import mujoco
import pytest
from test_aux_isolation import NOT_IMPORTING_AUX, SERVER_DIR, imported_and_used_names

from server import aux_mesh, aux_view_probe, config

ASSET_DIR = config.REPO_ROOT / "third_party" / "mujoco_menagerie" / "robotstudio_so101" / "assets"


def test_probe_uses_the_same_functions_as_aux_mesh():
    for name in ("read_binary_stl", "cluster_decimate", "decimate_to_fraction", "build_model", "visual_mesh_names", "drawn_mesh_faces", "STL_RECORD"):
        assert getattr(aux_view_probe, name) is getattr(aux_mesh, name), name


@pytest.mark.skipif(not ASSET_DIR.exists(), reason="third_party 메시가 이 컴퓨터에 없음")
def test_build_model_does_not_modify_any_input_file_nor_the_original_model():
    files = sorted(ASSET_DIR.glob("*.stl")) + [config.SCENE_PATH, ASSET_DIR.parent / "so101.xml"]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    original_faces = int(mujoco.MjModel.from_xml_path(str(config.SCENE_PATH)).mesh_facenum.sum())

    reduced = aux_mesh.build_model(config.SCENE_PATH, 0.1)

    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files} == before
    assert int(reduced.mesh_facenum.sum()) < 0.2 * original_faces
    # 원래 장면을 새로 읽으면 그대로(줄인 모델은 메모리 안에서만 만들어진다. 시뮬레이션과 StateRenderer는 이 장면을 쓴다)
    assert int(mujoco.MjModel.from_xml_path(str(config.SCENE_PATH)).mesh_facenum.sum()) == original_faces


def test_aux_mesh_depends_on_nothing_of_the_project():
    modules, names = imported_and_used_names(SERVER_DIR / "aux_mesh.py")
    assert not {m for m in modules if m.startswith("server")}, modules  # server 안 다른 모듈을 import하지 않는다
    assert not (names & {"StateRenderer", "make_record_camera", "record_view"})


@pytest.mark.parametrize("filename", NOT_IMPORTING_AUX + ["sim_worker.py", "control.py", "owner.py", "view.py", "render.py"])
def test_training_recording_auto_eval_and_sim_code_does_not_import_aux_mesh(filename):
    path = SERVER_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} 없음")
    modules, _ = imported_and_used_names(path)
    offenders = {m for m in modules if "aux_mesh" in m or "aux_view_probe" in m}
    assert not offenders, f"{filename}이 aux_mesh를 import한다: {offenders}"


def test_aux_view_does_not_import_the_measurement_tool():
    modules, _ = imported_and_used_names(SERVER_DIR / "aux_view.py")
    assert "server.aux_mesh" in modules
    assert not {m for m in modules if "probe" in m or "bench" in m}
