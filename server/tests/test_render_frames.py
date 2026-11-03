"""3-3(a) 프레임 렌더링: 프레임 k가 qpos[k]로 그려지는지, 저장이 안전한지, 이어하기 판정이 맞는지."""

import mujoco
import numpy as np
import pytest
from PIL import Image

from server import config
from server.render import StateRenderer
from server.render_frames import (
    IMAGE_FORMAT,
    episode_dir_name,
    episode_is_done,
    frame_path,
    qpos_sha256,
    render_episode,
    render_settings,
)


def assert_images_close(a: np.ndarray, b: np.ndarray, max_diff: int = 1) -> None:
    assert a.shape == b.shape
    worst = int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())
    assert worst <= max_diff, f"이미지 최대 차이 {worst} > {max_diff}"


@pytest.fixture(scope="module")
def renderer():
    with StateRenderer() as r:
        yield r


@pytest.fixture(scope="module")
def tiny_qpos():
    """블록 위치가 프레임마다 다른 가짜 qpos 4개(집게는 home 자세 그대로)."""
    model = mujoco.MjModel.from_xml_path(str(config.SCENE_PATH))
    home = model.key("home").qpos.copy()
    out = np.tile(home, (4, 1))
    for k in range(4):
        out[k, 6:9] = (0.19 + 0.02 * k, -0.08 + 0.03 * k, 0.015)
    return out


def test_frame_k_is_rendered_from_qpos_k(tmp_path, renderer, tiny_qpos):
    render_episode(renderer, tiny_qpos, tmp_path, 7)
    images = [np.asarray(Image.open(frame_path(tmp_path, 7, k))) for k in range(len(tiny_qpos))]
    for k, qpos in enumerate(tiny_qpos):
        # PNG는 무손실이다. 다만 렌더러 인스턴스의 첫 렌더링만 이후 렌더링과 몇 픽셀(실측 7/76800)이 1단계(0~255) 다르다
        # (새 렌더러로 그리면 첫 결과와 같다. 깊은 원인은 확인하지 못했다). 그래서 최대 오차 1까지 허용한다.
        assert_images_close(images[k], renderer.render(qpos))
    for a in range(len(images)):  # 블록이 다르니 프레임끼리 모두 다르다(순서가 섞여도 걸린다)
        for b in range(a + 1, len(images)):
            assert not np.array_equal(images[a], images[b])


def test_episode_is_written_atomically(tmp_path, renderer, tiny_qpos):
    render_episode(renderer, tiny_qpos, tmp_path, 3)
    assert (tmp_path / episode_dir_name(3)).is_dir()
    assert not (tmp_path / f"{episode_dir_name(3)}.tmp").exists()
    assert len(list((tmp_path / episode_dir_name(3)).glob(f"*.{IMAGE_FORMAT}"))) == len(tiny_qpos)


def test_leftover_tmp_folder_is_replaced(tmp_path, renderer, tiny_qpos):
    stale = tmp_path / f"{episode_dir_name(5)}.tmp"
    stale.mkdir()
    (stale / "frame_999999.png").write_bytes(b"x")
    render_episode(renderer, tiny_qpos, tmp_path, 5)
    assert not stale.exists()
    assert not (tmp_path / episode_dir_name(5) / "frame_999999.png").exists()


def test_qpos_hash_changes_with_any_value(tiny_qpos):
    base = qpos_sha256(tiny_qpos)
    assert base == qpos_sha256(tiny_qpos.copy())
    changed = tiny_qpos.copy()
    changed[2, 4] += 1e-9
    assert qpos_sha256(changed) != base


def test_episode_is_done_requires_matching_hash_and_files(tmp_path, renderer, tiny_qpos):
    render_episode(renderer, tiny_qpos, tmp_path, 1)
    manifest = {"episodes": {"1": {"n_frames": 4, "qpos_sha256": qpos_sha256(tiny_qpos)}}}
    assert episode_is_done(manifest, tmp_path, 1, tiny_qpos)
    assert not episode_is_done(manifest, tmp_path, 2, tiny_qpos)  # manifest에 없음
    other = tiny_qpos.copy()
    other[0, 6] += 0.01
    assert not episode_is_done(manifest, tmp_path, 1, other)  # qpos가 바뀜 → 다시 그려야 함
    frame_path(tmp_path, 1, 2).unlink()
    assert not episode_is_done(manifest, tmp_path, 1, tiny_qpos)  # 파일이 모자람


def test_render_settings_match_record_camera_config():
    s = render_settings()
    assert (s["height_px"], s["width_px"]) == (config.RECORD_IMAGE_HEIGHT_PX, config.RECORD_IMAGE_WIDTH_PX)
    assert s["camera"]["azimuth_deg"] == config.RECORD_CAMERA_AZIMUTH_DEG
    assert s["shadow"] is False and s["reflection"] is False
    assert s["image_format"] == "png"


@pytest.mark.skipif(not (config.RECORD_DIR / "episode_000007.npz").exists(), reason="data/raw가 이 컴퓨터에 없음")
def test_real_episode_frames_match_render_of_same_qpos(renderer):
    """실제 data/frames의 프레임이 data/raw의 같은 번호 qpos로 다시 그린 것과 정확히 같다(렌더링이 끝났을 때만)."""
    frames_dir = config.REPO_ROOT / "data" / "frames"
    if not frame_path(frames_dir, 7, 0).exists():
        pytest.skip("data/frames가 아직 없음")
    from server.episode_store import EpisodeStore

    qpos = EpisodeStore().load_arrays(7)["qpos"]
    for k in (0, len(qpos) // 2, len(qpos) - 1):
        assert_images_close(np.asarray(Image.open(frame_path(frames_dir, 7, k))), renderer.render(qpos[k]))
