import os

# mujoco를 import하기 전에 정해야 한다. 이 서버는 GPU가 없고 Mesa EGL(CPU)로 렌더링된다.
os.environ.setdefault("MUJOCO_GL", "egl")
