// 메인 화면 구석의 진짜 보조 카메라 화면 두 개(2-6, 보기 전용): 서버가 두 시점을 가로로 이어 보낸 JPEG 한 장을 CSS로 둘로 나눠 보여 준다.
// 옆에서 = 왼쪽 아래, 비스듬히 = 오른쪽 아래. 크기·위치·순서는 aux/config.js 한 곳, 스타일 계산은 aux/view.js(순수 함수).
// 클릭을 받지 않는다(메인 화면 끌기와 휠을 방해하지 않게). 창이 좁으면 index.css가 숨긴다.

import { AUX_TILES } from '../aux/config.js'
import { auxTileStyle } from '../aux/view.js'

// frame: App이 받은 마지막 aux_frame({ src, views, ... }). 없으면 아무것도 그리지 않는다.
export default function AuxMirrors({ frame }) {
  if (!frame) return null
  return (
    <div className="aux-mirrors" role="group" aria-label="보조 카메라 화면 두 개: 옆에서와 비스듬히">
      {AUX_TILES.map((tile, index) => (
        <div key={tile.id} className="aux-tile" style={auxTileStyle(tile, index, frame)} title={`${tile.label} (보기 전용)`}>
          <span className="aux-tile-label">{tile.label}</span>
        </div>
      ))}
    </div>
  )
}
