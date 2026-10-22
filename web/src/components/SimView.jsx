// 서버가 보낸 시뮬레이션 화면 표시. url은 JPEG Blob 주소(App이 만들고 해제한다).

export default function SimView({ url }) {
  return (
    <div className="sim-view">
      {url ? <img src={url} alt="시뮬레이션 top 카메라" /> : <div className="placeholder">시뮬레이션 화면 기다리는 중</div>}
    </div>
  )
}
