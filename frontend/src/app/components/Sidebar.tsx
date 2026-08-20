import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { formatDistanceToNow } from 'date-fns';
import { ko } from 'date-fns/locale';
import { PanelLeft, X, Plus, Search, Trash2, LogOut, UserRound, Settings } from 'lucide-react';
import { useSearch } from '../context/SearchContext';
import { useAuth } from '../context/AuthContext';
import { useSidebar } from '../context/SidebarContext';
import { startGoogleLogin, startKakaoLogin, startNaverLogin } from '../lib/auth';

const GoogleIcon = () => (
  <svg viewBox="0 0 24 24" className="w-4 h-4">
    <path
      fill="#4285F4"
      d="M23.52 12.27c0-.85-.08-1.67-.22-2.45H12v4.64h6.47a5.53 5.53 0 0 1-2.4 3.63v3h3.88c2.27-2.09 3.57-5.17 3.57-8.82Z"
    />
    <path
      fill="#34A853"
      d="M12 24c3.24 0 5.96-1.07 7.95-2.91l-3.88-3c-1.08.72-2.45 1.15-4.07 1.15-3.13 0-5.78-2.11-6.73-4.96H1.26v3.09A11.997 11.997 0 0 0 12 24Z"
    />
    <path
      fill="#FBBC05"
      d="M5.27 14.28A7.2 7.2 0 0 1 4.89 12c0-.79.14-1.56.38-2.28V6.63H1.26A11.997 11.997 0 0 0 0 12c0 1.94.46 3.77 1.26 5.37l4.01-3.09Z"
    />
    <path
      fill="#EA4335"
      d="M12 4.77c1.76 0 3.35.61 4.6 1.8l3.44-3.44C17.95 1.19 15.24 0 12 0 7.31 0 3.26 2.69 1.26 6.63l4.01 3.09C6.22 6.87 8.87 4.77 12 4.77Z"
    />
  </svg>
);

const KakaoIcon = () => (
  <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor">
    <path d="M12 3C6.48 3 2 6.48 2 10.8c0 2.76 1.85 5.19 4.63 6.58-.2.75-.73 2.73-.84 3.15-.13.52.19.51.4.37.17-.11 2.65-1.8 3.73-2.53.68.1 1.38.15 2.08.15 5.52 0 10-3.48 10-7.72C22 6.48 17.52 3 12 3z" />
  </svg>
);

const NaverIcon = () => (
  <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor">
    <path d="M16.3 12.6 8.6 2H3v20h5.7V11.4l7.7 10.6H21V2h-4.7z" />
  </svg>
);

const SOCIAL_BUTTON_CLASS =
  'w-full flex items-center justify-center gap-2 h-10 rounded-xl text-sm font-medium transition-opacity hover:opacity-90';

const AuthArea = () => {
  const { user, loading, setUser, logout } = useAuth();

  const handleGoogleLogin = async () => {
    try {
      setUser(await startGoogleLogin());
    } catch {
      // 로그인 실패/취소는 조용히 무시 — 버튼이 그대로 남아있어 재시도 가능
    }
  };

  if (loading) {
    return <div className="h-12 rounded-xl bg-neutral-100 animate-pulse" />;
  }

  if (user) {
    const initial = (user.name || user.email || '?').charAt(0).toUpperCase();
    return (
      <div className="flex items-center gap-3 p-2 rounded-xl hover:bg-neutral-100 transition-colors">
        {user.picture ? (
          <img src={user.picture} alt="" className="w-9 h-9 rounded-full object-cover shrink-0" />
        ) : (
          <div className="w-9 h-9 rounded-full bg-neutral-950 text-white flex items-center justify-center text-sm font-medium shrink-0">
            {initial}
          </div>
        )}
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-neutral-950 truncate">{user.name || '사용자'}</p>
          <p className="text-xs font-light text-neutral-500 truncate">{user.email || user.provider}</p>
        </div>
        <button
          type="button"
          onClick={logout}
          aria-label="로그아웃"
          className="shrink-0 p-2 text-neutral-400 hover:text-neutral-950 transition-colors"
        >
          <LogOut className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <button
        type="button"
        onClick={handleGoogleLogin}
        className={`${SOCIAL_BUTTON_CLASS} border border-black/10 bg-white text-neutral-700 hover:bg-neutral-50`}
      >
        <GoogleIcon />
        Google로 계속하기
      </button>
      <button
        type="button"
        onClick={startKakaoLogin}
        className={SOCIAL_BUTTON_CLASS}
        style={{ backgroundColor: '#FEE500', color: '#191919' }}
      >
        <KakaoIcon />
        카카오로 계속하기
      </button>
      <button
        type="button"
        onClick={startNaverLogin}
        className={`${SOCIAL_BUTTON_CLASS} text-white`}
        style={{ backgroundColor: '#03C75A' }}
      >
        <NaverIcon />
        네이버로 계속하기
      </button>
    </div>
  );
};

export const Sidebar = () => {
  const { isOpen, open: openSidebar, close, toggle } = useSidebar();
  const [historySearch, setHistorySearch] = useState('');
  const { history, loadFromHistory, deleteFromHistory, clearAllHistory, handleReset } = useSearch();
  const { user } = useAuth();

  const openWithAction = (action: () => void) => {
    action();
    closeSidebar();
  };

  const closeSidebar = () => {
    close();
    setHistorySearch('');
  };

  const filteredHistory = historySearch.trim()
    ? history.filter((entry) => entry.query.toLowerCase().includes(historySearch.trim().toLowerCase()))
    : history;

  // 레일(아이콘만)과 펼쳐진 패널(검색+기록)이 예전엔 서로 다른 두 개의 fixed 엘리먼트라
  // 열렸을 때 둘 다 동시에 화면에 남아있었다 - "사이드바 열면 사이드바가 또 생긴다"던
  // 사용자 보고(2026-08-12)의 실제 원인. 아이콘 열/기록 목록을 하나의 <aside> 안에
  // 같이 두고 너비만 68px <-> 368px로 애니메이션하는 식으로 합쳐서, 화면에는 항상
  // "사이드바 하나"만 존재하게 했다.
  const panelBody = (
    <>
      <div className="flex items-center justify-between px-5 pt-6 pb-4">
        <span
          className="text-2xl font-medium tracking-tighter"
          style={{ fontFamily: "'Times New Roman', Times, serif", color: 'rgb(64,117,38)' }}
        >
          αlpha Pick
        </span>
        <button
          type="button"
          onClick={closeSidebar}
          aria-label="닫기"
          className="p-1.5 text-neutral-400 hover:text-neutral-950 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="px-3">
        <button
          type="button"
          onClick={() => openWithAction(handleReset)}
          className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-black/10 text-sm font-light text-neutral-700 hover:bg-neutral-100 transition-colors"
        >
          <Plus className="w-4 h-4" />
          새 검색
        </button>
      </div>

      <div className="px-3 mt-2">
        <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-neutral-100">
          <Search className="w-4 h-4 text-neutral-400 shrink-0" strokeWidth={2} />
          <input
            type="text"
            value={historySearch}
            onChange={(e) => setHistorySearch(e.target.value)}
            placeholder="채팅 검색"
            className="flex-1 min-w-0 bg-transparent text-sm font-light text-neutral-800 placeholder:text-neutral-400 outline-none"
          />
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto">
        {/* 대화(conversations) 목록은 따로 안 보여준다(사용자 요청, 2026-08-15:
            "사이드바 열면 대화창이 새롭게 생겼는데 그냥 원래 있던 기록 창에
            냅둬") - "기록" 하나로만 보여준다. */}
        <div className="mt-4 px-5">
          <span className="text-[11px] font-mono uppercase tracking-widest text-neutral-400">기록</span>
        </div>
        <div className="px-3 mt-2 space-y-1">
        {filteredHistory.length === 0 ? (
          <p className="px-2 py-4 text-sm font-light text-neutral-400">
            {history.length === 0 ? '아직 검색 기록이 없습니다.' : '일치하는 기록이 없습니다.'}
          </p>
        ) : (
          filteredHistory.map((entry) => (
            <div
              key={entry.id}
              className="group flex items-center gap-1 rounded-lg hover:bg-neutral-100 transition-colors"
            >
              <button
                type="button"
                onClick={() => openWithAction(() => loadFromHistory(entry))}
                className="flex-1 min-w-0 text-left px-2.5 py-2"
              >
                <p className="text-sm font-light text-neutral-800 truncate">{entry.query}</p>
                <p className="text-[11px] text-neutral-400">
                  {formatDistanceToNow(entry.timestamp, { addSuffix: true, locale: ko })}
                </p>
              </button>
              <button
                type="button"
                onClick={() => deleteFromHistory(entry.id)}
                aria-label="기록 삭제"
                className="shrink-0 p-2 mr-1 text-neutral-300 opacity-0 group-hover:opacity-100 hover:text-neutral-950 transition-all"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))
        )}
        </div>
      </div>

      {user && (
        <>
          <div className="px-5 pt-3 pb-1">
            <span className="text-[11px] font-mono uppercase tracking-widest text-neutral-400">환경설정</span>
          </div>
          <div className="px-3 pb-4">
            <button
              type="button"
              onClick={clearAllHistory}
              disabled={history.length === 0}
              className="w-full text-left px-2.5 py-2 rounded-lg text-sm font-light text-neutral-500 hover:bg-neutral-100 hover:text-neutral-950 transition-colors disabled:opacity-40 disabled:hover:bg-transparent"
            >
              전체 기록 삭제
            </button>
          </div>
        </>
      )}

      <div className="border-t border-black/5 p-3">
        <AuthArea />
      </div>
    </>
  );

  const iconRail = (
    <div className="w-[68px] h-full shrink-0 flex flex-col items-center py-5 gap-2">
      <button
        type="button"
        onClick={toggle}
        aria-label={isOpen ? '기록 닫기' : '기록 열기'}
        className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors ${
          isOpen ? 'bg-neutral-100 text-neutral-950' : 'text-neutral-700 hover:bg-neutral-100'
        }`}
      >
        <PanelLeft className="w-5 h-5" strokeWidth={2} />
      </button>
      <button
        type="button"
        onClick={() => openWithAction(handleReset)}
        aria-label="새 검색"
        className="w-10 h-10 rounded-xl flex items-center justify-center text-neutral-700 hover:bg-neutral-100 transition-colors"
      >
        <Plus className="w-5 h-5" strokeWidth={2} />
      </button>

      <div className="flex-1" />

      {user && (
        <button
          type="button"
          onClick={openSidebar}
          aria-label="설정"
          className="w-10 h-10 rounded-xl flex items-center justify-center text-neutral-700 hover:bg-neutral-100 transition-colors"
        >
          <Settings className="w-5 h-5" strokeWidth={2} />
        </button>
      )}

      <button
        type="button"
        onClick={openSidebar}
        aria-label={user ? '계정' : '로그인'}
        className="w-10 h-10 rounded-full overflow-hidden flex items-center justify-center hover:opacity-80 transition-opacity"
      >
        {user ? (
          user.picture ? (
            <img src={user.picture} alt="" className="w-full h-full object-cover" />
          ) : (
            <div className="w-full h-full rounded-full bg-neutral-950 text-white flex items-center justify-center text-sm font-medium">
              {(user.name || user.email || '?').charAt(0).toUpperCase()}
            </div>
          )
        ) : (
          <div className="w-full h-full rounded-full bg-neutral-100 flex items-center justify-center text-neutral-400">
            <UserRound className="w-5 h-5" />
          </div>
        )}
      </button>
    </div>
  );

  return (
    <>
      {/* 모바일 전용 플로팅 토글 — 작은 화면에서는 레일이 화면을 너무 많이 잡아먹으므로
          md 이상에서만 레일을 보여주고, 그 아래에서는 이 버튼 하나로 대체한다. */}
      <button
        type="button"
        onClick={toggle}
        aria-label="기록 열기"
        className={`md:hidden fixed top-4 left-4 z-[60] w-11 h-11 rounded-full border border-black/10 bg-white/80 backdrop-blur-md shadow-[0_4px_20px_rgba(0,0,0,0.06)] flex items-center justify-center text-neutral-700 hover:bg-white transition-all ${isOpen ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}
      >
        <PanelLeft className="w-5 h-5" strokeWidth={2} />
      </button>

      {/* 데스크톱/태블릿 — 아이콘 레일과 펼쳐지는 패널이 하나의 aside다. 닫혀있으면
          68px 너비에 아이콘 레일만, 열리면 280px 너비에 기록 패널만 보인다 - 아이콘
          레일과 패널을 동시에 보여주면 그 자체가 "사이드바가 두 개"처럼 보인다는
          지적(2026-08-12)이 있어서 열렸을 때는 레일을 아예 렌더링하지 않는다. 화면을
          어둡게 덮는 backdrop도 없다 - 모달이 아니라 옆에 도킹되는 패널이라, 열려
          있는 동안에도 본문(검색 등)을 그대로 쓸 수 있어야 한다. 대신 App.tsx가
          isOpen을 보고 본문 padding-left를 넓혀서 자리를 비켜준다. */}
      <motion.aside
        initial={false}
        animate={{ width: isOpen ? 280 : 68 }}
        transition={{ type: 'tween', duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
        className="hidden md:flex md:flex-col fixed inset-y-0 left-0 z-[60] bg-white/95 backdrop-blur-md border-r border-black/10 overflow-hidden"
      >
        {isOpen ? panelBody : iconRail}
      </motion.aside>

      {/* 모바일 — 레일이 따로 없으니 열렸을 때 화면 대부분을 덮는 패널 하나만 쓴다. */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ x: '-100%' }}
            animate={{ x: 0 }}
            exit={{ x: '-100%' }}
            transition={{ type: 'tween', duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            className="md:hidden fixed inset-y-0 left-0 z-[65] w-[85vw] max-w-[300px] bg-white border-r border-black/10 flex flex-col shadow-[8px_0_30px_rgba(0,0,0,0.06)]"
          >
            {panelBody}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};
