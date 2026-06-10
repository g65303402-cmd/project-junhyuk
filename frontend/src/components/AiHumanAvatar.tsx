import { useEffect, useMemo, useRef, useState } from 'react';

import { fetchAvatarConfig } from '../api/client';

import {

  pickVideoUrl,

  resolveAvatarVisualState,

  type AvatarConfig,

  type AvatarDebugInfo,

  type AvatarVisualState,

} from '../types/avatar';



const EMPTY_CONFIG: AvatarConfig = {

  source: 'not-configured',

  hasVideos: false,

  idleVideos: [],

  speakingVideos: [],

  stateVideos: { userTyping: null, counselorConnect: null },

  fallbackMessage: '준혁이 준비 중이에요',

};



const DEV_FALLBACK =

  'SUPABASE_AVATAR_BASE_URL 또는 AVATAR_IDLE_VIDEO_URL / AVATAR_SPEAKING_VIDEO_URL 환경변수를 설정하세요.';



const USER_FALLBACK = '준혁이 기다리고 있어요';



export type AiHumanAvatarProps = {

  isSpeaking: boolean;

  isLoading: boolean;

  isUserTyping?: boolean;

  counselorConnect?: boolean;

  mode?: 'user' | 'dev';

  variant?: 'default' | 'hero';

  onDebugUpdate?: (info: AvatarDebugInfo) => void;

};



export function AiHumanAvatar({

  isSpeaking,

  isLoading,

  isUserTyping = false,

  counselorConnect = false,

  mode = 'user',

  variant = 'default',

  onDebugUpdate,

}: AiHumanAvatarProps) {

  const [config, setConfig] = useState<AvatarConfig>(EMPTY_CONFIG);

  const [videoError, setVideoError] = useState(false);

  const [failedUrl, setFailedUrl] = useState<string | null>(null);

  const [idleIndex, setIdleIndex] = useState(0);

  const [speakingIndex, setSpeakingIndex] = useState(0);

  const prevSpeakingRef = useRef(false);

  const prevVisualStateRef = useRef<AvatarVisualState>('idle');

  const videoRef = useRef<HTMLVideoElement>(null);

  const loadedUrlRef = useRef<string | null>(null);

  const retryCountRef = useRef(0);



  useEffect(() => {

    fetchAvatarConfig()

      .then(setConfig)

      .catch(() => setConfig(EMPTY_CONFIG));

  }, []);



  const visualState = resolveAvatarVisualState({

    counselorConnect,

    isSpeaking,

    isLoading,

    isUserTyping,

  });



  useEffect(() => {

    if (isSpeaking && !prevSpeakingRef.current && config.speakingVideos.length > 0) {

      setSpeakingIndex(Math.floor(Math.random() * config.speakingVideos.length));

    }

    prevSpeakingRef.current = isSpeaking;

  }, [isSpeaking, config.speakingVideos.length]);



  useEffect(() => {

    if (visualState === 'idle' && prevVisualStateRef.current !== 'idle') {

      // 말하기 종료 후 idle로 돌아올 때는 영상 URL을 바꾸지 않아 끊김을 줄인다.

      if (prevVisualStateRef.current !== 'speaking') {

        setIdleIndex((current) => (current + 1) % Math.max(config.idleVideos.length, 1));

      }

    }

    prevVisualStateRef.current = visualState;

  }, [visualState, config.idleVideos.length]);



  const activeVideoUrl = useMemo(

    () =>

      pickVideoUrl(config, visualState, {

        idle: idleIndex,

        speaking: speakingIndex,

      }),

    [config, visualState, idleIndex, speakingIndex],

  );



  useEffect(() => {

    setVideoError(false);

    setFailedUrl(null);

    retryCountRef.current = 0;

  }, [activeVideoUrl]);



  const showVideo = Boolean(activeVideoUrl) && !videoError;



  useEffect(() => {

    const video = videoRef.current;

    if (!video || !activeVideoUrl || !showVideo) return;

    if (loadedUrlRef.current === activeVideoUrl) {

      if (video.paused) {

        void video.play().catch(() => {});

      }

      return;

    }



    loadedUrlRef.current = activeVideoUrl;

    video.src = activeVideoUrl;

    video.load();

    void video.play().catch(() => {});

  }, [activeVideoUrl, showVideo]);



  useEffect(() => {

    onDebugUpdate?.({

      visualState,

      activeVideoUrl,

      videoError,

      idleCount: config.idleVideos.length,

      speakingCount: config.speakingVideos.length,

      hasUserTyping: Boolean(config.stateVideos.userTyping),

      hasCounselorConnect: Boolean(config.stateVideos.counselorConnect),

    });

  }, [

    visualState,

    activeVideoUrl,

    videoError,

    config,

    onDebugUpdate,

  ]);



  const statusLabel =

    visualState === 'counselorConnect'

      ? '연결 안내'

      : visualState === 'speaking'

        ? '말하는 중'

        : visualState === 'loading'

          ? '생각 중'

          : visualState === 'userTyping'

            ? '듣는 중'

            : '대기 중';



  function handleVideoError() {

    const video = videoRef.current;

    if (video && activeVideoUrl && retryCountRef.current < 1) {

      retryCountRef.current += 1;

      video.src = activeVideoUrl;

      video.load();

      void video.play().catch(() => {

        setVideoError(true);

        setFailedUrl(activeVideoUrl);

      });

      return;

    }



    setVideoError(true);

    setFailedUrl(activeVideoUrl);

  }



  return (

    <section

      className={`avatar-panel${variant === 'hero' ? ' avatar-panel--hero' : ''}`}

      aria-label="준혁 AI 휴먼"

    >

      <div className={`avatar-frame${variant === 'hero' ? ' avatar-frame--hero' : ''}`}>

        {visualState === 'loading' && mode === 'dev' && (

          <div className="avatar-status avatar-status--subtle">생각 중…</div>

        )}



        {showVideo ? (

          <video

            ref={videoRef}

            className={`avatar-video${variant === 'hero' ? ' avatar-video--hero' : ''}`}

            autoPlay

            loop

            muted

            playsInline

            preload="auto"

            onError={handleVideoError}

          />

        ) : (

          <div className={`avatar-fallback${variant === 'hero' ? ' avatar-fallback--hero' : ''}`}>

            <div className="avatar-fallback-icon" aria-hidden="true">

              준혁

            </div>

            {variant !== 'hero' && (

              <p>

                {mode === 'dev' && !config.hasVideos ? DEV_FALLBACK : USER_FALLBACK}

              </p>

            )}

            {mode === 'dev' && videoError && failedUrl && (

              <p className="avatar-fallback-path">로딩 실패: {failedUrl}</p>

            )}

            {mode === 'dev' && config.localPaths && !config.hasVideos && (

              <p className="avatar-fallback-path">

                legacy: {config.localPaths.idle}

                <br />

                {config.localPaths.speaking}

              </p>

            )}

          </div>

        )}

      </div>



      {mode === 'dev' && (

        <p className="avatar-state-label">

          {statusLabel}

          {activeVideoUrl ? ` · ${visualState}` : ''}

        </p>

      )}

    </section>

  );

}


