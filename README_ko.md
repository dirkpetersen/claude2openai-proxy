# claude-to-openai 프록시

## 개요
Anthropic 스타일의 /v1/messages 요청을 OpenAI 형식으로 변환해 LiteLLM을 통해 호출하는 FastAPI 프록시입니다. 모델 매핑, 컨텐츠 블록 변환, 스트리밍(SSE), 툴 호출, 토큰 카운트, 저소음 로깅을 지원합니다.

- 원본 코드: https://github.com/1rgs/claude-code-proxy
- 범용 사용을 위해 수정됨

## 특징
- 환경변수와 접두사 기반 모델 매핑(anthropic/* 패스스루, openai/* 정규화)
- Anthropic 컨텐츠 블록(text/image/tool_use/tool_result) → OpenAI 메시지 변환
- Anthropic 호환 SSE 스트리밍(message_start, content_block_* 이벤트)
- 툴 호출 패스스루 및 응답을 Anthropic tool_use 블록으로 변환
- LiteLLM token_counter 기반 토큰 카운트 엔드포인트
- 불필요한 로그 필터링과 보기 좋은 요청 로그

## 실행
```
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
```

## 컨테이너

### 빌드
```
docker build -t claude-proxy .
```

### 실행
```
docker run -d \
  --name claude-proxy \
  -p 8082:8082 \
  -e OPENAI_API_BASE="http://10.2.2.10:4000/v1" \
  -e BIG_MODEL=gpt-4.1 \
  -e SMALL_MODEL=gpt-4.1-mini \
  -e MAX_TOKENS=65535 \
  -e BIG_PREFIXES="opus,sonnet" \
  -e SMALL_PREFIXES="haiku" \
  claude-proxy
```

### Docker Compose
docker-compose.yml:
```
version: "3.8"
services:
  claude-proxy:
    image: claude-proxy
    container_name: claude-proxy
    ports:
      - "8082:8082"
    environment:
      OPENAI_API_BASE: "http://10.2.2.10:4000/v1"
      BIG_MODEL: "gpt-4.1"
      SMALL_MODEL: "gpt-4.1-mini"
      MAX_TOKENS: "65535"
      BIG_PREFIXES: "opus,sonnet"
      SMALL_PREFIXES: "haiku"
    restart: unless-stopped
```
실행:
```
docker compose up -d
```
## 사용 예시

- Claude Code
```
ANTHROPIC_BASE_URL=http://localhost:8082 \
ANTHROPIC_API_KEY=sk-keyhere \
ANTHROPIC_MODEL="big-size-model" \
ANTHROPIC_SMALL_FAST_MODEL="small-size-model" \
claude
```

## 환경변수
- OPENAI_API_KEY: OpenAI 키(X-API-Key 헤더로 전달 가능)
- BIG_MODEL: 기본 대형 모델(기본값: gpt-4.1)
- SMALL_MODEL: 기본 소형 모델(기본값: gpt-4.1-mini)
- BIG_PREFIXES: 접두사를 BIG_MODEL로 매핑(기본값: opus,sonnet)
- SMALL_PREFIXES: 접두사를 SMALL_MODEL로 매핑(기본값: haiku)
- MAX_TOKENS: OpenAI 모델 max_tokens 상한(기본값: 65535)

## 모델 결정 로직
- 요청 body.model이 있으면 그대로 사용(필요 시 provider 접두사 보강)
- 없으면 body.size=="small"일 때 openai/SMALL_MODEL, 그 외 openai/BIG_MODEL
- OpenAI의 알려진 모델 ID가 접두사 없이 오면 openai/<model>로 보강

## 엔드포인트
- POST /v1/messages
  - 요청: Anthropic 유사 포맷 { model, max_tokens, messages, system, tools, tool_choice, temperature, stream }
  - 동작: OpenAI/LiteLLM 포맷으로 변환, OpenAI 모델 대상 max_tokens를 MAX_TOKENS로 캡
  - 스트리밍: stream=true 시 Anthropic SSE 이벤트 전송(message_start, content_block_start/delta/stop, message_delta, message_stop)
- POST /v1/messages/count_tokens
  - 응답: { input_tokens }
- GET /
  - 응답: { message }

## 컨텐츠 변환 주의사항
- user/assistant 메시지는 문자열 또는 컨텐츠 블록 허용; OpenAI로 보낼 때는 텍스트로 평탄화
- user의 tool_result 블록은 평문 텍스트 세그먼트로 변환
- 이미지 블록은 텍스트 전용 엔드포인트에선 플레이스홀더 텍스트로 처리
- OpenAI API가 허용하지 않는 필드는 제거

## 인증/헤더
- X-API-Key: <OpenAI 키>
- 없으면 401 반환

## 예제
단순 완료:
```
curl -sS -X POST http://localhost:8082/v1/messages \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $OPENAI_API_KEY" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Say hi"}
    ]
  }'
```

스트리밍:
```
curl -N -sS -X POST http://localhost:8082/v1/messages \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $OPENAI_API_KEY" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "stream": true,
    "max_tokens": 64,
    "messages": [
      {"role": "user", "content": "Stream a short sentence"}
    ]
  }'
```

토큰 카운트:
```
curl -sS -X POST http://localhost:8082/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $OPENAI_API_KEY" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "How many tokens?"}
    ]
  }'
```

## 로깅
- 전역 WARN 레벨, uvicorn access/error 억제
- LiteLLM 등 노이즈 로그 필터링
- Anthropic→OpenAI 매핑과 카운트를 요약한 예쁜 요청 로그 출력

## 보안 주의사항
- 비밀값 로깅 금지; API 키는 요청 헤더에서 읽어 호출 동안 프로세스 env에 설정 후 복원
- 가능한 입력 검증; 상위 API가 허용하지 않는 필드는 업스트림 호출 전 제거
