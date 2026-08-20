# 백엔드 배포 (GPU 인스턴스, Ubuntu 24.04)

프론트엔드(GitHub Pages)는 그대로 두고, FastAPI 백엔드만 이 인스턴스에서 Docker로 띄운다.
열려 있는 포트는 22/80/443뿐이므로 nginx가 80/443을 받아 컨테이너의 8000번으로 프록시한다.

> **시작하기 전에**: 강사/보조 강사에게 이 인스턴스의 **Elastic IP(고정 IP) 배정을 요청**해두자.
> 기본으로는 안 걸어주지만 필요한 팀에는 설정해준다고 확인받았다 — 고정 IP가 있어야 아래
> nip.io 도메인이 재시작해도 안 바뀐다(3~4단계 참고).

## 1. 최초 1회 설정

```bash
# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER   # 이후 재접속 필요

# nginx + certbot
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx

# 저장소
git clone https://github.com/Alpha-pick00/Alpha-pick00.github.io.git alpha-pick
cd alpha-pick/backend
cp .env.example .env
vi .env   # 실제 API 키 채워넣기 (이 파일은 git에 올라가지 않음)
```

## 2. 백엔드 실행

```bash
cd deploy
docker compose up -d --build
curl http://127.0.0.1:8000/health   # {"status":"ok"} 확인
```

## 3. nginx + TLS (nip.io — 도메인 구매/DNS 관리 없이 바로 사용)

도메인을 따로 사지 않고, IP를 그대로 호스트명으로 바꿔주는 무료 서비스인
[nip.io](https://nip.io)를 쓴다. `1-2-3-4.nip.io`는 자동으로 `1.2.3.4`를 가리키므로
계정 생성도, DNS 레코드 설정도 필요 없다.

```bash
# 1) 이 인스턴스의 퍼블릭 IP 확인
PUBLIC_IP=$(curl -4 -s ifconfig.me)
DOMAIN="$(echo $PUBLIC_IP | tr '.' '-').nip.io"
echo "$DOMAIN"   # 예: 3-38-123-45.nip.io — 이 값을 기억해둔다

# 2) nginx 설정
sudo cp alpha-pick-api.nginx.conf /etc/nginx/sites-available/alpha-pick-api
sudo sed -i "s/YOUR_DOMAIN/$DOMAIN/" /etc/nginx/sites-available/alpha-pick-api
sudo ln -s /etc/nginx/sites-available/alpha-pick-api /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 3) TLS 인증서 발급 (443 블록을 certbot이 자동으로 추가)
sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m you@example.com
```

배포 후 `https://$DOMAIN/health`가 응답하면 완료. 이 `https://$DOMAIN`이
프론트엔드의 `VITE_API_URL`(GitHub 저장소 Settings → Secrets and variables →
Actions → Variables)에 넣을 값이다.

## 4. 인스턴스 재시작과 Elastic IP

강사/보조 강사 확인 결과 **Elastic IP(고정 IP)는 기본으로는 안 걸어주지만, 필요한 팀에는
요청하면 배정해준다**. 이 배포는 고정 IP가 있다는 걸 전제로 설계했다 — 먼저 요청해서
받아두자.

- **고정 IP를 받은 경우 (전제 조건)**: 인스턴스를 껐다 켜도 퍼블릭 IP가 그대로이므로,
  위 3단계에서 만든 `$DOMAIN`(nip.io)과 발급받은 TLS 인증서, 프론트엔드의 `VITE_API_URL`
  전부 그대로 유효하다. 재시작할 때마다 다시 할 일이 없다 — `docker compose up -d`로
  컨테이너만 다시 띄우면 끝.
- **혹시 고정 IP 없이 진행해야 한다면**: 재시작마다 IP가 바뀌므로 nip.io 도메인도 매번
  바뀐다. 그때는 3단계를 처음부터 다시 실행해서 새 도메인으로 nginx/인증서를 재설정하고,
  GitHub 저장소의 `VITE_API_URL`도 새 도메인으로 갱신한 뒤 Actions를 재실행해야 한다.

## 5. 코드 업데이트할 때

```bash
cd ~/alpha-pick
git pull
cd backend/deploy
docker compose up -d --build
```

## 6. CORS

`app/main.py`에 `https://alpha-pick00.github.io`와 로컬 개발 origin만 허용해뒀다.
커스텀 도메인으로 프론트를 옮기면 `allow_origins`에 그 도메인도 추가해야 브라우저에서 호출 가능.
