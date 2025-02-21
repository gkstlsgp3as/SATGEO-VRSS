# Docker 사용법

### a. Dockerfile Build/Run

#### Docker Build - Dockerfile

```bash
#!/bin/bash
# 현재 일시 YYYYmmdd-HHMMSS
TAG_DATE=$(date +%Y%m%d-%H%M%S)
# Docker 이미지 빌드 및 태그 지정
docker build -t app:$TAG_DATE .
# -t: tag, . 현재 경로에서 dockerfile 실행
docker build -t app:$TAG_DATE -f Dockerfile ```

#### 스크립트 권한 변경
```bash
chmod +x build.sh
```

#### 스크립트 실행
```bash
chmod +x build.sh
```

#### 컨테이너 실행
```bash
docker images # Docker 이미지 확인
docker run -v "Document/data:/root/project/data" -d -ti app:1.0 /bin/bash

```
- d: Detached mode, 백그라운드 모드에서 컨테이너를 실행
- ti: 대화형 모드에서 컨테이너를 실행하여 터미널의 입력을 컨테이너로 전달 
- v: 외부 볼륨을 컨테이너에 마운트
- /bin/bash: 컨테이너 내에서 Bash 환경 실행

---
### b. Connection to IDE

#### VS Code
1. 마켓플레이스 Docker 확장 사용
   - Docker 컨테이너에 연결
     참고: [https://with-rl.tistory.com/entry/VSCode에서-docker에-접속하기](https://with-rl.tistory.com/entry/VSCode%EC%97%90%EC%84%9C-docker%EC%97%90-%EC%A0%91%EC%86%8D%ED%95%98%EA%B8%B0)
   - remote development 설치 > 좌측 remote explore > dev containers > attach
     - dev containers에 가동 중인 이미지가 표출되지 않는다면 ssh 연결 확인
     - F1 → Remote-SSH: Connect to Host → image를 가동한 서버로 연결
     - ssh 연결 안되면 C:/Users/user/.ssh/known_hosts 삭제
       
2. 외부 데이터 마운트
   - 외부 데이터를 컨테이너에 마운트하여 작업 가능
     - docker run 시 -v로 원하는 외부 데이터 마운트
```bash
docker build -t app:1.0 .
docker run -d -ti -p 8888:8888 app:1.0

# VS Code attach + repo clone
# VS Code 종료

docker stop app:1.0
docker run -v "/path/to/mount/data/:/root/data" -d -t -i -p 8888:8888 app:1.0     # /root/data 하위에 외부 데이터 마운트
```

#### Spyder IDE
1. Spyder 태그 생성/Build
   - Dockerfile에 아래 내용을 추가하여 Spyder를 설치
```dockerfile
# Spyder 설치
RUN pip install spyder

# 작업 디렉터리 설정
WORKDIR /workspace

# Spyder 실행
CMD ["spyder"]
```

2. Docker Build 및 실행
```bash
docker build -t app:spyder .
docker run -v "Document/data:/root/code" -d -t -p 8888:8888 app:spyder
```
