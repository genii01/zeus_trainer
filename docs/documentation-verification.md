# HTML 설명 문서 검증

검증일: 2026-09-14. 문서는 현재 Compose와 Python 구현을 읽고 작성했습니다.

- `python3 scripts/check_docs.py`: HTML 내부 anchor 13개와 링크·asset 18개 확인, 배포 디렉터리 밖의 상대 링크 없음.
- `node --check docs/assets/app.js`: 문법 검사 통과.
- Chrome headless 1440×1000 / 390×844: 페이지 가로 넘침 없음.
- 경로 강조 버튼: 선택한 경로 외 2개가 흐려지고 전체 보기로 복원.
- 구성 요소 10개: 선택 시 역할·주소·설정 파일 상세 변경 확인.
- 운영 목차 anchor와 접기/펼치기 동작 확인. 브라우저 JavaScript 오류 없음.
- `.venv/bin/python -m pytest -q`: 기존 테스트 37개 통과 (0.91s).
- `docker compose config --quiet`: 구성 검사 통과.

GitHub Actions의 PR 검증은 문서 링크와 JS 문법을 검사합니다. Pages 배포는 master에서만 실행하도록 제한했습니다. 로컬 검증과 실제 Pages 배포 성공은 별개이며, 원격 실행 결과는 PR/Actions에서 확인합니다.

이번 문서 작업에서는 데이터를 삭제하는 `down -v`를 실행하지 않았습니다. 종료·재실행 명령과 볼륨 보존 설명은 Compose 설정을 기준으로 작성했습니다. 기존 추론·로그 복구 검증의 실측 결과는 [서버 검증 보고서](verification.md)를 참고하세요.
