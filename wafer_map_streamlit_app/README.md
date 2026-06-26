# Streamlit Wafer Map Analyzer

반도체 wafer map CSV/Excel 데이터를 바로 선택해 두께와 시트저항 분포를 heatmap으로 확인하고, 기본 통계와 대표 불량 패턴을 판정하는 Streamlit 웹앱입니다.

## 포함 파일

- `app.py`: Streamlit 웹앱
- `data/`: 앱에서 자동 인식하는 예시 데이터 폴더
- `data/synthetic_wafer_map.csv`: 기본 synthetic wafer-map 예시 데이터
- `data/synthetic_deposition_edge_center.csv`: 증착 edge ring + center anomaly 예시
- `data/synthetic_etch_gradient_local.csv`: 식각 gradient + local defect 예시
- `scripts/generate_synthetic_data.py`: 기본 예시 데이터 재생성 스크립트
- `requirements.txt`: 로컬 실행 의존성

## 데이터 선택 방식

앱 사이드바의 `데이터 소스`에서 아래 방식을 선택할 수 있습니다.

- `data 폴더`: `data/` 폴더 안의 `.csv`, `.xlsx`, `.xls` 파일을 자동 인식하고 파일명을 클릭해 바로 로드합니다.
- `파일 업로드`: 단일 `.csv`, `.xlsx`, `.xls` 파일을 업로드합니다.
- `파일 업로드`: `.zip` 파일을 업로드하면 ZIP 내부의 `.csv`, `.xlsx`, `.xls` 파일 목록을 보여주며, 원하는 파일을 선택해 분석할 수 있습니다.

Excel 파일에 여러 시트가 있으면 사이드바에서 시트도 선택할 수 있습니다.

## CSV/Excel 형식

기본 예시 데이터는 아래 컬럼을 포함합니다.

| 컬럼 | 설명 |
| --- | --- |
| `lot_id` | Lot 식별자 |
| `wafer_id` | Wafer 식별자 |
| `site_id` | 측정 site 식별자 |
| `die_x`, `die_y` | Wafer map 좌표 |
| `radius_norm`, `theta_deg` | 정규화 반경과 각도 |
| `zone` | center/middle/edge 구역 |
| `thickness_nm` | 두께 측정값 |
| `sheet_resistance_ohm_sq` | 시트저항 측정값 |
| `synthetic_pattern_note` | synthetic 데이터에 심은 패턴 힌트 |

컬럼명이 다르더라도 앱 왼쪽 사이드바에서 X/Y 좌표, 두께, 시트저항 컬럼을 직접 선택할 수 있습니다.

## 분석 기능

- `data/` 폴더 내 데이터 파일 자동 인식
- CSV, XLSX, XLS, ZIP 업로드 지원
- 두께 heatmap, 시트저항 heatmap
- 평균, 표준편차, 균일도 `(max - min) / (2 * mean) * 100`, CV 계산
- Edge ring, center anomaly, gradient, local defect 패턴 판정
- 공정 선택별 원인 후보와 추가 분석 장비/로그 추천
- 패턴 판정 결과 및 분석 데이터 CSV 다운로드

## 로컬 실행 방법

```powershell
cd C:\Users\schoo\Documents\Codex\2026-06-23\python-streamlit-wafer-map-csv-heatmap\outputs\wafer_map_streamlit_app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

브라우저가 자동으로 열리지 않으면 터미널에 표시되는 `http://localhost:8501` 주소를 열면 됩니다.

## 데이터 파일 추가 방법

새 wafer map 파일을 `data/` 폴더에 넣고 앱을 새로고침하면 사이드바 목록에 자동으로 나타납니다.

지원 형식:

- `.csv`
- `.xlsx`
- `.xls`
- `.zip` 업로드 내부의 `.csv`, `.xlsx`, `.xls`

## synthetic CSV 재생성

```powershell
python scripts\generate_synthetic_data.py
```

예시 데이터에는 edge ring, center anomaly, gradient, local defect가 섞여 있습니다. 실제 공정 데이터에서는 패턴별 threshold를 sidebar에서 조정해 장비별 noise 수준에 맞추는 것을 권장합니다.
