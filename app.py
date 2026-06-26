from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
SAMPLE_CSV = DATA_DIR / "synthetic_wafer_map.csv"
SUPPORTED_DATA_EXTENSIONS = {".csv", ".xlsx", ".xls"}
EXCEL_EXTENSIONS = {".xlsx", ".xls"}


PROCESS_KNOWLEDGE: dict[str, dict[str, list[str]]] = {
    "증착": {
        "base_causes": [
            "가스 공급 분포 불균일",
            "웨이퍼 받침대 온도 편차",
            "가장자리 제외 영역 또는 고정 링 조건 변화",
            "챔버 상태 변화 또는 벽면 누적 영향",
        ],
        "base_tools": [
            "박막 두께 측정기",
            "조성 및 두께 확인 장비",
            "표면 거칠기 측정",
            "가스 유량, 압력, 전력 기록 확인",
        ],
    },
    "식각": {
        "base_causes": [
            "가장자리 링 또는 초점 링 마모",
            "플라즈마 분포 또는 전압 조건 불균일",
            "가스 유량, 압력, 종료 시점 편차",
            "감광막 잔류물 또는 식각 부산물 재부착",
        ],
        "base_tools": [
            "선폭 및 단면 형상 측정",
            "잔막 두께 맵 확인",
            "식각 종료 신호 기록 확인",
            "표면 잔류물 분석",
        ],
    },
    "포토": {
        "base_causes": [
            "노광량 또는 초점 조건 변화",
            "감광막 코팅 두께 및 가장자리 제거 조건",
            "베이크 온도 균일도",
            "노광 장비 수평, 정렬, 마스크 조건 변화",
        ],
        "base_tools": [
            "선폭 균일도 측정",
            "오버레이 정렬 측정",
            "감광막 두께 맵 확인",
            "결함 검사 및 육안 수준 검사",
        ],
    },
}


PATTERN_GUIDANCE: dict[str, dict[str, dict[str, list[str]]]] = {
    "edge_ring": {
        "증착": {
            "causes": ["가장자리 가스 공급 부족", "가장자리 온도 저하", "웨이퍼 지지 링 접촉 또는 회전 문제"],
            "tools": ["가장자리 제외 영역 조건 확인", "가장자리 두께 측정", "웨이퍼 받침대 온도 기록 확인"],
        },
        "식각": {
            "causes": ["초점 링 마모", "가장자리 플라즈마 분포 변화", "웨이퍼 고정 또는 뒷면 냉각 조건 이상"],
            "tools": ["초점 링 상태 점검", "반경 방향 플라즈마 신호 확인", "웨이퍼 고정 장치 냉각 기록 확인"],
        },
        "포토": {
            "causes": ["가장자리 감광막 제거 조건 변화", "회전 코팅 속도 변화", "가장자리 도포량 불균일"],
            "tools": ["외관 검사", "가장자리 감광막 두께 측정", "도포 장비 가장자리 제거 기록 확인"],
        },
    },
    "center_anomaly": {
        "증착": {
            "causes": ["중앙부 가스 분포 치우침", "웨이퍼 휨 또는 접촉 열전달 변화", "중앙부 반응 가스 정체"],
            "tools": ["중앙-가장자리 두께 비교", "웨이퍼 휨 측정", "챔버 유량 기록 확인"],
        },
        "식각": {
            "causes": ["중앙부 플라즈마 집중", "식각 시간 과다 또는 부족", "냉각 영역별 온도 불균일"],
            "tools": ["식각 후 잔류물 맵 확인", "온도 영역별 기록 확인", "중앙부 선폭 및 형상 측정"],
        },
        "포토": {
            "causes": ["초점 또는 수평 보정 오차", "중앙부 감광막 도포 흔적", "베이크 판 중앙 온도 편차"],
            "tools": ["초점-노광 조건 확인", "감광막 두께 맵 확인", "베이크 판 온도 보정 확인"],
        },
    },
    "gradient": {
        "증착": {
            "causes": ["웨이퍼 한쪽 방향 가스 흐름 치우침", "온도 영역 치우침", "웨이퍼 회전 불안정"],
            "tools": ["가스 유량 및 압력 추세 확인", "다중 온도 영역 기록 확인", "회전 장치 점검"],
        },
        "식각": {
            "causes": ["가스 유입 방향 불균형", "플라즈마가 한쪽으로 기울어짐", "전력 매칭 조건 변화"],
            "tools": ["공간별 플라즈마 신호 확인", "전력 매칭 기록 확인", "챔버 부품 대칭 상태 점검"],
        },
        "포토": {
            "causes": ["도포 노즐 방향성 편차", "노광 장비 초점 경향 변화", "베이크 판 영역별 온도 차이"],
            "tools": ["위치별 선폭 균일도 확인", "정렬 및 초점 맵 확인", "도포와 베이크 영역별 기록 확인"],
        },
    },
    "local_defect": {
        "증착": {
            "causes": ["입자에 의한 국소 가림", "미세 스크래치", "국소 박막 성장 지연"],
            "tools": ["광학 결함 검사", "전자현미경 리뷰", "표면 형상 또는 현미경 확인"],
        },
        "식각": {
            "causes": ["입자에 의한 국소 식각 방해", "국소 부산물 잔류", "척 표면 오염"],
            "tools": ["결함 위치 전자현미경 확인", "식각 후 잔류물 검사", "세정 조건 비교 확인"],
        },
        "포토": {
            "causes": ["입자 또는 기포", "마스크 오염", "국소 감광막 코팅 불량"],
            "tools": ["결함 검사 장비 확인", "마스크 검사", "외관 및 현미경 확인"],
        },
    },
}


METRIC_LABELS = {
    "thickness": "두께",
    "sheet": "시트저항",
}

PATTERN_LABELS = {
    "edge_ring": "가장자리 링 이상",
    "center_anomaly": "중앙부 이상",
    "gradient": "한쪽 방향 변화",
    "local_defect": "국소 결함",
    "insufficient": "데이터 부족",
}


def guess_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {column.lower().replace(" ", "").replace("-", "_"): column for column in columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("-", "_")
        if key in normalized:
            return normalized[key]
    for column in columns:
        key = column.lower().replace(" ", "").replace("-", "_")
        if any(candidate in key for candidate in candidates):
            return column
    return None


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


def is_excel_file(file_name: str | Path) -> bool:
    return Path(file_name).suffix.lower() in EXCEL_EXTENSIONS


def list_data_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(
        [path for path in DATA_DIR.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_DATA_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def display_data_file_name(path: Path) -> str:
    labels = {
        "synthetic_wafer_map.csv": "기본 예시 데이터",
        "synthetic_deposition_edge_center.csv": "증착 예시: 가장자리/중앙부 이상",
        "synthetic_etch_gradient_local.csv": "식각 예시: 방향성 변화/국소 결함",
    }
    return labels.get(path.name, path.name)


def read_tabular_data(source: Any, file_name: str, sheet_name: str | None = None) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in EXCEL_EXTENSIONS:
        return pd.read_excel(source, sheet_name=sheet_name or 0)
    raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")


@st.cache_data(show_spinner=False)
def load_local_data(path_text: str, modified_ns: int, sheet_name: str | None = None) -> pd.DataFrame:
    path = Path(path_text)
    return read_tabular_data(path, path.name, sheet_name)


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_name: str, content: bytes, sheet_name: str | None = None) -> pd.DataFrame:
    return read_tabular_data(BytesIO(content), file_name, sheet_name)


@st.cache_data(show_spinner=False)
def load_archive_member(archive_name: str, content: bytes, member_name: str, sheet_name: str | None = None) -> pd.DataFrame:
    with ZipFile(BytesIO(content)) as archive:
        return read_tabular_data(BytesIO(archive.read(member_name)), member_name, sheet_name)


@st.cache_data(show_spinner=False)
def get_local_excel_sheets(path_text: str, modified_ns: int) -> list[str]:
    return list(pd.ExcelFile(Path(path_text)).sheet_names)


@st.cache_data(show_spinner=False)
def get_uploaded_excel_sheets(file_name: str, content: bytes) -> list[str]:
    return list(pd.ExcelFile(BytesIO(content)).sheet_names)


@st.cache_data(show_spinner=False)
def get_archive_members(archive_name: str, content: bytes) -> list[str]:
    with ZipFile(BytesIO(content)) as archive:
        return sorted(
            [
                info.filename
                for info in archive.infolist()
                if not info.is_dir() and Path(info.filename).suffix.lower() in SUPPORTED_DATA_EXTENSIONS
            ],
            key=str.lower,
        )


@st.cache_data(show_spinner=False)
def get_archive_excel_sheets(archive_name: str, content: bytes, member_name: str) -> list[str]:
    with ZipFile(BytesIO(content)) as archive:
        return get_uploaded_excel_sheets(member_name, archive.read(member_name))


def to_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    converted = df.copy()
    for column in columns:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")
    return converted.dropna(subset=columns)


def add_geometry(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    out = df.copy()
    center_x = (out[x_col].min() + out[x_col].max()) / 2
    center_y = (out[y_col].min() + out[y_col].max()) / 2
    x_centered = out[x_col] - center_x
    y_centered = out[y_col] - center_y
    radius = np.sqrt(np.square(x_centered) + np.square(y_centered))
    max_radius = float(radius.max()) if float(radius.max()) > 0 else 1.0
    out["_x_norm"] = x_centered / max_radius
    out["_y_norm"] = y_centered / max_radius
    out["_radius_norm"] = radius / max_radius
    out["_theta_deg"] = np.degrees(np.arctan2(out["_y_norm"], out["_x_norm"]))
    return out


def metric_stats(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"mean": np.nan, "std": np.nan, "uniformity": np.nan, "min": np.nan, "max": np.nan}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    denominator = abs(mean) if abs(mean) > 1e-12 else np.nan
    uniformity = float((clean.max() - clean.min()) / (2 * denominator) * 100) if denominator == denominator else np.nan
    return {
        "mean": mean,
        "std": std,
        "uniformity": uniformity,
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def status_from_score(score: float, threshold: float) -> str:
    if not np.isfinite(score):
        return "데이터 부족"
    if score >= threshold:
        return "감지"
    if score >= threshold * 0.7:
        return "주의"
    return "정상"


def direction_name(gx: float, gy: float) -> str:
    if abs(gx) < 1e-12 and abs(gy) < 1e-12:
        return "방향성 낮음"
    angle = (math.degrees(math.atan2(gy, gx)) + 360) % 360
    sectors = [
        ("동쪽", 22.5),
        ("북동", 67.5),
        ("북쪽", 112.5),
        ("북서", 157.5),
        ("서쪽", 202.5),
        ("남서", 247.5),
        ("남쪽", 292.5),
        ("남동", 337.5),
        ("동쪽", 360.0),
    ]
    for name, boundary in sectors:
        if angle < boundary:
            return name
    return "동쪽"


def format_change(value: float) -> str:
    return f"{abs(value):.3g}"


def safe_std(values: pd.Series) -> float:
    std = float(values.std(ddof=0))
    return std if std > 1e-12 else 1.0


def detect_patterns(
    df: pd.DataFrame,
    metric_col: str,
    metric_name: str,
    radial_threshold: float,
    center_threshold: float,
    local_threshold: float,
    gradient_r2_threshold: float,
) -> list[dict[str, Any]]:
    data = df.dropna(subset=[metric_col, "_x_norm", "_y_norm", "_radius_norm"]).copy()
    data[metric_col] = pd.to_numeric(data[metric_col], errors="coerce")
    data = data.dropna(subset=[metric_col])
    if len(data) < 8:
        return [
            {
                "metric": metric_name,
                "pattern_key": "insufficient",
                "pattern": PATTERN_LABELS["insufficient"],
                "status": "데이터 부족",
                "score": np.nan,
                "evidence": "유효 측정점이 8개 미만입니다.",
            }
        ]

    global_std = safe_std(data[metric_col])
    results: list[dict[str, Any]] = []

    edge = data[data["_radius_norm"] >= 0.82]
    inner = data[data["_radius_norm"] <= 0.65]
    if len(edge) >= 3 and len(inner) >= 3:
        delta = float(edge[metric_col].mean() - inner[metric_col].mean())
        score = abs(delta) / global_std
        direction = "높게" if delta > 0 else "낮게"
        results.append(
            {
                "metric": metric_name,
                "pattern_key": "edge_ring",
                "pattern": PATTERN_LABELS["edge_ring"],
                "status": status_from_score(score, radial_threshold),
                "score": score,
                "evidence": f"가장자리 평균이 내부보다 {format_change(delta)} {direction} 나타났습니다. 전체 산포 대비 차이는 {score:.2f}배입니다.",
            }
        )
    else:
        results.append(
            {
                "metric": metric_name,
                "pattern_key": "edge_ring",
                "pattern": PATTERN_LABELS["edge_ring"],
                "status": "데이터 부족",
                "score": np.nan,
                "evidence": "가장자리 또는 내부 영역의 측정점이 부족합니다.",
            }
        )

    center = data[data["_radius_norm"] <= 0.25]
    mid = data[(data["_radius_norm"] >= 0.35) & (data["_radius_norm"] <= 0.75)]
    if len(center) >= 3 and len(mid) >= 3:
        delta = float(center[metric_col].mean() - mid[metric_col].mean())
        score = abs(delta) / global_std
        direction = "높게" if delta > 0 else "낮게"
        results.append(
            {
                "metric": metric_name,
                "pattern_key": "center_anomaly",
                "pattern": PATTERN_LABELS["center_anomaly"],
                "status": status_from_score(score, center_threshold),
                "score": score,
                "evidence": f"중앙부 평균이 중간 영역보다 {format_change(delta)} {direction} 나타났습니다. 전체 산포 대비 차이는 {score:.2f}배입니다.",
            }
        )
    else:
        results.append(
            {
                "metric": metric_name,
                "pattern_key": "center_anomaly",
                "pattern": PATTERN_LABELS["center_anomaly"],
                "status": "데이터 부족",
                "score": np.nan,
                "evidence": "중앙부 또는 중간 영역의 측정점이 부족합니다.",
            }
        )

    z = data[metric_col].to_numpy(dtype=float)
    x = data["_x_norm"].to_numpy(dtype=float)
    y = data["_y_norm"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(data)), x, y])
    beta, *_ = np.linalg.lstsq(design, z, rcond=None)
    prediction = design @ beta
    ss_res = float(np.sum(np.square(z - prediction)))
    ss_tot = float(np.sum(np.square(z - np.mean(z))))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    gradient_strength = float(math.sqrt(beta[1] ** 2 + beta[2] ** 2) / global_std)
    gradient_score = r2
    gradient_detected = r2 >= gradient_r2_threshold and gradient_strength >= 0.45
    gradient_status = "감지" if gradient_detected else ("주의" if r2 >= gradient_r2_threshold * 0.7 else "정상")
    results.append(
        {
            "metric": metric_name,
            "pattern_key": "gradient",
            "pattern": PATTERN_LABELS["gradient"],
            "status": gradient_status,
            "score": gradient_score,
            "evidence": f"웨이퍼에서 {direction_name(beta[1], beta[2])} 방향으로 값이 높아지는 경향이 있습니다. 방향성 설명력은 {r2:.2f}, 변화 강도는 {gradient_strength:.2f}입니다.",
        }
    )

    median = float(np.median(z))
    mad = float(np.median(np.abs(z - median)))
    if mad > 1e-12:
        robust_z = 0.6745 * (z - median) / mad
    else:
        robust_z = (z - np.mean(z)) / global_std
    data["_robust_z"] = robust_z
    local = data[np.abs(data["_robust_z"]) >= local_threshold].copy()
    max_abs_z = float(np.max(np.abs(robust_z))) if len(robust_z) else np.nan
    if not local.empty:
        top = local.reindex(local["_robust_z"].abs().sort_values(ascending=False).index).head(3)
        coords = ", ".join(
            f"({row['_x_norm']:.2f}R, {row['_y_norm']:.2f}R)"
            for _, row in top.iterrows()
        )
        evidence = f"주변 분포에서 벗어난 측정점이 {len(local)}개 보입니다. 대표 위치: {coords}"
    else:
        evidence = f"뚜렷하게 튀는 측정점은 보이지 않습니다. 최대 이상 정도는 {max_abs_z:.1f}입니다."
    results.append(
        {
            "metric": metric_name,
            "pattern_key": "local_defect",
            "pattern": PATTERN_LABELS["local_defect"],
            "status": "감지" if not local.empty else ("주의" if max_abs_z >= local_threshold * 0.7 else "정상"),
            "score": max_abs_z,
            "evidence": evidence,
        }
    )
    return results


def style_status_table(df: pd.DataFrame) -> Any:
    colors = {
        "감지": "background-color: #ffe1df; color: #7a1e16;",
        "주의": "background-color: #fff0c2; color: #654600;",
        "정상": "background-color: #ddf4e8; color: #14532d;",
        "데이터 부족": "background-color: #eceff3; color: #334155;",
    }
    styler = df.style
    if hasattr(styler, "map"):
        return styler.map(lambda value: colors.get(value, ""), subset=["판정"])
    return styler.applymap(lambda value: colors.get(value, ""), subset=["판정"])


def create_wafer_map(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    metric_col: str,
    title: str,
    colorscale: str,
    unit: str,
    local_threshold: float,
) -> go.Figure:
    plot_df = df.dropna(subset=[x_col, y_col, metric_col]).copy()
    if plot_df.empty:
        return go.Figure()

    x_span = max(float(plot_df[x_col].max() - plot_df[x_col].min()), 1.0)
    y_span = max(float(plot_df[y_col].max() - plot_df[y_col].min()), 1.0)
    size = max(8, min(22, int(420 / max(math.sqrt(len(plot_df)), 1))))
    median = float(plot_df[metric_col].median())
    mad = float(np.median(np.abs(plot_df[metric_col] - median)))
    if mad > 1e-12:
        robust_z = 0.6745 * (plot_df[metric_col] - median) / mad
    else:
        robust_z = (plot_df[metric_col] - plot_df[metric_col].mean()) / safe_std(plot_df[metric_col])
    plot_df["_local_flag"] = np.abs(robust_z) >= local_threshold

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df[x_col],
            y=plot_df[y_col],
            mode="markers",
            marker={
                "symbol": "square",
                "size": size,
                "color": plot_df[metric_col],
                "colorscale": colorscale,
                "showscale": True,
                "colorbar": {"title": unit},
                "line": {"width": 0.5, "color": "rgba(17,24,39,0.35)"},
            },
            customdata=np.stack(
                [
                    plot_df[metric_col],
                    plot_df["_radius_norm"] if "_radius_norm" in plot_df else np.zeros(len(plot_df)),
                ],
                axis=-1,
            ),
            hovertemplate=(
                f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>"
                f"{title}: %{{customdata[0]:.4g}} {unit}<br>정규화 반경: %{{customdata[1]:.2f}}R<extra></extra>"
            ),
            name=title,
        )
    )

    flagged = plot_df[plot_df["_local_flag"]]
    if not flagged.empty:
        fig.add_trace(
            go.Scatter(
                x=flagged[x_col],
                y=flagged[y_col],
                mode="markers",
                marker={
                    "symbol": "x",
                    "size": size + 5,
                    "color": "black",
                    "line": {"width": 2, "color": "black"},
                },
                hovertemplate=f"국소 결함 후보<br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
                name="국소 결함 후보",
            )
        )

    center_x = (float(plot_df[x_col].min()) + float(plot_df[x_col].max())) / 2
    center_y = (float(plot_df[y_col].min()) + float(plot_df[y_col].max())) / 2
    wafer_radius = max(x_span, y_span) / 2
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=center_x - wafer_radius,
        y0=center_y - wafer_radius,
        x1=center_x + wafer_radius,
        y1=center_y + wafer_radius,
        line={"color": "rgba(15,23,42,0.65)", "width": 2},
    )
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=center_x - wafer_radius * 0.82,
        y0=center_y - wafer_radius * 0.82,
        x1=center_x + wafer_radius * 0.82,
        y1=center_y + wafer_radius * 0.82,
        line={"color": "rgba(15,23,42,0.25)", "width": 1, "dash": "dot"},
    )
    fig.update_layout(
        title=title,
        height=560,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
        xaxis={"title": x_col, "scaleanchor": "y", "scaleratio": 1, "showgrid": False, "zeroline": False},
        yaxis={"title": y_col, "showgrid": False, "zeroline": False},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        plot_bgcolor="white",
    )
    return fig


def build_recommendations(process: str, detected: pd.DataFrame) -> pd.DataFrame:
    active = detected[detected["판정"].isin(["감지", "주의"])]
    rows: list[dict[str, str]] = []
    for _, row in active.iterrows():
        key = str(row["pattern_key"])
        guidance = PATTERN_GUIDANCE.get(key, {}).get(process)
        if not guidance:
            continue
        rows.append(
            {
                "측정치": str(row["측정치"]),
                "이상 유형": str(row["이상 유형"]),
                "의심 원인": "\n".join(guidance["causes"]),
                "추가 확인 방법": "\n".join(guidance["tools"]),
            }
        )

    if not rows:
        base = PROCESS_KNOWLEDGE[process]
        rows.append(
            {
                "측정치": "공정 공통",
                "이상 유형": "명확한 이상 유형 없음",
                "의심 원인": "\n".join(base["base_causes"]),
                "추가 확인 방법": "\n".join(base["base_tools"]),
            }
        )
    return pd.DataFrame(rows)


def show_metric_cards(label: str, stats: dict[str, float], unit: str) -> None:
    cols = st.columns(3)
    cols[0].metric(f"{label} 평균", f"{stats['mean']:.3g} {unit}")
    cols[1].metric(f"{label} 표준편차", f"{stats['std']:.3g} {unit}")
    cols[2].metric(
        f"{label} 균일도",
        f"±{stats['uniformity']:.2f}%",
        help="웨이퍼 전체에서 값이 얼마나 고르게 분포하는지 보는 지표입니다. 값이 낮을수록 전체가 더 균일합니다. 계산식: (최댓값 - 최솟값) / (2 × 평균) × 100",
    )


def main() -> None:
    st.set_page_config(page_title="웨이퍼 맵 분석기", page_icon=None, layout="wide")
    st.title("웨이퍼 맵 분석기")

    selected_local_file: Path | None = None
    uploaded_file = None
    uploaded_bytes: bytes | None = None
    selected_archive_member: str | None = None
    selected_sheet: str | None = None

    with st.sidebar:
        st.subheader("데이터")
        source_mode = st.radio("데이터 선택 방식", ["예시 데이터", "파일 업로드"], horizontal=True)

        if source_mode == "예시 데이터":
            local_files = list_data_files()
            if local_files:
                selected_local_file = st.radio(
                    "파일 선택",
                    local_files,
                    format_func=display_data_file_name,
                )
                st.caption(f"예시 데이터 {len(local_files)}개를 인식했습니다.")
                if selected_local_file and is_excel_file(selected_local_file):
                    sheet_names = get_local_excel_sheets(
                        str(selected_local_file),
                        selected_local_file.stat().st_mtime_ns,
                    )
                    selected_sheet = st.selectbox("엑셀 시트", sheet_names)
            else:
                st.warning("예시 데이터 폴더에 표 파일이 없습니다.")
        else:
            uploaded_file = st.file_uploader("표 파일 업로드", type=["csv", "xlsx", "xls", "zip"])
            if uploaded_file is not None:
                uploaded_bytes = uploaded_file.getvalue()
                uploaded_suffix = Path(uploaded_file.name).suffix.lower()
                if uploaded_suffix == ".zip":
                    try:
                        archive_members = get_archive_members(uploaded_file.name, uploaded_bytes)
                    except BadZipFile:
                        st.error("압축 파일을 읽을 수 없습니다.")
                        st.stop()
                    if not archive_members:
                        st.error("압축 파일 안에서 읽을 수 있는 표 파일을 찾지 못했습니다.")
                        st.stop()
                    selected_archive_member = st.selectbox(
                        "압축 파일 안의 데이터",
                        archive_members,
                        format_func=lambda member: Path(member).name,
                    )
                    if selected_archive_member and is_excel_file(selected_archive_member):
                        sheet_names = get_archive_excel_sheets(
                            uploaded_file.name,
                            uploaded_bytes,
                            selected_archive_member,
                        )
                        selected_sheet = st.selectbox("엑셀 시트", sheet_names)
                elif is_excel_file(uploaded_file.name):
                    sheet_names = get_uploaded_excel_sheets(uploaded_file.name, uploaded_bytes)
                    selected_sheet = st.selectbox("엑셀 시트", sheet_names)
                elif uploaded_suffix not in SUPPORTED_DATA_EXTENSIONS:
                    st.error("표 파일 또는 압축 파일만 지원합니다.")
                    st.stop()

        st.divider()
        process = st.selectbox("공정 선택", ["증착", "식각", "포토"])
        st.caption("기본 설정만으로 바로 분석할 수 있습니다.")

        with st.expander("고급 설정", expanded=False):
            st.caption("숫자를 낮추면 작은 변화도 더 잘 잡고, 높이면 확실한 이상만 잡습니다.")
            radial_threshold = st.slider(
                "가장자리 이상 기준",
                0.5,
                3.0,
                1.15,
                0.05,
                help="가장자리 링 이상 판정 기준입니다. 숫자가 낮을수록 더 민감하게 감지합니다.",
            )
            center_threshold = st.slider(
                "중앙 이상 기준",
                0.5,
                3.0,
                1.15,
                0.05,
                help="중앙부 이상 판정 기준입니다. 숫자가 낮을수록 더 민감하게 감지합니다.",
            )
            gradient_r2_threshold = st.slider(
                "방향성 변화 기준",
                0.05,
                0.80,
                0.22,
                0.01,
                help="한쪽 방향 변화 판정 기준입니다. 숫자가 낮을수록 방향성 변화를 더 쉽게 감지합니다.",
            )
            local_threshold = st.slider(
                "국소 결함 기준",
                2.0,
                8.0,
                3.5,
                0.1,
                help="국소 결함 판정 기준입니다. 숫자가 낮을수록 튀는 측정점을 더 쉽게 표시합니다.",
            )

    try:
        if source_mode == "예시 데이터":
            if selected_local_file is None:
                if not SAMPLE_CSV.exists():
                    st.error("예시 데이터 폴더에 읽을 수 있는 표 파일이 없습니다.")
                    st.stop()
                raw_df = load_sample_data()
                source_name = "내장 예시 데이터"
            else:
                raw_df = load_local_data(
                    str(selected_local_file),
                    selected_local_file.stat().st_mtime_ns,
                    selected_sheet,
                )
                source_name = display_data_file_name(selected_local_file)
        elif uploaded_file is None or uploaded_bytes is None:
            raw_df = load_sample_data()
            source_name = "내장 예시 데이터"
        elif selected_archive_member:
            raw_df = load_archive_member(
                uploaded_file.name,
                uploaded_bytes,
                selected_archive_member,
                selected_sheet,
            )
            source_name = f"{uploaded_file.name} > {selected_archive_member}"
        else:
            raw_df = load_uploaded_data(uploaded_file.name, uploaded_bytes, selected_sheet)
            source_name = uploaded_file.name

        if selected_sheet:
            source_name = f"{source_name} / 시트: {selected_sheet}"
    except FileNotFoundError:
        st.error("synthetic_wafer_map.csv 파일을 찾을 수 없습니다. scripts/generate_synthetic_data.py를 실행해 주세요.")
        st.stop()
    except Exception as exc:
        st.error(f"데이터 파일을 읽는 중 오류가 발생했습니다: {exc}")
        st.stop()

    if raw_df.empty:
        st.error("선택한 데이터 파일에 데이터가 없습니다.")
        st.stop()

    numeric_candidates = [
        column
        for column in raw_df.columns
        if pd.api.types.is_numeric_dtype(raw_df[column]) or pd.to_numeric(raw_df[column], errors="coerce").notna().mean() > 0.8
    ]
    if len(numeric_candidates) < 4:
        st.error("x, y, 두께, 시트저항에 대응되는 숫자형 컬럼이 필요합니다.")
        st.stop()

    guessed_x = guess_column(list(raw_df.columns), ["die_x", "x", "site_x", "map_x"])
    guessed_y = guess_column(list(raw_df.columns), ["die_y", "y", "site_y", "map_y"])
    guessed_thickness = guess_column(list(raw_df.columns), ["thickness_nm", "thickness", "film_thickness", "두께"])
    guessed_sheet = guess_column(
        list(raw_df.columns),
        ["sheet_resistance_ohm_sq", "sheet_resistance", "rs", "resistance", "시트저항"],
    )

    with st.sidebar:
        st.divider()
        with st.expander("데이터 컬럼 설정", expanded=False):
            st.caption("파일 컬럼명이 다를 때만 조정하세요.")
            x_col = st.selectbox("가로 좌표", numeric_candidates, index=numeric_candidates.index(guessed_x) if guessed_x in numeric_candidates else 0)
            y_col = st.selectbox("세로 좌표", numeric_candidates, index=numeric_candidates.index(guessed_y) if guessed_y in numeric_candidates else min(1, len(numeric_candidates) - 1))
            thickness_col = st.selectbox(
                "두께",
                numeric_candidates,
                index=numeric_candidates.index(guessed_thickness) if guessed_thickness in numeric_candidates else min(2, len(numeric_candidates) - 1),
            )
            sheet_col = st.selectbox(
                "시트저항",
                numeric_candidates,
                index=numeric_candidates.index(guessed_sheet) if guessed_sheet in numeric_candidates else min(3, len(numeric_candidates) - 1),
            )

    wafer_id_col = guess_column(list(raw_df.columns), ["wafer_id", "wafer", "웨이퍼"])
    selected_df = raw_df.copy()
    if wafer_id_col and raw_df[wafer_id_col].nunique() > 1:
        wafer_options = list(raw_df[wafer_id_col].dropna().astype(str).unique())
        selected_wafer = st.sidebar.selectbox("웨이퍼 번호", wafer_options)
        selected_df = raw_df[raw_df[wafer_id_col].astype(str) == selected_wafer].copy()

    required_cols = [x_col, y_col, thickness_col, sheet_col]
    data = to_numeric_frame(selected_df, required_cols)
    if data.empty:
        st.error("선택한 컬럼 조합에서 유효한 숫자 데이터가 없습니다.")
        st.stop()
    data = add_geometry(data, x_col, y_col)

    st.caption(f"데이터: {source_name} | 측정점: {len(data):,}개 | 공정: {process}")

    thickness_stats = metric_stats(data[thickness_col])
    sheet_stats = metric_stats(data[sheet_col])
    show_metric_cards("두께", thickness_stats, "nm")
    show_metric_cards("시트저항", sheet_stats, "Ω/□")

    detected_rows = []
    detected_rows.extend(
        detect_patterns(data, thickness_col, "두께", radial_threshold, center_threshold, local_threshold, gradient_r2_threshold)
    )
    detected_rows.extend(
        detect_patterns(data, sheet_col, "시트저항", radial_threshold, center_threshold, local_threshold, gradient_r2_threshold)
    )
    detected = pd.DataFrame(detected_rows)
    pattern_table = detected.rename(
        columns={
            "metric": "측정치",
            "pattern": "이상 유형",
            "status": "판정",
            "score": "점수",
            "evidence": "판정 이유",
        }
    )
    pattern_table["점수"] = pattern_table["점수"].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")

    st.subheader("히트맵 비교")
    heatmap_cols = st.columns(2)
    with heatmap_cols[0]:
        fig = create_wafer_map(data, x_col, y_col, thickness_col, "두께 히트맵", "Viridis", "nm", local_threshold)
        st.plotly_chart(fig, use_container_width=True)
    with heatmap_cols[1]:
        fig = create_wafer_map(data, x_col, y_col, sheet_col, "시트저항 히트맵", "RdBu_r", "Ω/□", local_threshold)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("이상 판정")
    status_counts = pattern_table["판정"].value_counts()
    summary_cols = st.columns(3)
    summary_cols[0].metric("감지", int(status_counts.get("감지", 0)))
    summary_cols[1].metric("주의", int(status_counts.get("주의", 0)))
    summary_cols[2].metric("정상", int(status_counts.get("정상", 0)))
    st.dataframe(
        style_status_table(pattern_table[["측정치", "이상 유형", "판정", "판정 이유"]]),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("상세 점수 보기", expanded=False):
        st.dataframe(
            style_status_table(pattern_table[["측정치", "이상 유형", "판정", "점수", "판정 이유"]]),
            use_container_width=True,
            hide_index=True,
        )
    st.download_button(
        "이상 판정 결과 내려받기",
        pattern_table.to_csv(index=False).encode("utf-8-sig"),
        file_name="wafer_pattern_judgement.csv",
        mime="text/csv",
    )

    st.subheader("의심 원인과 추가 확인 방법")
    recommendations = build_recommendations(process, pattern_table)
    st.dataframe(recommendations, use_container_width=True, hide_index=True)
    st.download_button(
        "분석 데이터 내려받기",
        data.to_csv(index=False).encode("utf-8-sig"),
        file_name="wafer_map_analyzed.csv",
        mime="text/csv",
    )

    with st.expander("데이터 미리보기", expanded=False):
        st.dataframe(data.head(100), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
