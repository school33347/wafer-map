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
            "샤워헤드 유량 분포 또는 precursor depletion",
            "susceptor/ESC 온도 편차",
            "edge exclusion, clamp ring, wafer rotation 조건",
            "챔버 seasoning 상태 또는 wall loading 변화",
        ],
        "base_tools": [
            "타원계/reflectometer 두께 맵",
            "XRR 또는 XRF 조성/두께 확인",
            "AFM 표면 거칠기",
            "RGA, MFC log, chamber pressure/RF log",
        ],
    },
    "식각": {
        "base_causes": [
            "edge ring 소모 또는 focus ring 상태 변화",
            "플라즈마 밀도/RF bias 불균일",
            "gas flow, pressure, endpoint timing 편차",
            "photoresist 잔막 또는 polymer redeposition",
        ],
        "base_tools": [
            "CD-SEM / OCD profile",
            "ellipsometer 잔막 두께 맵",
            "OES endpoint trace",
            "XPS/ToF-SIMS 표면 잔류물 분석",
        ],
    },
    "포토": {
        "base_causes": [
            "exposure dose/focus drift",
            "spin coat 두께 및 EBR 조건",
            "hot plate bake 온도 균일도",
            "scanner leveling, overlay, reticle/illumination 조건",
        ],
        "base_tools": [
            "CD-SEM CD uniformity",
            "overlay metrology",
            "film thickness mapper",
            "defect inspection / macro inspection",
        ],
    },
}


PATTERN_GUIDANCE: dict[str, dict[str, dict[str, list[str]]]] = {
    "edge_ring": {
        "증착": {
            "causes": ["edge gas depletion", "edge temperature drop", "carrier ring contact/rotation issue"],
            "tools": ["edge bead/edge exclusion review", "wafer edge ellipsometry", "susceptor temperature log"],
        },
        "식각": {
            "causes": ["focus ring wear", "edge plasma sheath change", "clamp/ESC backside He leakage"],
            "tools": ["focus ring inspection", "OES radial trend", "ESC helium leak log"],
        },
        "포토": {
            "causes": ["edge bead removal drift", "spin speed acceleration profile", "edge dispense imbalance"],
            "tools": ["macro inspection", "resist thickness edge scan", "track EBR dispense log"],
        },
    },
    "center_anomaly": {
        "증착": {
            "causes": ["center showerhead distribution bias", "wafer bow/contact thermal shift", "center precursor stagnation"],
            "tools": ["center-to-edge thickness scan", "wafer bow measurement", "chamber flow simulation/log review"],
        },
        "식각": {
            "causes": ["center plasma density peak", "endpoint over/under etch", "ESC cooling zone imbalance"],
            "tools": ["post-etch residue map", "thermal zone log", "OCD/CD-SEM center sampling"],
        },
        "포토": {
            "causes": ["focus leveling error", "resist puddle/spin center mark", "hot plate center temperature bias"],
            "tools": ["focus-exposure matrix", "resist thickness map", "hot plate temperature calibration"],
        },
    },
    "gradient": {
        "증착": {
            "causes": ["cross-wafer flow directionality", "temperature zone skew", "wafer rotation slip"],
            "tools": ["MFC/pressure trend", "multi-zone heater log", "rotation hardware check"],
        },
        "식각": {
            "causes": ["gas inlet asymmetry", "magnetic field/plasma tilt", "RF matching drift"],
            "tools": ["plasma/OES spatial check", "RF match log", "chamber hardware symmetry check"],
        },
        "포토": {
            "causes": ["track nozzle gradient", "scanner slit/focus trend", "bake plate zone offset"],
            "tools": ["CD uniformity by slit position", "overlay/focus map", "track dispense and bake zone log"],
        },
    },
    "local_defect": {
        "증착": {
            "causes": ["particle shadowing", "micro-scratch", "local nucleation delay"],
            "tools": ["bright-field defect inspection", "SEM review", "AFM or optical microscope review"],
        },
        "식각": {
            "causes": ["particle micromasking", "local polymer residue", "chuck spot contamination"],
            "tools": ["defect review SEM", "post-etch residue inspection", "wet clean split check"],
        },
        "포토": {
            "causes": ["particle/air bubble", "reticle contamination", "local resist coating defect"],
            "tools": ["KLA/defect inspection", "reticle inspection", "macro/microscope review"],
        },
    },
}


METRIC_LABELS = {
    "thickness": "두께",
    "sheet": "시트저항",
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
        return {"mean": np.nan, "std": np.nan, "uniformity": np.nan, "cv": np.nan, "min": np.nan, "max": np.nan}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    denominator = abs(mean) if abs(mean) > 1e-12 else np.nan
    uniformity = float((clean.max() - clean.min()) / (2 * denominator) * 100) if denominator == denominator else np.nan
    cv = float(std / denominator * 100) if denominator == denominator else np.nan
    return {
        "mean": mean,
        "std": std,
        "uniformity": uniformity,
        "cv": cv,
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
                "pattern": "데이터 수",
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
        direction = "edge high" if delta > 0 else "edge low"
        results.append(
            {
                "metric": metric_name,
                "pattern_key": "edge_ring",
                "pattern": "Edge ring",
                "status": status_from_score(score, radial_threshold),
                "score": score,
                "evidence": f"{direction}, edge-inner {delta:+.3g}, score {score:.2f}",
            }
        )
    else:
        results.append(
            {
                "metric": metric_name,
                "pattern_key": "edge_ring",
                "pattern": "Edge ring",
                "status": "데이터 부족",
                "score": np.nan,
                "evidence": "edge 또는 inner 영역 측정점 부족",
            }
        )

    center = data[data["_radius_norm"] <= 0.25]
    mid = data[(data["_radius_norm"] >= 0.35) & (data["_radius_norm"] <= 0.75)]
    if len(center) >= 3 and len(mid) >= 3:
        delta = float(center[metric_col].mean() - mid[metric_col].mean())
        score = abs(delta) / global_std
        direction = "center high" if delta > 0 else "center low"
        results.append(
            {
                "metric": metric_name,
                "pattern_key": "center_anomaly",
                "pattern": "Center anomaly",
                "status": status_from_score(score, center_threshold),
                "score": score,
                "evidence": f"{direction}, center-mid {delta:+.3g}, score {score:.2f}",
            }
        )
    else:
        results.append(
            {
                "metric": metric_name,
                "pattern_key": "center_anomaly",
                "pattern": "Center anomaly",
                "status": "데이터 부족",
                "score": np.nan,
                "evidence": "center 또는 mid 영역 측정점 부족",
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
            "pattern": "Gradient",
            "status": gradient_status,
            "score": gradient_score,
            "evidence": f"R2 {r2:.2f}, gradient {gradient_strength:.2f} sigma/radius, high side {direction_name(beta[1], beta[2])}",
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
            f"({row['_x_norm']:.2f}R,{row['_y_norm']:.2f}R z={row['_robust_z']:+.1f})"
            for _, row in top.iterrows()
        )
        evidence = f"{len(local)} site(s), max |robust z| {max_abs_z:.1f}: {coords}"
    else:
        evidence = f"max |robust z| {max_abs_z:.1f}"
    results.append(
        {
            "metric": metric_name,
            "pattern_key": "local_defect",
            "pattern": "Local defect",
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
                f"{title}: %{{customdata[0]:.4g}} {unit}<br>radius: %{{customdata[1]:.2f}}R<extra></extra>"
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
                hovertemplate=f"local defect candidate<br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
                name="local defect",
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
                "패턴": str(row["패턴"]),
                "원인 후보": "\n".join(guidance["causes"]),
                "추가 분석 장비/로그": "\n".join(guidance["tools"]),
            }
        )

    if not rows:
        base = PROCESS_KNOWLEDGE[process]
        rows.append(
            {
                "측정치": "공정 공통",
                "패턴": "명확한 이상 패턴 없음",
                "원인 후보": "\n".join(base["base_causes"]),
                "추가 분석 장비/로그": "\n".join(base["base_tools"]),
            }
        )
    return pd.DataFrame(rows)


def show_metric_cards(label: str, stats: dict[str, float], unit: str) -> None:
    cols = st.columns(4)
    cols[0].metric(f"{label} 평균", f"{stats['mean']:.3g} {unit}")
    cols[1].metric(f"{label} 표준편차", f"{stats['std']:.3g} {unit}")
    cols[2].metric(f"{label} 균일도", f"±{stats['uniformity']:.2f}%")
    cols[3].metric(f"{label} CV", f"{stats['cv']:.2f}%")


def main() -> None:
    st.set_page_config(page_title="Wafer Map Analyzer", page_icon=None, layout="wide")
    st.title("Wafer Map Analyzer")

    selected_local_file: Path | None = None
    uploaded_file = None
    uploaded_bytes: bytes | None = None
    selected_archive_member: str | None = None
    selected_sheet: str | None = None

    with st.sidebar:
        st.subheader("데이터")
        source_mode = st.radio("데이터 소스", ["data 폴더", "파일 업로드"], horizontal=True)

        if source_mode == "data 폴더":
            local_files = list_data_files()
            if local_files:
                selected_local_file = st.radio(
                    "파일 선택",
                    local_files,
                    format_func=lambda path: path.name,
                )
                st.caption(f"{DATA_DIR.name}/ 폴더에서 {len(local_files)}개 파일 인식")
                if selected_local_file and is_excel_file(selected_local_file):
                    sheet_names = get_local_excel_sheets(
                        str(selected_local_file),
                        selected_local_file.stat().st_mtime_ns,
                    )
                    selected_sheet = st.selectbox("Excel 시트", sheet_names)
            else:
                st.warning("data 폴더에 CSV/Excel 파일이 없습니다.")
        else:
            uploaded_file = st.file_uploader("CSV/Excel/ZIP 업로드", type=["csv", "xlsx", "xls", "zip"])
            if uploaded_file is not None:
                uploaded_bytes = uploaded_file.getvalue()
                uploaded_suffix = Path(uploaded_file.name).suffix.lower()
                if uploaded_suffix == ".zip":
                    try:
                        archive_members = get_archive_members(uploaded_file.name, uploaded_bytes)
                    except BadZipFile:
                        st.error("ZIP 파일을 읽을 수 없습니다.")
                        st.stop()
                    if not archive_members:
                        st.error("ZIP 안에서 CSV/Excel 파일을 찾지 못했습니다.")
                        st.stop()
                    selected_archive_member = st.selectbox(
                        "ZIP 내부 파일",
                        archive_members,
                        format_func=lambda member: Path(member).name,
                    )
                    if selected_archive_member and is_excel_file(selected_archive_member):
                        sheet_names = get_archive_excel_sheets(
                            uploaded_file.name,
                            uploaded_bytes,
                            selected_archive_member,
                        )
                        selected_sheet = st.selectbox("Excel 시트", sheet_names)
                elif is_excel_file(uploaded_file.name):
                    sheet_names = get_uploaded_excel_sheets(uploaded_file.name, uploaded_bytes)
                    selected_sheet = st.selectbox("Excel 시트", sheet_names)
                elif uploaded_suffix not in SUPPORTED_DATA_EXTENSIONS:
                    st.error("CSV, XLSX, XLS, ZIP 파일만 지원합니다.")
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
                help="Edge ring 판정 기준입니다. 숫자가 낮을수록 더 민감하게 감지합니다.",
            )
            center_threshold = st.slider(
                "중앙 이상 기준",
                0.5,
                3.0,
                1.15,
                0.05,
                help="Center anomaly 판정 기준입니다. 숫자가 낮을수록 더 민감하게 감지합니다.",
            )
            gradient_r2_threshold = st.slider(
                "방향성 변화 기준",
                0.05,
                0.80,
                0.22,
                0.01,
                help="Gradient 판정 기준입니다. 숫자가 낮을수록 방향성 변화를 더 쉽게 감지합니다.",
            )
            local_threshold = st.slider(
                "국소 결함 기준",
                2.0,
                8.0,
                3.5,
                0.1,
                help="Local defect 판정 기준입니다. 숫자가 낮을수록 튀는 측정점을 더 쉽게 표시합니다.",
            )

    try:
        if source_mode == "data 폴더":
            if selected_local_file is None:
                if not SAMPLE_CSV.exists():
                    st.error("data 폴더에 읽을 수 있는 CSV/Excel 파일이 없습니다.")
                    st.stop()
                raw_df = load_sample_data()
                source_name = "내장 synthetic CSV"
            else:
                raw_df = load_local_data(
                    str(selected_local_file),
                    selected_local_file.stat().st_mtime_ns,
                    selected_sheet,
                )
                source_name = selected_local_file.name
        elif uploaded_file is None or uploaded_bytes is None:
            raw_df = load_sample_data()
            source_name = "내장 synthetic CSV"
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
            source_name = f"{source_name} / sheet: {selected_sheet}"
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
            x_col = st.selectbox("X 좌표", numeric_candidates, index=numeric_candidates.index(guessed_x) if guessed_x in numeric_candidates else 0)
            y_col = st.selectbox("Y 좌표", numeric_candidates, index=numeric_candidates.index(guessed_y) if guessed_y in numeric_candidates else min(1, len(numeric_candidates) - 1))
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
        selected_wafer = st.sidebar.selectbox("Wafer ID", wafer_options)
        selected_df = raw_df[raw_df[wafer_id_col].astype(str) == selected_wafer].copy()

    required_cols = [x_col, y_col, thickness_col, sheet_col]
    data = to_numeric_frame(selected_df, required_cols)
    if data.empty:
        st.error("선택한 컬럼 조합에서 유효한 숫자 데이터가 없습니다.")
        st.stop()
    data = add_geometry(data, x_col, y_col)

    st.caption(f"Data source: {source_name} | points: {len(data):,} | process: {process}")

    thickness_stats = metric_stats(data[thickness_col])
    sheet_stats = metric_stats(data[sheet_col])
    show_metric_cards("두께", thickness_stats, "nm")
    show_metric_cards("시트저항", sheet_stats, "ohm/sq")

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
            "pattern": "패턴",
            "status": "판정",
            "score": "점수",
            "evidence": "근거",
        }
    )
    pattern_table["점수"] = pattern_table["점수"].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")

    st.subheader("Heatmap 비교")
    heatmap_cols = st.columns(2)
    with heatmap_cols[0]:
        fig = create_wafer_map(data, x_col, y_col, thickness_col, "두께 heatmap", "Viridis", "nm", local_threshold)
        st.plotly_chart(fig, use_container_width=True)
    with heatmap_cols[1]:
        fig = create_wafer_map(data, x_col, y_col, sheet_col, "시트저항 heatmap", "RdBu_r", "ohm/sq", local_threshold)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("패턴 판정")
    status_counts = pattern_table["판정"].value_counts()
    summary_cols = st.columns(3)
    summary_cols[0].metric("감지", int(status_counts.get("감지", 0)))
    summary_cols[1].metric("주의", int(status_counts.get("주의", 0)))
    summary_cols[2].metric("정상", int(status_counts.get("정상", 0)))
    st.dataframe(
        style_status_table(pattern_table[["측정치", "패턴", "판정", "근거"]]),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("상세 점수 보기", expanded=False):
        st.dataframe(
            style_status_table(pattern_table[["측정치", "패턴", "판정", "점수", "근거"]]),
            use_container_width=True,
            hide_index=True,
        )
    st.download_button(
        "패턴 판정 CSV 다운로드",
        pattern_table.to_csv(index=False).encode("utf-8-sig"),
        file_name="wafer_pattern_judgement.csv",
        mime="text/csv",
    )

    st.subheader("원인 후보")
    recommendations = build_recommendations(process, pattern_table)
    st.dataframe(recommendations, use_container_width=True, hide_index=True)
    st.download_button(
        "분석 결과 CSV 다운로드",
        data.to_csv(index=False).encode("utf-8-sig"),
        file_name="wafer_map_analyzed.csv",
        mime="text/csv",
    )

    with st.expander("데이터 미리보기", expanded=False):
        st.dataframe(data.head(100), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
