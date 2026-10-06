"""
生成した3Dモデル(STEP)を第三角法の三面図(＋左側面図)として線画PNGに描画するモジュール。
OpenCascadeの隠線処理(HLR)で、外形線=実線(黒) / かくれ線=破線(灰) を描き分ける。

座標系の約束（プロンプトでGeminiにも同じ約束を指示している）:
  正面図 : -Y側から見る (右=+X, 上=+Z)
  平面図 : +Z側から見る (右=+X, 上=+Y)   ※正面図の上に配置
  右側面図: +X側から見る (右=+Y, 上=+Z)   ※正面図の右に配置
  左側面図: -X側から見る (右=-Y, 上=+Z)   ※正面図の左に配置
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cadquery as cq
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir

plt.rcParams["font.family"] = ["MS Gothic", "Yu Gothic", "Meiryo", "sans-serif"]

# (名前, 視線の向き=観察者側の方向, 図の右方向)
VIEWS = {
    "front": ("正面図 (-Y側から見る / 右=+X, 上=+Z)", (0, -1, 0), (1, 0, 0)),
    "top":   ("平面図 (+Z側から見る / 右=+X, 上=+Y)", (0, 0, 1), (1, 0, 0)),
    "right": ("右側面図 (+X側から見る / 右=+Y, 上=+Z)", (1, 0, 0), (0, 1, 0)),
    "left":  ("左側面図 (-X側から見る / 右=-Y, 上=+Z)", (-1, 0, 0), (0, -1, 0)),
}


def _edges_to_polylines(compound):
    if compound is None or compound.IsNull():
        return []
    lines = []
    for e in cq.Shape.cast(compound).Edges():
        n = 2 if e.geomType() == "LINE" else 48
        pts = [e.positionAt(t) for t in np.linspace(0, 1, n)]
        lines.append(np.array([(p.x, p.y) for p in pts]))
    return lines


def project(shape, view_dir, x_dir):
    """HLRで投影し、(可視線リスト, かくれ線リスト) を2D座標で返す"""
    algo = HLRBRep_Algo()
    algo.Add(shape.wrapped)
    algo.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*view_dir), gp_Dir(*x_dir))))
    algo.Update()
    algo.Hide()
    hlr = HLRBRep_HLRToShape(algo)
    visible, hidden = [], []
    for c in (hlr.VCompound(), hlr.Rg1LineVCompound(), hlr.OutLineVCompound()):
        visible += _edges_to_polylines(c)
    for c in (hlr.HCompound(), hlr.OutLineHCompound()):
        hidden += _edges_to_polylines(c)
    return visible, hidden


def _bbox(lines):
    pts = np.vstack(lines) if lines else np.zeros((1, 2))
    return pts.min(axis=0), pts.max(axis=0)


def render_views(step_path, png_path, title=""):
    """STEPファイルを読み込み、第三角法配置の線画PNGを保存する"""
    shape = cq.importers.importStep(step_path).val()
    if isinstance(shape, cq.Workplane):
        shape = shape.val()

    proj = {k: project(shape, d, x) for k, (_, d, x) in VIEWS.items()}
    bbs = {k: _bbox(v + h) for k, (v, h) in proj.items()}
    size = {k: bbs[k][1] - bbs[k][0] for k in bbs}
    gap = 0.25 * max(max(s) for s in size.values())

    # 正面図を基準に配置（平面図は上、右側面図は右、左側面図は左。高さ/幅方向をそろえる）
    f_min, f_max = bbs["front"]
    offsets = {"front": np.array([0.0, 0.0])}
    offsets["top"] = np.array([0.0, (f_max[1] + gap) - bbs["top"][0][1]])
    offsets["top"][0] = 0.0  # X方向は正面図と共通
    offsets["right"] = np.array([(f_max[0] + gap) - bbs["right"][0][0], 0.0])
    offsets["left"] = np.array([(f_min[0] - gap) - bbs["left"][1][0], 0.0])

    fig, ax = plt.subplots(figsize=(16, 11))
    for k, (vis, hid) in proj.items():
        off = offsets[k]
        for l in hid:
            ax.plot(l[:, 0] + off[0], l[:, 1] + off[1], color="0.55", lw=0.8, ls=(0, (4, 3)))
        for l in vis:
            ax.plot(l[:, 0] + off[0], l[:, 1] + off[1], color="black", lw=1.4)
        lo, hi = bbs[k][0] + off, bbs[k][1] + off
        label, _, _ = VIEWS[k]
        ax.text((lo[0] + hi[0]) / 2, lo[1] - gap * 0.25, f"{label}\n幅 {size[k][0]:.1f} × 高さ {size[k][1]:.1f}",
                ha="center", va="top", fontsize=9, color="blue")
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    fig.savefig(png_path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    import sys
    render_views(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "views.png", "test")

