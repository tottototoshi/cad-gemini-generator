import os
import io
import re
import sys
import time
import datetime
import subprocess
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
import ezdxf
from ezdxf.addons.drawing import matplotlib as ezdxf_matplotlib
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.config import Configuration, BackgroundPolicy, ColorPolicy
import matplotlib
matplotlib.use("Agg")  # GUI不要のバックエンド
import matplotlib.pyplot as plt
import argparse

# スクリプトのフォルダを基準に動作させる（どこから実行しても input/ output/ model_views.py が見つかるように）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ============================================================
# コマンドライン引数の設定
# ============================================================
parser = argparse.ArgumentParser(description="DXFから3Dモデル(STEP)を生成するスクリプト")
parser.add_argument("dxf_path", nargs="?", default="input/24-56-02-106_従動ケース.dxf", help="処理するDXFファイルのパス")
parser.add_argument("--hints", help="補足指示テキストのパス (省略時は 同じフォルダの DXF名_hints.txt を探します)")
args = parser.parse_args()

DXF_PATH = args.dxf_path
if not os.path.exists(DXF_PATH):
    print(f"エラー: DXFファイル '{DXF_PATH}' が見つかりません。")
    sys.exit(1)

# ヒントファイルパスの自動設定（指定がなければ DXFファイル名_hints.txt を探す）
HINTS_PATH = args.hints
if not HINTS_PATH:
    base, _ = os.path.splitext(DXF_PATH)
    HINTS_PATH = base + "_hints.txt"

# ============================================================
# その他の設定
# ============================================================
OUTPUT_DIR = "output"

# 画像化・形状抽出から除外するレイヤー（図枠・表題欄など。図面ごとに調整してください）
EXCLUDE_LAYERS_FOR_IMAGE = {"ZUWAKU", "表題", "KCAD_ZUWAKU", "DEFPOINTS"}
# 形状データ(線・円など)の抽出から除外するレイヤー（バルーンの引出線なども除外）
EXCLUDE_LAYERS_FOR_GEOMETRY = EXCLUDE_LAYERS_FOR_IMAGE | {"KCAD_BALL"}
# 文字化け(□)対策: 全文字スタイルをこのフォントに置き換える
JP_FONT = "msgothic.ttc"

MAX_IMAGE_SIZE = 3072      # 送信画像の最大辺(px)
MAX_FIX_ATTEMPTS = 3       # 実行エラー時にGeminiへ修正させる最大回数(通算)
VISUAL_VERIFY_ROUNDS = 2   # 実行成功後、モデルの三面図画像を図面と比較させて修正させる最大回数(0で無効)
API_RETRIES = 3
API_RETRY_DELAY = 5        # 秒

# 試行するモデルの優先順位（賢いProモデルから順に試し、最終的に実績のあるFlashにフォールバック）
MODELS_TO_TRY = [
    'gemini-3.1-pro-preview', # 最新のプレビュー版Pro（最も推論・空間認識に優れる）
    'gemini-2.5-pro',         # 安定版のProモデル
    'gemini-pro-latest',      # 最新Proへの自動振り分け
    'gemini-3.8-flash'        # 最終手段（動作実績あり）
]

# ============================================================
# 1. 環境変数の読み込み / クライアントの準備
# ============================================================
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("エラー: APIキーが読み込めていません！(.envを確認してください)")
    sys.exit(1)

print(f"APIキーを読み込みました (先頭: {api_key[:5]}...)")
client = genai.Client(api_key=api_key)

if not os.path.exists(DXF_PATH):
    print(f"エラー: DXFファイル '{DXF_PATH}' が見つかりません。inputフォルダに配置してください。")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# 2. DXF → 画像 (白背景・黒線・日本語フォント・図枠除外)
# ============================================================
def render_dxf_to_image(doc) -> Image.Image:
    # 文字化け対策: SHXフォント(TXT/BIGFONT)などをTrueTypeの日本語フォントに置き換え
    for style in doc.styles:
        style.dxf.font = JP_FONT
        if style.dxf.hasattr("bigfont"):
            style.dxf.discard("bigfont")

    msp = doc.modelspace()
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_axes([0, 0, 1, 1])

    config = Configuration(
        background_policy=BackgroundPolicy.WHITE,  # 図面部分の背景も白に
        color_policy=ColorPolicy.BLACK,            # 線・文字を黒に統一（細い色線は読みにくいため）
        lineweight_scaling=1.5,
    )
    frontend = Frontend(RenderContext(doc), ezdxf_matplotlib.MatplotlibBackend(ax), config=config)
    frontend.draw_layout(
        msp,
        finalize=True,
        filter_func=lambda e: e.dxf.layer not in EXCLUDE_LAYERS_FOR_IMAGE,
    )

    buf = io.BytesIO()
    # bbox_inches='tight' で余白を切り落とし、図形部分の解像度を確保
    fig.savefig(buf, format="png", dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")
    if max(img.size) > MAX_IMAGE_SIZE:
        img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
        print(f"送信サイズ最適化のため画像をリサイズしました: {img.size}")
    return img


# ============================================================
# 3. DXF → テキストデータ (正確な座標・寸法値・文字)
# ============================================================
def _p(v, nd=2):
    """座標を丸めて文字列化"""
    return "(" + ", ".join(f"{round(c, nd):g}" for c in tuple(v)[:2]) + ")"


def _linetype(e, doc):
    """実線/中心線/かくれ線などの線種を取得 (BYLAYERならレイヤーの線種)"""
    lt = e.dxf.get("linetype", "BYLAYER")
    if lt.upper() == "BYLAYER":
        try:
            lt = doc.layers.get(e.dxf.layer).dxf.linetype
        except Exception:
            lt = "CONTINUOUS"
    return lt


def _iter_entities(entities):
    """ブロック参照(INSERT)は分解して中身を列挙"""
    for e in entities:
        if e.dxftype() == "INSERT":
            try:
                yield from _iter_entities(e.virtual_entities())
            except Exception:
                continue
        else:
            yield e


def extract_dxf_info(doc) -> str:
    msp = doc.modelspace()
    geom_lines = []
    drawing_texts = []
    title_texts = []

    for e in _iter_entities(msp):
        t = e.dxftype()
        layer = e.dxf.layer
        try:
            if t in ("TEXT", "MTEXT", "ATTRIB"):
                txt = e.plain_text() if t == "MTEXT" else e.dxf.text
                txt = txt.strip()
                if not txt:
                    continue
                rot = e.dxf.get("rotation", 0)
                rec = f"'{txt}' at={_p(e.dxf.insert, 1)}" + (f" rot={rot:g}" if rot else "") + f" layer={layer}"
                (title_texts if layer in EXCLUDE_LAYERS_FOR_IMAGE else drawing_texts).append((e.dxf.insert, rec))
                continue

            if layer in EXCLUDE_LAYERS_FOR_GEOMETRY:
                continue

            if t == "DIMENSION":
                m = e.get_measurement()
                mval = f"{m:.2f}" if isinstance(m, (int, float)) else str(m)
                geom_lines.append(f"DIM value={mval} text='{e.dxf.get('text', '')}' defpoint={_p(e.dxf.defpoint)} layer={layer}")
            elif t == "LINE":
                geom_lines.append(f"LINE {_p(e.dxf.start)}->{_p(e.dxf.end)} lt={_linetype(e, doc)} layer={layer}")
            elif t == "CIRCLE":
                geom_lines.append(f"CIRCLE c={_p(e.dxf.center)} r={e.dxf.radius:.3g} lt={_linetype(e, doc)} layer={layer}")
            elif t == "ARC":
                geom_lines.append(
                    f"ARC c={_p(e.dxf.center)} r={e.dxf.radius:.3g} "
                    f"{e.dxf.start_angle:.0f}->{e.dxf.end_angle:.0f}deg lt={_linetype(e, doc)} layer={layer}")
            elif t == "LWPOLYLINE":
                pts = [_p(p) for p in e.get_points("xy")]
                geom_lines.append(f"PLINE closed={e.closed} pts=[{', '.join(pts)}] layer={layer}")
            elif t == "POLYLINE":
                pts = [_p(v.dxf.location) for v in e.vertices]
                geom_lines.append(f"PLINE closed={e.is_closed} pts=[{', '.join(pts)}] layer={layer}")
            elif t == "ELLIPSE":
                geom_lines.append(f"ELLIPSE c={_p(e.dxf.center)} major={_p(e.dxf.major_axis)} ratio={e.dxf.ratio:.3g} layer={layer}")
        except Exception:
            continue

    # 表題欄・部品表のテキストは「同じ行(Y)ごと」に並べると表として読みやすい
    title_texts.sort(key=lambda r: (-round(r[0][1]), r[0][0]))
    drawing_texts.sort(key=lambda r: (-r[0][1], r[0][0]))

    return (
        "■ 図面内の文字（寸法値・注記・引出線の指示。at=文字の配置座標）\n"
        + "\n".join(r for _, r in drawing_texts)
        + "\n\n■ 表題欄・部品表の文字（Y座標が同じものが同じ行）\n"
        + "\n".join(r for _, r in title_texts)
        + "\n\n■ 図形要素（座標単位mm。lt=線種: CONTINUOUS=外形線, CENTER系=中心線, HIDDEN/DASHED系=かくれ線）\n"
        + "\n".join(geom_lines)
    )


print("DXFファイルを読み込んでいます...")
try:
    doc = ezdxf.readfile(DXF_PATH)
    dxf_text = extract_dxf_info(doc)  # フォント置換前に抽出
    print(f"DXFデータを抽出しました ({len(dxf_text.splitlines())} 行)")

    print("DXFファイルをPNG画像に変換しています...（初回はフォントキャッシュ作成で時間がかかります）")
    drawing_image = render_dxf_to_image(doc)
    drawing_image.save("input/debug_converted.png")  # デバッグ用：送信する画像を保存
    with open("input/debug_dxf_extract.txt", "w", encoding="utf-8") as f:
        f.write(dxf_text)
    print("画像変換が完了しました。(input/debug_converted.png, input/debug_dxf_extract.txt)")
except Exception as e:
    print(f"DXFの読み込み・変換中にエラーが発生しました:\n{e}")
    sys.exit(1)


# ============================================================
# 4. プロンプト
# ============================================================
# 図面ごとの補足指示 (input/hints.txt があれば最優先事項として追加)
hints_text = ""
if os.path.exists(HINTS_PATH):
    with open(HINTS_PATH, encoding="utf-8") as f:
        hints_text = "\n".join(
            line for line in f.read().splitlines() if line.strip() and not line.lstrip().startswith("#")
        )
    if hints_text:
        print(f"補足指示を読み込みました: {HINTS_PATH}")
hints_section = f"""
【ユーザーからの補足指示（最優先。図面の解釈がこれと矛盾する場合はこちらに従う）】
{hints_text}
""" if hints_text else ""

prompt = f"""
添付した機械図面（画像）と、同じDXFから抽出した正確な数値データをもとに、
3D形状を作成するPythonスクリプトを作成してください。
CADライブラリとして「CadQuery (import cadquery as cq)」を使用してください。
{hints_section}
【図面の前提】
- 日本の機械製図(JIS)で、投影法は「第三角法」です。
  正面図の上に平面図、右に右側面図、左に左側面図が配置されます。「A断面」「A矢視」などは補助的な図です。
- 部品表(表題欄)に各部品の品名・板厚・材質が記載されています。板金(鋼板 t○○)や平鋼(t○○x幅)を
  溶接した構造物の場合は、部品表の板厚を正しく使い、部品ごとに作成してから union で結合してください。
- かくれ線(HIDDEN/DASHED)は見えない部分の形状、中心線(CENTER)は穴・対称軸の位置を示します。
- 寸法値は「DXF抽出データ」の文字(寸法値)と座標を最優先してください。画像はビューの配置や形状の把握に使ってください。
- ( )付きの寸法は参考寸法です。「2-11キリ」は直径11の穴が2個、「2-C5」はC5面取りが2箇所、「M6タップ」はM6ねじ穴(下穴φ5程度でよい)を意味します。

【3D座標系の約束（必ず守ること）】
- 正面図 = -Y側から見た図（正面図の右 = +X、上 = +Z）
- 平面図 = +Z側から見下ろした図（右 = +X、上 = +Y = 奥側）
- 右側面図 = +X側から見た図（右 = +Y = 奥側、上 = +Z）
- 左側面図 = -X側から見た図（右 = -Y = 手前側、上 = +Z）
※ 作成したモデルはこの約束どおりに三面図として描画し、元の図面と比較します。

【寸法の読み方の注意】
- 寸法の起点・終点は、寸法補助線が「どの線から出ているか」で判断してください。
  外形線から出ていれば外形まで、中心線（穴・円弧の中心）から出ていれば中心までの距離です。
  例: 長穴の寸法補助線が両端の円弧の中心線から出ていれば、その寸法は「中心間距離」です。長穴の外側から出ていれば「全長」です。
- 同じ方向に並んだ寸法（直列寸法）は、合計が全体寸法と一致するか検算し、各寸法がどこからどこまでかを確定してください。
- 片側からの寸法（例: 上から65）と反対側からの寸法（例:下から60）を取り違えないでください。
- 各部品の位置は、正面図だけでなく平面図・側面図・断面図でも確認してください（特に奥行方向の位置）。
- 図面に描かれていない穴・切り欠き・形状を推測で追加しないでください。ある図に穴が見えても、
  別の部品の穴が透けて見えているだけの場合があります。どのビューのどの線が根拠かを確認してください。

【STEP 1: 解析（コードを書く前に必ず実行し、テキストで書き出す）】
- 各ビューがどれか（正面図・平面図・側面図など）
- 基準となる全体の原点(0,0,0)
- 全体の外形寸法 (X×Y×Z)
- 部品ごとの形状・板厚・XYZ寸法・配置座標（根拠となる寸法値も併記）
- 全ての穴・スロット・カットアウトのリスト（中心座標、サイズ、貫通方向、どのビューの何が根拠か）
- 直列寸法の検算結果

【STEP 2: モデリングのルール】
1. STEP 1で整理した座標と寸法に忠実にモデリングしてください。
2. cut() する工具形状は、貫通方向に十分長く（例: 対象の厚み+20mm以上）作ってください。
3. 長穴(スロット)を `slot2D(length, width)` で作成する場合、第一引数 `length` は「長穴の全長（端から端まで）」を指定する仕様です。中心間距離ではないため、図面の寸法が「全長」を示している場合はそのままの値を指定してください。
4. 【重要】図面に「深サ」や「ザグリ」の記載がある場合、あるいは「かくれ線」で表現されている円・形状は、出っ張り（extrude）ではなく「へこみ・穴」を意味します。必ず `cut()` を用いて対象を削り取る加工としてモデリングしてください。
5. 【重要】部品の端にある「U字型の切り欠き（片側が開放された形状）」に対しては絶対に `slot2D` を使用しないでください（`slot2D` は完全に閉じた両丸の長穴しか作れません）。端部の開放された切り欠きは、端からはみ出すように十分な大きさの図形を描き `cut()` で切り抜いてください。
6. edges() や faces() では複雑な論理式（例: "|X and <Z"）を使わず、`.edges("<Z").edges("|X")` のようにメソッドチェーンで絞り込んでください。
7. 板金部品の「曲げR」について：図面に明確なR寸法の文字記載がない場合は、一般常識に従い「内R ＝ 板厚(t)」「外R ＝ 板厚(t) ＋ 板厚(t)」とみなして `fillet()` で曲げRをモデリングしてください。
8. 最終的な3Dモデルは「result」という変数(cq.Workplane)に格納してください。
9. cq.exporters.export などの保存処理やshow_objectは書かないでください（システム側で行います）。

【出力形式】
「## 解析」の見出しの後にSTEP 1の内容を書き、最後に ```python のコードブロックを「1つだけ」出力してください。

==================== DXF抽出データ ====================
{dxf_text}
=======================================================
"""


# ============================================================
# 5. Gemini呼び出し (モデル自動探索 + リトライ + チャット履歴で修正依頼)
# ============================================================
def make_config(model_name):
    # Gemini 3系は temperature を既定値(1.0)のまま使うことが推奨されているため指定しない
    if model_name.startswith("gemini-3"):
        return None
    return types.GenerateContentConfig(temperature=0.2)


def start_chat(contents):
    """モデルを優先順に試してチャットを開始し、(chat, response, model_name) を返す"""
    for attempt in range(API_RETRIES):
        for model_name in MODELS_TO_TRY:
            try:
                print(f"モデル '{model_name}' に接続しています...")
                chat = client.chats.create(model=model_name, config=make_config(model_name))
                response = chat.send_message(contents)
                if response and response.text:
                    print(f"★ 成功: {model_name} から応答を受信しました！")
                    return chat, response, model_name
            except Exception as e:
                if "404" in str(e) or "NOT_FOUND" in str(e):
                    print(f" -> {model_name} は利用できませんでした。次のモデルを探します...")
                else:
                    print(f" -> {model_name} 実行中にエラー: {e}")
        if attempt < API_RETRIES - 1:
            print(f"{API_RETRY_DELAY}秒後に再試行します...")
            time.sleep(API_RETRY_DELAY)
    return None, None, None


def send_followup(chat, message):
    """同じチャット(画像・これまでの会話を保持)に追加メッセージを送る"""
    for attempt in range(API_RETRIES):
        try:
            response = chat.send_message(message)
            if response and response.text:
                return response
        except Exception as e:
            print(f" -> 送信エラー: {e}")
        time.sleep(API_RETRY_DELAY)
    return None


def extract_code(text):
    """最後の ```python ブロックを取り出す（解析部分に別のコード片があっても誤抽出しない）"""
    blocks = re.findall(r"```(?:python|py)[ \t]*\r?\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    # 言語指定なしのブロックしかない場合（開始・終了を順にペアにする）
    parts = text.split("```")
    if len(parts) >= 3:
        candidates = [parts[i] for i in range(1, len(parts) - 1, 2)]
        code = candidates[-1]
        return code.split("\n", 1)[1].strip() if code and not code.startswith("\n") and "\n" in code else code.strip()
    return None


# ============================================================
# 6. 生成コードの保存・実行
# ============================================================
FORCE_SAVE_CODE = r'''

# --- ここから下はシステム側で自動追加された保存処理 ---
import os as _os
import cadquery as _cq

_os.makedirs("__OUTPUT_DIR__", exist_ok=True)
_filename = "__STEP_PATH__"

_target = None
for _name in ("result", "model", "shape"):
    if globals().get(_name) is not None:
        _target = globals()[_name]
        break

if _target is None:
    raise RuntimeError("保存する3Dモデル(result変数)が見つかりませんでした。")

_cq.exporters.export(_target, _filename)
print(f"STEPデータを保存しました: {_filename}")

# 形状チェック用のサマリー（Geminiへの自己チェックにも使用）
_shape = _target.findSolid() if isinstance(_target, _cq.Workplane) else _target
_bb = _shape.BoundingBox()
print("=== MODEL SUMMARY ===")
print(f"bbox_min=({_bb.xmin:.1f}, {_bb.ymin:.1f}, {_bb.zmin:.1f}) bbox_max=({_bb.xmax:.1f}, {_bb.ymax:.1f}, {_bb.zmax:.1f})")
print(f"size X={_bb.xlen:.1f} Y={_bb.ylen:.1f} Z={_bb.zlen:.1f}")
print(f"volume={_shape.Volume():.0f} mm3, solids={len(_shape.Solids())}, faces={len(_shape.Faces())}")
print("=== END SUMMARY ===")
'''


def run_generated_code(code, round_no):
    script_path = os.path.join(OUTPUT_DIR, f"generated_script_{RUN_ID}_r{round_no}.py")
    step_path = f"{OUTPUT_DIR}/output_model_{RUN_ID}_r{round_no}.step"
    save_code = FORCE_SAVE_CODE.replace("__OUTPUT_DIR__", OUTPUT_DIR).replace("__STEP_PATH__", step_path)

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code + "\n" + save_code)
    # 最新版を固定名でもコピー（従来の generated_script.py と互換）
    with open(os.path.join(OUTPUT_DIR, "generated_script.py"), "w", encoding="utf-8") as f:
        f.write(code + "\n" + save_code)

    print(f"生成スクリプトを実行しています: {script_path}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, script_path],  # 現在と同じPython環境で実行
            capture_output=True, text=True, encoding="utf-8", env=env, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, "", "実行が300秒でタイムアウトしました（無限ループや重すぎる処理の可能性）。", step_path

    ok = proc.returncode == 0 and os.path.exists(step_path)
    return ok, proc.stdout.strip(), proc.stderr.strip(), step_path


def save_response_log(text, round_no):
    path = os.path.join(OUTPUT_DIR, f"gemini_response_{RUN_ID}_r{round_no}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ============================================================
# 7. メイン処理: 生成 → 実行 → (エラーなら修正依頼) → (成功なら三面図で目視比較・修正)
# ============================================================
def render_model_views(step_path, round_no):
    """生成モデルを三面図PNGにして返す（失敗時は None）"""
    try:
        import model_views  # cadquery の読み込みに時間がかかるため必要時のみ import
        png_path = os.path.join(OUTPUT_DIR, f"model_views_{RUN_ID}_r{round_no}.png")
        model_views.render_views(step_path, png_path, title=f"生成モデルの三面図 (round {round_no})")
        print(f"生成モデルの三面図を保存しました: {png_path}")
        return png_path
    except Exception as e:
        print(f"三面図の描画に失敗しました（目視比較をスキップします）: {e}")
        return None


VERIFY_MESSAGE = """
生成コードは正常に実行されました。添付画像は、作成した3Dモデルを【3D座標系の約束】どおりに投影した図です
（黒の実線=見える線、灰色の破線=かくれ線。各図の下に外形寸法を表示）。
最初に送った元の図面の、対応するビュー（正面図・平面図・右側面図・左側面図）と1つずつ見比べてください。

{summary}

【確認項目】
1. 各ビューの外形の大きさ・形が元の図面と一致しているか（座標系の約束どおりの向きになっているか）
2. 穴・長穴・切り欠きの「数・位置・大きさ・長さ」が一致しているか（元の図面にない穴や切り欠きがないか）
3. 各部品の位置（特に奥行方向）と、かくれ線の位置が元の図面と一致しているか
4. 面取りなどの細部

【回答ルール】
- 食い違いがあれば、「どのビューの・何が・図面ではどうなっているか（根拠の寸法値）」を箇条書きで説明し、
  修正した完全なコードを ```python ブロック1つで出力してください。
- 【重要】画像の比較は「穴の抜け忘れ」「貫通方向の間違い」「向きの反転」などの大きな形状エラーを発見するために行います。寸法や穴の位置は、最初の解析でDXFテキストデータから計算した数値の方が正しいことが多いため、画像の見た目の錯覚（線の重なりなど）だけで安易に座標計算を変更しないでください。明確な寸法値の読み落としがあった場合のみ数値を修正してください。
- 元の図面に明確な根拠がない変更はしないでください。一致している部分は変更しないでください。
- すべて一致していれば、コードブロックを付けずに「OK」とだけ答えてください。
"""

print("Geminiに図面データとリクエストを送信中...")
chat, response, used_model = start_chat([prompt, drawing_image])
if not response:
    print("最大再試行回数に達しました。処理を中断します。")
    sys.exit(1)

final_step = None
final_views = None
fix_count = 0      # 実行エラー修正の回数
verify_count = 0   # 目視比較による修正依頼の回数
round_no = 1
max_rounds = 1 + MAX_FIX_ATTEMPTS + VISUAL_VERIFY_ROUNDS + 2  # 安全上限

while response and round_no <= max_rounds:
    save_response_log(response.text, round_no)
    code = extract_code(response.text)
    if not code:
        if final_step:
            # 目視比較で「OK」と判断された（コードなし）
            print("Geminiの目視チェック: 図面と一致していると判断されました。")
            break
        print("応答からPythonコードを抽出できませんでした。コードの出力を再依頼します...")
        response = send_followup(chat, "```python のコードブロックを1つだけ出力してください。")
        round_no += 1
        continue

    ok, stdout, stderr, step_path = run_generated_code(code, round_no)
    if stdout:
        print(stdout)

    if ok:
        final_step = step_path
        views_png = render_model_views(step_path, round_no)
        final_views = views_png or final_views
        if verify_count >= VISUAL_VERIFY_ROUNDS or views_png is None:
            break
        verify_count += 1
        summary = stdout[stdout.find("=== MODEL SUMMARY ==="):] if "=== MODEL SUMMARY ===" in stdout else ""
        print(f"\n三面図をGeminiに送り、元の図面と比較させています... ({verify_count}/{VISUAL_VERIFY_ROUNDS})")
        response = send_followup(chat, [VERIFY_MESSAGE.format(summary=summary), Image.open(views_png)])
    else:
        print("\n--- スクリプト実行時エラー ---")
        print(stderr[-3000:])
        print("------------------------------")
        if fix_count >= MAX_FIX_ATTEMPTS:
            print("エラー修正の最大回数に達しました。")
            break
        fix_count += 1
        print(f"エラー内容をGeminiに送り、修正を依頼します... ({fix_count}/{MAX_FIX_ATTEMPTS})")
        response = send_followup(chat, f"""
生成したコードを実行したところ、以下のエラーが発生しました。原因を特定し、修正した完全なコードを ```python ブロック1つで出力してください。
fillet/chamfer/セレクタが原因の場合は、その処理を単純化または省略してかまいません。

--- エラー(末尾) ---
{stderr[-3000:]}
""")
    round_no += 1

print("\n==============================")
if final_step:
    print(f"★ 完了しました！ 最終STEPファイル: {final_step}  (使用モデル: {used_model})")
    if final_views:
        print(f"  最終モデルの三面図: {final_views}")
    print(f"  Geminiの解析・修正内容は {OUTPUT_DIR}/gemini_response_{RUN_ID}_r*.md で確認できます。")
    print("  ※ 修正で悪化する場合もあるため、各roundの三面図(model_views_*.png)を見比べて最良のSTEPを選んでください。")
else:
    print("STEPファイルを生成できませんでした。output フォルダ内の生成スクリプトと応答ログを確認してください。")
