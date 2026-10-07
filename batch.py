import os
import glob
import subprocess
import sys
import time

def main():
    input_dir = "input"
    
    # inputフォルダ内のすべてのDXFファイルを取得
    dxf_files = glob.glob(os.path.join(input_dir, "*.dxf"))
    
    if not dxf_files:
        print(f"{input_dir} フォルダにDXFファイルが見つかりません。")
        return

    print(f"合計 {len(dxf_files)} 個のDXFファイルを見つけました。")
    print("-" * 40)

    success_count = 0
    fail_count = 0

    for i, dxf_path in enumerate(dxf_files, 1):
        print(f"\n[{i}/{len(dxf_files)}] 処理開始: {dxf_path}")
        
        # main.py を別プロセスとして呼び出す
        # タイムアウトや予期せぬクラッシュが起きても次のファイルに進めるように subprocess.run を使用
        try:
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            result = subprocess.run(
                [sys.executable, "main.py", dxf_path],
                check=True,
                env=env
            )
            print(f"[{i}/{len(dxf_files)}] 成功: {dxf_path}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"[{i}/{len(dxf_files)}] 失敗 (エラーコード {e.returncode}): {dxf_path}")
            fail_count += 1
        except Exception as e:
            print(f"[{i}/{len(dxf_files)}] 予期せぬエラー: {dxf_path}\n{e}")
            fail_count += 1
            
        # API制限（RPM）を回避するための少しの待機（必要に応じて調整）
        time.sleep(5)

    print("\n" + "=" * 40)
    print("一括処理が完了しました！")
    print(f"成功: {success_count}件, 失敗: {fail_count}件")
    print("=" * 40)

if __name__ == "__main__":
    main()

