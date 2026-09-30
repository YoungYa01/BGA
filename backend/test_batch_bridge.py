# -*- coding: utf-8 -*-
"""
🌟 BGA 焊点【仅桥连短路】批量测试脚本 (Bridge Only Batch Test)
--------------------------------------------------------------
调用函数: inspect_bga_bridge()
质检维度: 仅检测焊球间桥连短路缺陷 (跳过气泡分割，耗时极短)
"""
import os
import sys
import json
import time

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from bga_pipeline import inspect_bga_bridge


def run_batch_bridge_test():
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(backend_dir)
    weights_path = os.path.join(backend_dir, "best.pt")
    input_imgs_dir = os.path.join(project_dir, "data", "imgs")
    output_dir = os.path.join(backend_dir, "output", "bridge_only")
    os.makedirs(output_dir, exist_ok=True)

    supported_exts = (".jpg", ".jpeg", ".png", ".bmp")
    all_image_paths = [
        os.path.join(input_imgs_dir, f)
        for f in sorted(os.listdir(input_imgs_dir))
        if f.lower().endswith(supported_exts)
    ]

    if not all_image_paths:
        print(f"[ERROR] 目录中未找到任何图片: {input_imgs_dir}")
        return

    print("=" * 70)
    print("🚀 BGA 焊点【仅桥连短路】批量质检测试启动 (Bridge Only)")
    print(f"📦 权重模型: {weights_path}")
    print(f"📂 待测目录: {input_imgs_dir} ({len(all_image_paths)} 张)")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 70)

    report_list = []
    total_start = time.time()
    total_pass = 0
    total_ng = 0

    for idx, img_path in enumerate(all_image_paths, start=1):
        fname = os.path.basename(img_path)
        print(f"\n[{idx}/{len(all_image_paths)}] 正在巡检: {fname} ...")

        try:
            res = inspect_bga_bridge(
                input_image_path=img_path,
                weights_path=weights_path,
                conf_threshold=0.50,
                device="0",
                save_debug_image=True,
                debug_output_dir=output_dir,
            )

            status = res["board_status"]
            summary = res["summary"]
            bridge_cnt = summary["bridge_defect_count"]
            elapsed = summary["elapsed_ms"]

            if status == "PASS":
                total_pass += 1
                status_str = "✅ PASS"
            else:
                total_ng += 1
                status_str = "❌ NG"

            print(f"    -> 判定结果: {status_str} | 耗时: {elapsed:.1f}ms | 焊球: {summary['solder_count']} 个 | 桥连缺陷: {bridge_cnt} 处")
            for r in res["ng_reasons"]:
                print(f"       ⚠️ {r}")

            report_list.append({
                "file_name": fname,
                "board_status": status,
                "summary": summary,
                "ng_reasons": res["ng_reasons"],
                "visual_image": res["visual_output_path"],
            })

        except Exception as e:
            print(f"    ❌ 检测失败: {e}")
            import traceback
            traceback.print_exc()

    total_time = round(time.time() - total_start, 2)
    print("\n" + "=" * 70)
    print("🏁 【仅桥连】批量测试完成！")
    print(f"   总测试图片: {len(all_image_paths)} 张 | PASS: {total_pass} 张 | NG: {total_ng} 张")
    print(f"   总耗时: {total_time} 秒 (平均每张: {round(total_time / len(all_image_paths), 2)} 秒)")
    print(f"   标注图已保存至: {output_dir}")

    report_json = os.path.join(output_dir, "bridge_batch_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report_list, f, ensure_ascii=False, indent=2)
    print(f"   结构化报告已保存至: {report_json}")
    print("=" * 70)


if __name__ == "__main__":
    run_batch_bridge_test()
