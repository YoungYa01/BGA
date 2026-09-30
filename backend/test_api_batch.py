import os
import json
import glob

# 🌟 依然从你的 API 模块中导入核心函数
from bga_void_seg import predict_and_generate_mask

def simulate_backend_batch_request():
    """模拟后端批量处理文件夹下所有图片的检测请求"""
    
    # ==========================================
    # 1. 配置输入输出路径
    # ==========================================
    # 1. 获取当前脚本所在的目录 (例如 .../BGA_Void_Detection/src)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    DIR = 'new-img'

    # 拼装精准的相对路径
    input_dir = os.path.join(SCRIPT_DIR, 'data', DIR) 
    
    # 如果不存在，就直接读取当前脚本所在的目录
    if not os.path.exists(input_dir):
        input_dir = SCRIPT_DIR 

    # YOLO 权重路径
    model_path = os.path.join(SCRIPT_DIR, 'best.pt')  

    # 输出结果保存到根目录下的 output 文件夹中
    out_dir = os.path.join(SCRIPT_DIR, 'output', DIR)
    
    # 获取支持的图片文件
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.lower().endswith(supported_formats)
    ]

    if not image_paths:
        print(f"❌ [Backend] 在目录 {input_dir} 下没有找到任何支持的图片文件！")
        return

    print("="*60)
    print(f"🚀 [Backend] 收到批量检测请求，准备开始流水线...")
    print(f"📁 [Backend] 图片来源: {input_dir}")
    print(f"📁 [Backend] 调试图输出: {out_dir}")
    print(f"🔍 [Backend] 共找到 {len(image_paths)} 张待测图片")
    print("="*60)

    # ==========================================
    # 2. 开始遍历图片进行批量检测
    # ==========================================
    # 准备全局统计变量
    global_pass_count = 0
    global_ng_count = 0
    global_solder_count = 0
    batch_results = {} # 存放所有图片的详细检测结果

    for i, img_path in enumerate(image_paths, 1):
        img_name = os.path.basename(img_path)
        print(f"\n▶️ [{i}/{len(image_paths)}] 正在处理: {img_name} ...")
        
        try:
            # 🌟 调用核心算法
            results = predict_and_generate_mask(
                model=model_path,               
                input_image_path=img_path, 
                conf_threshold=0.5,             
                save_debug_image=True,          # 依然为每张图保存 Debug 圈画图
                debug_output_dir=out_dir, 
                ng_threshold=0.25               
            )

            if results:
                # 统计当前图片的业务数据
                ng_count = sum(1 for item in results if item["void_rate"] > 0.25)
                pass_count = len(results) - ng_count
                
                # 累加到全局统计
                global_pass_count += pass_count
                global_ng_count += ng_count
                global_solder_count += len(results)
                
                # 保存该图片的详细数据到字典
                batch_results[img_name] = {
                    "solder_count": len(results),
                    "pass_count": pass_count,
                    "ng_count": ng_count,
                    "details": results # 存入底层 API 返回的完整字典
                }
                
                print(f"  => ✅ 处理成功! 提取 {len(results):03d} 颗焊球 (🟢 PASS: {pass_count:03d} | 🔴 NG: {ng_count:03d})")
            else:
                print(f"  => ⚠️ 未检测到任何有效焊球。")
                batch_results[img_name] = {"solder_count": 0, "pass_count": 0, "ng_count": 0, "details": []}

        except Exception as e:
            import traceback
            print(f"  => ❌ [异常] 处理 {img_name} 时底层算法抛出错误:")
            traceback.print_exc()
            batch_results[img_name] = {"error": str(e)}

    # ==========================================
    # 3. 打印最终的批量统计报告，并序列化 JSON
    # ==========================================
    print("\n" + "="*60)
    print("🎉 [Backend] 批量检测任务全部完成！")
    print(f"📊 [汇总报告] 总计处理图片: {len(image_paths)} 张")
    print(f"📊 [汇总报告] 检出焊球总数: {global_solder_count} 颗")
    print(f"📊 [汇总报告] 质量综合评定: 🟢 PASS {global_pass_count} 颗 | 🔴 NG {global_ng_count} 颗")
    
    # 🌟 将这批图片的所有 JSON 数据聚合成一个大报告写出
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    report_path = os.path.join(out_dir, 'batch_report.json')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, indent=4, ensure_ascii=False)
        
    print(f"📦 [Backend] 完整的数据分析大报告已保存至: {report_path}")
    print(f"💡 [提示] 请去 {out_dir} 查看所有 Debug 画圈图！")
    print("="*60 + "\n")

if __name__ == '__main__':
    simulate_backend_batch_request()