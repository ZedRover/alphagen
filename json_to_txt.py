import json
import os
import re


def get_latest_json_file(subdir):
    # 获取所有匹配格式的文件
    json_files = [f for f in os.listdir(subdir) if re.match(r"\d+_steps_pool\.json", f)]

    # 如果没有匹配的文件，返回 None
    if not json_files:
        return None

    # 从文件名中提取 time_steps 数字并排序
    json_files.sort(key=lambda x: int(x.split("_")[0]), reverse=True)

    # 返回 time_steps 数字最大的文件
    return os.path.join(subdir, json_files[0])


def combine_json_to_txt(checkpoints_dir, output_txt, selected_files_path=None):
    selected_dirs = []
    if selected_files_path:
        with open(selected_files_path, "r") as sf:
            selected_dirs = [line.strip() for line in sf]
    print(len(selected_dirs))

    with open(output_txt, "w") as txt_file:
        # 遍历 checkpoints 文件夹下的所有子目录
        for subdir in [
            os.path.join(checkpoints_dir, d)
            for d in os.listdir(checkpoints_dir)
            if os.path.isdir(os.path.join(checkpoints_dir, d))
        ]:
            # 如果提供了 selected_files_path，只处理选定的子目录
            if selected_dirs and os.path.basename(subdir) not in selected_dirs:
                print(f"Skip {subdir}")
                continue

            json_file = get_latest_json_file(subdir)
            if json_file:
                with open(json_file, "r") as jf:
                    data = json.load(jf)
                    exprs = data.get("exprs", [])
                    weights = data.get("weights", [])

                    # 将 exprs 和 weights 组合成一个公式
                    formula = " + ".join(
                        [f"({weights[i]} * {exprs[i]})" for i in range(len(exprs))]
                    ).replace("$", "")
                    txt_file.write(formula + "\n")


# 使用函数
checkpoints_dir = "checkpoints_10d"
output_txt = "ret10d_90.txt"
selected_files = "/home/ray/workspace/signal_research/notebook/ret10d_90.txt"  # 可选参数
combine_json_to_txt(checkpoints_dir, output_txt, selected_files_path=selected_files)