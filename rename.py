import os
import re
from glob import glob


def rename_folder(folder_name):
    folder_name = re.sub(r"^(\d+)", lambda m: m.group(1).ljust(14, "0"), folder_name)
    folder_name = re.sub(
        r"(lexpr)(\d+)", lambda m: m.group(1) + m.group(2).zfill(2), folder_name
    )
    folder_name = re.sub(r"_(\d+)$", lambda m: "_" + m.group(1).zfill(4), folder_name)

    return folder_name


folders = glob("checkpoints/*")
for folder in folders:
    new_folder_name = rename_folder(folder)
    os.rename(folder, new_folder_name)
    print(f"Renamed {folder} to {new_folder_name}")
