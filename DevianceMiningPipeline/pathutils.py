import os, shutil
from pathlib import Path

def move_files(inp_folder, output_folder, split_nr, base):
    source = inp_folder
    dest1 = os.path.join(output_folder, "split"+str(split_nr), base)
    Path(dest1).mkdir(parents=True, exist_ok=True)
    files = os.listdir(source)
    for f in files:
        dstFile = os.path.join(dest1, f)
        if (os.path.exists(dstFile)): os.remove(dstFile)
        shutil.move(source + f, dstFile)