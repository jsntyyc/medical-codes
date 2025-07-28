import glob

# 在 /mnt/nvme4 下递归查找 patients.csv.gz
paths = glob.glob("/mnt/nvme4/**/*.csv.gz", recursive=True)
patients_paths = [p for p in paths if p.endswith("patients.csv.gz")]
print("找到的 patients.csv.gz 路径：")
for p in patients_paths:
    print(" ", p)