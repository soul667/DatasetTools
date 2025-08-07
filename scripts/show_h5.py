import h5py

with h5py.File("your_file.h5", "r") as f:
    def print_structure(name, obj):
        print(name)
    f.visititems(print_structure)  # 打印文件结构

    # 读取一个数据集
    data = f["some/dataset"][()]
    print(data)
