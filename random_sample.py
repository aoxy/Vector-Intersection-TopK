import random
import os

def create_directory(directory_path):
    try:
        os.mkdir(directory_path)
        print("文件夹创建成功：", directory_path)
    except FileExistsError:
        print("文件夹已经存在：", directory_path)
        raise

def random_sample_and_delete(input_file, output_file_path, sample_size):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    print(f"read {input_file} done")

    random.shuffle(lines)

    sample_lines = lines[:sample_size]
    # remaining_lines = lines[sample_size:]

    for i in range(0, sample_size):
        output_file_name = os.path.join(output_file_path, str(i)+".txt")
        with open(output_file_name, 'w') as file:
            file.write(sample_lines[i])
            print(f"write {output_file_name} done")

# 示例用法
input_file = "docs.txt"
# output_file_remaining = "/path/to/output_file_remaining.txt"
sample_size = 200
output_file_path = "querys_" + str(sample_size)

create_directory(output_file_path)

random_sample_and_delete(input_file, output_file_path, sample_size)
