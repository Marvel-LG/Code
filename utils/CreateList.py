import random

file_path = "test.flist"
with open(file_path, "r") as file:

    lines = file.readlines()

random_lines = random.sample(lines, 1000)

new_file_path = "valid.flist"
with open(new_file_path, "w") as new_file:
    for line in random_lines:
        new_file.write(line)