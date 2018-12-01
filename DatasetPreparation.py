import os

DATASET_PATH = r'E:\handwritten_dataset'
WRITER_IMG_PAIR_INFO_PATH = r'D:\Projects\Pattern_Recognition\Writer_Identification\forms.txt'


def rename_images():
    writers_index = []
    writers_dict = {}

    with open(WRITER_IMG_PAIR_INFO_PATH, "r") as f:
        for line in f:
            if line[0] != '#':
                line_list = line.split(' ')
                writer_index = int(line_list[1])

                if writer_index in writers_dict:
                    writers_dict[writer_index] += 1
                else:
                    writers_dict[writer_index] = 1

                if writer_index not in writers_index:
                    writers_index.append(writer_index)

    with open(WRITER_IMG_PAIR_INFO_PATH, "r") as f:
            for line in f:
                if line[0] != '#':
                    line_list = line.split(' ')
                    writer_index = int(line_list[1])

                    filename = os.path.join(DATASET_PATH, line_list[0] + ".png")

                    if writers_dict[writer_index] > 1:
                        new_filename = os.path.join(DATASET_PATH, str(writer_index) + "_" + line_list[0] + ".png")

                    else:
                        new_filename = os.path.join(DATASET_PATH, "X" + str(writer_index) + "_" + line_list[0] + ".png")

                    os.rename(filename, new_filename)


if __name__ == '__main__':
    rename_images()