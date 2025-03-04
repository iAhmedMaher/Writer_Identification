import os
import FLAGS


def rename_images():
    writers_index = []
    writers_dict = {}

    with open(FLAGS.WRITER_IMG_PAIR_INFO_PATH, "r") as f:
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

    with open(FLAGS.WRITER_IMG_PAIR_INFO_PATH, "r") as f:
            for line in f:
                if line[0] != '#':
                    line_list = line.split(' ')
                    writer_index = int(line_list[1])

                    filename = os.path.join(FLAGS.TWO_FORM_DATASET_PATH, line_list[0] + ".png")

                    if writers_dict[writer_index] > 1:
                        new_filename = os.path.join(FLAGS.TWO_FORM_DATASET_PATH, str(writer_index) + "_" + line_list[0] + ".png")

                    else:
                        new_filename = os.path.join(FLAGS.TWO_FORM_DATASET_PATH, "X" + str(writer_index) + "_" + line_list[0] + ".png")

                    os.rename(filename, new_filename)


if __name__ == '__main__':
    rename_images()
