def get_writer_form_id(filename):
    i = 0
    while filename[i] != '_':
        i += 1

    return filename[:i], filename[i+1:len(filename)-4]
