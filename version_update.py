from mtutils import file_read_lines
from mtutils import file_write_lines

file_path = 'setup.py'
lines = file_read_lines(file_path)
key_word = 'str_version'

for index, line in enumerate(lines):
    if len(line) > len(key_word):
        if line[:len(key_word)] == key_word:
            pre, last = line.split('=')
            main, minor, tiny = last.split('.')
            tiny = tiny.replace("'", '')
            tiny = str(int(tiny) + 1)
            line_str = '='.join([pre, '.'.join([main, minor, tiny + "'"])])
            lines[index] = line_str
            break

file_write_lines(lines, file_path, overwrite=True)

pass