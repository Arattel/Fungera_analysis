lines = []
new_lines = []
with open('initial.gen', 'r') as file:
    lines = file.readlines()

lines = list(map(lambda x: x.strip(), lines))


def transform_line(line):
    first_line = ''.join(list(map(lambda x: f'{x}E{x}', line)))
    middle_line = f'E{"EE".join(line)}E'
    return [first_line, middle_line, first_line]


for line in lines:
    new_lines += transform_line(line)

with open('initial_interpreted.gen', 'w') as file:
    for line in new_lines:
        file.write(f'{line}\n')
