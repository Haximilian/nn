import jinja2

with open('network.template.hpp') as f:
    t = jinja2.Template(f.read())

size = [(0, 2), (1, 8), (2, 8), (3, 2)]
dimensions = [(0, 8, 2), (1, 8, 8), (2, 2, 8)]
dimensions = [(i, R, C + 1) for (i, R, C) in dimensions]

print(t.render({"dimensions": dimensions, "size": size}))