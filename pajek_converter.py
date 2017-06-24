def convert_pajek(fn):
    with open(fn) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    vertices_start = 0
    vertices_end = 0
    edges_start = 0
    edges_end = len(content)
    for idx in range(len(content)):
        if content[idx][1:9] == "Vertices":
            vertices_start = idx+1
            break
    for idx in range(len(content)):
        if content[idx][1:5] == "Arcs":
            vertices_end = idx
            edges_start = idx+1
            break
    vertices = {}
    for idx in range(vertices_start,vertices_end):
        v_w = content[idx].split(" ", 1)
        vertices[v_w[0]] = v_w[1]
    edges = []
    for idx in range(edges_start, edges_end):
        v1_v2_f = content[idx].split(" ")
        for f in range(int(v1_v2_f[2])):
            if vertices[v1_v2_f[1]] != '""':
                edges.append(vertices[v1_v2_f[0]]+";"+vertices[v1_v2_f[1]]+';"x";"x"')
    with open(fn+"_plain", 'w') as f:
        f.writelines('"cue";"asso1";"asso2";"asso3"\n')
        for e in edges:
            f.writelines(e+"\n")

if __name__ == "__main__":
    convert_pajek("./EAT/shrunkEAT.net")