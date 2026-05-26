from manim import *
import numpy as np

from visualization.projections import nested_cube


class NCubeVisualizer(ThreeDScene):
    dim_colors = [RED, GREEN, BLUE, YELLOW, PURPLE, ORANGE, TEAL, MAROON]
    projection = "nested"
    show_labels = True
    show_legend = True

    def value_to_color(self, value):
        return interpolate_color(BLUE, RED, value)

    def dim_to_letter(self, dim):
        return chr(65 + dim) if dim < 26 else f"D{dim}"

    def get_vertex_position(self, coords, n_dims):
        if self.projection == "nested":
            return nested_cube(coords, n_dims)
        return nested_cube(coords, n_dims)

    def create_vertex(self, position, value, radius=0.1):
        color = self.value_to_color(value)
        sphere = Sphere(radius=radius, resolution=(15, 15))
        sphere.set_color(color)
        sphere.move_to(position)
        label = DecimalNumber(value, num_decimal_places=2, font_size=14)
        label.next_to(position, OUT)
        label.add_updater(lambda m: m.next_to(position, OUT))
        return VGroup(sphere, label)

    def create_ncube(self, data, include_labels=True):
        n_dims = len(data.shape)
        vertices = []
        edges = set()
        vertex_objs = VGroup()
        edge_objs = VGroup()

        for indices in np.ndindex(data.shape):
            pos = self.get_vertex_position(indices, n_dims)
            vertices.append(pos)
            sphere = Sphere(radius=0.1, resolution=(15, 15))
            sphere.set_color(self.value_to_color(data[indices]))
            sphere.move_to(pos)
            vertex_objs.add(sphere)

            if include_labels:
                label = DecimalNumber(data[indices], num_decimal_places=2, font_size=10)
                label.next_to(pos, OUT)
                label.add_updater(lambda m, p=pos: m.next_to(p, OUT))
                vertex_objs.add(label)

        edge_cache = set()
        for i, indices_i in enumerate(np.ndindex(data.shape)):
            for dim in range(n_dims):
                if indices_i[dim] == 0:
                    indices_j = list(indices_i)
                    indices_j[dim] = 1
                    indices_j = tuple(indices_j)
                    edge_key = tuple(sorted([indices_i, indices_j]))
                    if edge_key not in edge_cache:
                        edge_cache.add(edge_key)
                        line = Line3D(
                            vertices[i],
                            vertices[np.ravel_multi_index(indices_j, data.shape)],
                            color=self.dim_colors[dim % len(self.dim_colors)],
                            thickness=0.02,
                        )
                        edge_objs.add(line)

        return VGroup(edge_objs, vertex_objs)

    def create_color_legend(self):
        gradient = VGroup()
        segments = 10
        for i in range(segments):
            value = i / (segments - 1)
            rect = Square(side_length=0.2, fill_opacity=1)
            rect.set_color(self.value_to_color(value))
            rect.move_to(RIGHT * 3.5 + UP * (1.5 - i * 0.3))
            gradient.add(rect)
        label_min = Text("0.0", font_size=12).next_to(gradient, DOWN, buff=0.1)
        label_max = Text("1.0", font_size=12).next_to(gradient, UP, buff=0.1)
        title = Text("Valor", font_size=14).next_to(gradient, RIGHT, buff=0.2)
        return VGroup(gradient, label_min, label_max, title)
