from manim import *
import numpy as np

from visualization.cube_base import NCubeVisualizer
from visualization import data


class NCubeBasic(NCubeVisualizer):
    """Static N-dimensional hypercube."""
    n_dims = 4

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)
        title = Text(f"NCube {self.n_dims}D", font_size=36)
        title.to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)

        cube_data = data.generate(self.n_dims)
        ncube = self.create_ncube(cube_data)
        self.play(Create(ncube))
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(3)


class ReductionSequence(NCubeVisualizer):
    """Dimensional reduction 4D -> 3D -> 2D -> 1D -> 0D via np.mean."""
    n_dims = 4
    dim_names = [
        "Teseracto (4D)", "Cubo (3D)", "Cuadrado (2D)",
        "L\u00ednea (1D)", "Punto (0D)",
    ]

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=30 * DEGREES)
        title = Text("Reducci\u00f3n Dimensional con np.mean", font_size=36)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        data_4d = data.generate(self.n_dims)
        reduction_data = [data_4d]
        for axis in range(self.n_dims - 1, -1, -1):
            reduction_data.append(np.mean(reduction_data[-1], axis=axis))

        current_structure = self.create_ncube(data_4d)
        subtitle = Text(self.dim_names[0], font_size=28)
        subtitle.next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(subtitle)
        self.play(Create(current_structure), Write(subtitle))
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(2)
        self.stop_ambient_camera_rotation()

        for i in range(1, len(self.dim_names)):
            new_subtitle = Text(self.dim_names[i], font_size=28)
            new_subtitle.next_to(title, DOWN)
            old_subtitle = subtitle
            subtitle = new_subtitle

            formula = MathTex(
                r"\text{np.mean(data, axis=", f"{self.n_dims - i}", r")}"
            )
            formula.scale(0.8)
            formula.to_edge(DOWN)
            self.add_fixed_in_frame_mobjects(formula)
            self.play(Write(formula))

            new_structure = self.create_ncube(
                reduction_data[i], include_labels=(i < len(self.dim_names) - 1)
            )
            self.play(
                ReplacementTransform(current_structure, new_structure),
                Transform(old_subtitle, subtitle),
            )
            self.wait(1)
            current_structure = new_structure
            self.remove(formula)

        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(3)
