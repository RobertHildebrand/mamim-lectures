"""
Convex Optimization Presentations using Manim Slides
For ISE 5405 - Optimization Theory

This creates interactive presentations you can control like PowerPoint!

Usage:
1. Render:    manim-slides render convex_slides.py GradientDescentSlide
2. Present:   manim-slides present GradientDescentSlide
3. Export:    manim-slides convert GradientDescentSlide slides.html --open

The HTML export works on iPad Safari!
"""

from manim import *
from manim_slides import Slide, ThreeDSlide
import numpy as np


class GradientDescentSlide(Slide):
    """Interactive gradient descent presentation."""
    
    def construct(self):
        # Slide 1: Title
        title = Text("Gradient Descent", font_size=48)
        subtitle = Text("on a Convex Function", font_size=32, color=BLUE)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title), Write(subtitle))
        self.next_slide()  # PAUSE - wait for keypress
        
        # Slide 2: Show axes and function
        self.play(FadeOut(title), FadeOut(subtitle))
        
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            axis_config={"include_tip": True}
        )
        labels = axes.get_axis_labels(x_label="x_1", y_label="x_2")
        
        func_text = MathTex(r"f(x) = x_1^2 + 2x_2^2", font_size=36)
        func_text.to_corner(UR)
        
        self.play(Create(axes), Write(labels), Write(func_text))
        self.next_slide()
        
        # Slide 3: Draw level sets
        level_sets = VGroup()
        colors = [BLUE_E, BLUE_D, BLUE_C, BLUE_B, BLUE_A, GREEN_A]
        levels = [0.5, 1, 2, 4, 6, 8]
        
        for level, color in zip(levels, colors):
            ellipse = ParametricFunction(
                lambda t, c=level: axes.c2p(
                    np.sqrt(c) * np.cos(t),
                    np.sqrt(c/2) * np.sin(t)
                ),
                t_range=[0, 2*PI],
                color=color,
                stroke_width=2
            )
            level_sets.add(ellipse)
        
        level_text = Text("Level sets (contours)", font_size=24)
        level_text.to_corner(UL)
        
        self.play(Create(level_sets), Write(level_text), run_time=2)
        self.next_slide()
        
        # Slide 4: Show minimum
        min_point = Dot(axes.c2p(0, 0), color=YELLOW, radius=0.12)
        min_label = MathTex(r"x^* = (0,0)", font_size=28, color=YELLOW)
        min_label.next_to(min_point, DR, buff=0.2)
        
        self.play(Create(min_point), Write(min_label))
        self.next_slide()
        
        # Slide 5: Starting point
        self.play(FadeOut(level_text))
        
        x0 = np.array([2.5, 1.5])
        start_point = Dot(axes.c2p(*x0), color=RED, radius=0.1)
        start_label = MathTex(r"x^0", font_size=28, color=RED)
        start_label.next_to(start_point, UR, buff=0.1)
        
        algo_text = MathTex(r"x^{k+1} = x^k - \alpha \nabla f(x^k)", font_size=32)
        algo_text.to_corner(UL)
        
        self.play(Create(start_point), Write(start_label), Write(algo_text))
        self.next_slide()
        
        # Slides 6-13: Gradient descent iterations (one per slide)
        step_size = 0.3
        x = x0.copy()
        current_dot = start_point
        arrows = VGroup()
        
        for i in range(8):
            grad = np.array([2*x[0], 4*x[1]])
            x_new = x - step_size * grad
            
            start_pos = axes.c2p(*x)
            end_pos = axes.c2p(*x_new)
            
            arrow = Arrow(start_pos, end_pos, buff=0, color=RED, 
                         stroke_width=3, max_tip_length_to_length_ratio=0.15)
            new_dot = Dot(end_pos, color=RED, radius=0.08)
            
            iter_label = MathTex(f"k = {i+1}", font_size=24)
            iter_label.next_to(algo_text, DOWN, aligned_edge=LEFT)
            
            if i == 0:
                self.play(Create(arrow), Create(new_dot), Write(iter_label))
            else:
                self.play(Create(arrow), Create(new_dot), 
                         Transform(prev_label, iter_label))
            
            arrows.add(arrow)
            prev_label = iter_label if i == 0 else prev_label
            x = x_new
            
            self.next_slide()
        
        # Final slide: Convergence
        converged = Text("Converged!", font_size=36, color=GREEN)
        converged.to_edge(DOWN)
        
        self.play(Write(converged))
        self.next_slide()


class ConvexSetSlide(Slide):
    """Interactive convex set and supporting hyperplane presentation."""
    
    def construct(self):
        # Title slide
        title = Text("Convex Sets", font_size=48)
        subtitle = Text("& Supporting Hyperplanes", font_size=32, color=BLUE)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title), Write(subtitle))
        self.next_slide()
        
        self.play(FadeOut(title), FadeOut(subtitle))
        
        # Axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            x_length=8,
            y_length=6,
        )
        self.play(Create(axes))
        self.next_slide()
        
        # Convex set
        convex_set = Ellipse(
            width=4, height=2.5,
            color=BLUE,
            fill_opacity=0.3
        ).move_to(axes.c2p(0, 0))
        
        set_label = MathTex(r"\mathcal{C}", font_size=36, color=BLUE)
        set_label.move_to(axes.c2p(0, 0))
        
        self.play(Create(convex_set), Write(set_label))
        self.next_slide()
        
        # Show convexity with line segment
        definition = MathTex(
            r"\text{Convex: } \lambda x + (1-\lambda)y \in \mathcal{C}",
            font_size=28
        ).to_edge(UP)
        
        p1 = axes.c2p(-1.5, 0.5)
        p2 = axes.c2p(1.2, -0.8)
        
        dot1 = Dot(p1, color=YELLOW)
        dot2 = Dot(p2, color=YELLOW)
        line_segment = Line(p1, p2, color=YELLOW, stroke_width=3)
        
        x_label = MathTex("x", font_size=24, color=YELLOW).next_to(dot1, UL, buff=0.1)
        y_label = MathTex("y", font_size=24, color=YELLOW).next_to(dot2, DR, buff=0.1)
        
        self.play(Write(definition))
        self.play(Create(dot1), Create(dot2), Write(x_label), Write(y_label))
        self.next_slide()
        
        self.play(Create(line_segment))
        
        inside_text = Text("Line segment stays inside!", font_size=24, color=GREEN)
        inside_text.to_edge(DOWN)
        self.play(Write(inside_text))
        self.next_slide()
        
        # Remove line segment demo
        self.play(FadeOut(dot1), FadeOut(dot2), FadeOut(line_segment),
                 FadeOut(x_label), FadeOut(y_label), FadeOut(inside_text),
                 FadeOut(definition))
        self.next_slide()
        
        # Supporting hyperplane
        hyp_title = Text("Supporting Hyperplane Theorem", font_size=28)
        hyp_title.to_edge(UP)
        self.play(Write(hyp_title))
        
        boundary_point = axes.c2p(2, 0)
        boundary_dot = Dot(boundary_point, color=RED, radius=0.12)
        bar_x = MathTex(r"\bar{x}", font_size=28, color=RED)
        bar_x.next_to(boundary_dot, UR, buff=0.1)
        
        self.play(Create(boundary_dot), Write(bar_x))
        self.next_slide()
        
        # Supporting hyperplane line
        hyperplane = DashedLine(
            axes.c2p(2, -2.5),
            axes.c2p(2, 2.5),
            color=RED,
            stroke_width=2
        )
        
        self.play(Create(hyperplane))
        self.next_slide()
        
        # Normal vector
        normal = Arrow(
            boundary_point,
            axes.c2p(3.2, 0),
            color=GREEN,
            buff=0
        )
        normal_label = MathTex(r"a", font_size=28, color=GREEN)
        normal_label.next_to(normal, UP)
        
        self.play(Create(normal), Write(normal_label))
        self.next_slide()
        
        # Theorem statement
        theorem = MathTex(
            r"a^\top x \leq a^\top \bar{x} \quad \forall x \in \mathcal{C}",
            font_size=32
        ).to_edge(DOWN)
        
        box = SurroundingRectangle(theorem, color=YELLOW, buff=0.2)
        
        self.play(Write(theorem), Create(box))
        self.next_slide()


class LPSlide(Slide):
    """Interactive linear programming visualization."""
    
    def construct(self):
        # Title
        title = Text("Linear Programming", font_size=48)
        subtitle = Text("Geometric Interpretation", font_size=32, color=BLUE)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title), Write(subtitle))
        self.next_slide()
        
        self.play(FadeOut(title), FadeOut(subtitle))
        
        # Problem statement
        problem = MathTex(
            r"\max\; & 3x_1 + 2x_2 \\",
            r"\text{s.t.}\; & x_1 + x_2 \leq 4 \\",
            r"& 2x_1 + x_2 \leq 5 \\",
            r"& x_1, x_2 \geq 0",
            font_size=28
        ).to_corner(UR)
        
        self.play(Write(problem))
        self.next_slide()
        
        # Axes
        axes = Axes(
            x_range=[-0.5, 5, 1],
            y_range=[-0.5, 5, 1],
            x_length=7,
            y_length=5.5,
            axis_config={"include_tip": True}
        ).shift(DOWN * 0.3 + LEFT * 1)
        
        labels = axes.get_axis_labels(x_label="x_1", y_label="x_2")
        self.play(Create(axes), Write(labels))
        self.next_slide()
        
        # Feasible region
        vertices = [
            axes.c2p(0, 0),
            axes.c2p(2.5, 0),
            axes.c2p(1, 3),
            axes.c2p(0, 4),
        ]
        
        feasible_region = Polygon(
            *vertices,
            color=BLUE,
            fill_opacity=0.3,
            stroke_width=2
        )
        
        region_label = Text("Feasible Region", font_size=24, color=BLUE)
        region_label.next_to(feasible_region, DOWN)
        
        self.play(Create(feasible_region), Write(region_label))
        self.next_slide()
        
        # Show vertices
        vertex_dots = VGroup()
        for v in vertices:
            vertex_dots.add(Dot(v, color=WHITE, radius=0.08))
        
        vertex_text = Text("Vertices (corners)", font_size=20)
        vertex_text.to_corner(UL)
        
        self.play(Create(vertex_dots), Write(vertex_text))
        self.next_slide()
        
        self.play(FadeOut(region_label), FadeOut(vertex_text))
        
        # Sweep objective level sets
        obj_label = MathTex(r"3x_1 + 2x_2 = z", font_size=28, color=YELLOW)
        obj_label.to_corner(UL)
        self.play(Write(obj_label))
        
        level_values = [2, 4, 6, 8, 9]
        
        current_line = None
        current_z = None
        
        for z_val in level_values:
            # x2 = (z - 3*x1) / 2
            new_line = axes.plot(
                lambda x, z=z_val: (z - 3*x) / 2,
                x_range=[-0.3, min(4, (z_val+1)/3)],
                color=YELLOW,
                stroke_width=3
            )
            
            z_label = MathTex(f"z = {z_val}", font_size=28, color=YELLOW)
            z_label.next_to(obj_label, DOWN, aligned_edge=LEFT)
            
            if current_line is None:
                self.play(Create(new_line), Write(z_label))
                current_line = new_line
                current_z = z_label
            else:
                self.play(Transform(current_line, new_line),
                         Transform(current_z, z_label))
            
            self.next_slide()
        
        # Highlight optimal vertex
        opt_point = axes.c2p(1, 3)
        opt_dot = Dot(opt_point, color=RED, radius=0.15)
        opt_label = MathTex(r"x^* = (1, 3)", font_size=28, color=RED)
        opt_label.next_to(opt_dot, UP)
        
        self.play(Create(opt_dot), Write(opt_label))
        self.next_slide()
        
        # Optimal value
        opt_value = MathTex(r"z^* = 3(1) + 2(3) = 9", font_size=32, color=GREEN)
        opt_value.to_edge(DOWN)
        
        box = SurroundingRectangle(opt_value, color=GREEN, buff=0.2)
        
        self.play(Write(opt_value), Create(box))
        self.next_slide()
        
        # Key insight
        insight = Text("Optimal solution is always at a vertex!", 
                      font_size=28, color=YELLOW)
        insight.next_to(opt_value, UP)
        
        self.play(Write(insight))
        self.next_slide()


class SubgradientSlide(Slide):
    """Interactive subgradient visualization."""
    
    def construct(self):
        # Title
        title = Text("Subgradients", font_size=48)
        subtitle = Text("for Non-Smooth Functions", font_size=32, color=BLUE)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title), Write(subtitle))
        self.next_slide()
        
        self.play(FadeOut(title), FadeOut(subtitle))
        
        # Axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-0.5, 3, 1],
            x_length=8,
            y_length=5,
            axis_config={"include_tip": True}
        )
        labels = axes.get_axis_labels(x_label="x", y_label="f(x)")
        
        self.play(Create(axes), Write(labels))
        self.next_slide()
        
        # |x| function
        left_part = axes.plot(lambda x: -x, x_range=[-2.5, 0], color=BLUE, stroke_width=3)
        right_part = axes.plot(lambda x: x, x_range=[0, 2.5], color=BLUE, stroke_width=3)
        
        func_label = MathTex(r"f(x) = |x|", font_size=32, color=BLUE)
        func_label.to_corner(UR)
        
        self.play(Create(left_part), Create(right_part), Write(func_label))
        self.next_slide()
        
        # Point at x=0
        origin = axes.c2p(0, 0)
        origin_dot = Dot(origin, color=RED, radius=0.12)
        
        kink_label = Text("Non-differentiable!", font_size=24, color=RED)
        kink_label.next_to(origin_dot, DOWN, buff=0.3)
        
        self.play(Create(origin_dot), Write(kink_label))
        self.next_slide()
        
        # Definition
        self.play(FadeOut(kink_label))
        
        definition = MathTex(
            r"g \in \partial f(\bar{x}) \Leftrightarrow f(x) \geq f(\bar{x}) + g(x - \bar{x})",
            font_size=24
        ).to_edge(DOWN)
        
        self.play(Write(definition))
        self.next_slide()
        
        # Show subgradients one by one
        subgrad_values = [-1, -0.5, 0, 0.5, 1]
        
        info_text = MathTex(r"\partial f(0) = [-1, 1]", font_size=28)
        info_text.to_corner(UL)
        self.play(Write(info_text))
        
        current_line = None
        current_label = None
        
        for g in subgrad_values:
            new_line = axes.plot(
                lambda x, slope=g: slope * x,
                x_range=[-2.5, 2.5],
                color=GREEN,
                stroke_width=2
            )
            
            g_label = MathTex(f"g = {g}", font_size=28, color=GREEN)
            g_label.next_to(info_text, DOWN, aligned_edge=LEFT)
            
            tangent_text = Text("Supporting line (tangent)", font_size=20, color=GREEN)
            tangent_text.next_to(g_label, DOWN, aligned_edge=LEFT)
            
            if current_line is None:
                self.play(Create(new_line), Write(g_label), Write(tangent_text))
                current_line = new_line
                current_label = g_label
                current_tangent = tangent_text
            else:
                self.play(Transform(current_line, new_line),
                         Transform(current_label, g_label))
            
            self.next_slide()
        
        # Optimality condition
        self.play(FadeOut(current_tangent))
        
        opt_condition = MathTex(
            r"0 \in \partial f(x^*) \Rightarrow x^* \text{ is optimal}",
            font_size=28, color=YELLOW
        )
        opt_condition.next_to(definition, UP)
        
        self.play(Write(opt_condition))
        self.next_slide()


class EpigraphSlide(ThreeDSlide):
    """Interactive 3D epigraph visualization."""
    
    def construct(self):
        # Set up camera
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        # Title (fixed in frame)
        title = Text("The Epigraph", font_size=36)
        title.to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        self.next_slide()
        
        # 3D axes
        axes = ThreeDAxes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            z_range=[0, 4, 1],
            x_length=4,
            y_length=4,
            z_length=4,
        )
        
        self.play(Create(axes))
        self.next_slide()
        
        # Surface
        surface = Surface(
            lambda u, v: axes.c2p(u, v, u**2 + v**2),
            u_range=[-1.8, 1.8],
            v_range=[-1.8, 1.8],
            resolution=(30, 30),
            fill_opacity=0.7,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )
        
        func_label = MathTex(r"f(x) = x_1^2 + x_2^2", font_size=28)
        func_label.to_corner(UR)
        self.add_fixed_in_frame_mobjects(func_label)
        
        self.play(Create(surface), Write(func_label), run_time=2)
        self.next_slide()
        
        # Rotate to show shape
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(2)
        self.stop_ambient_camera_rotation()
        self.next_slide()
        
        # Epigraph definition
        epi_def = MathTex(
            r"\text{epi}(f) = \{(x, t) : f(x) \leq t\}",
            font_size=28
        )
        epi_def.next_to(func_label, DOWN, aligned_edge=RIGHT)
        self.add_fixed_in_frame_mobjects(epi_def)
        self.play(Write(epi_def))
        self.next_slide()
        
        # Key result
        key_result = MathTex(
            r"f \text{ convex} \Leftrightarrow \text{epi}(f) \text{ convex}",
            font_size=28, color=YELLOW
        )
        key_result.next_to(epi_def, DOWN, aligned_edge=RIGHT)
        self.add_fixed_in_frame_mobjects(key_result)
        self.play(Write(key_result))
        self.next_slide()
        
        # Final rotation
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        self.next_slide()


# Usage instructions at the bottom
"""
USAGE:

1. Render a presentation:
   manim-slides render convex_slides.py GradientDescentSlide

2. Present it (opens GUI):
   manim-slides present GradientDescentSlide

3. Export to HTML (works on iPad!):
   manim-slides convert GradientDescentSlide gradient_descent.html --open

4. Render and present all slides:
   manim-slides render convex_slides.py GradientDescentSlide ConvexSetSlide LPSlide SubgradientSlide
   manim-slides present GradientDescentSlide ConvexSetSlide LPSlide SubgradientSlide

CONTROLS:
- Space / Right Arrow: Next slide
- Left Arrow: Previous slide  
- R: Replay current animation
- Q: Quit

For iPad: Export to HTML and open in Safari!
"""
