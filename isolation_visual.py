from manim import *
import numpy as np

class IsolationForestDemo(Scene):
    def construct(self):
        # --- Scene Setup ---
        self.camera.background_color = BLACK

        # --- 1. Data Generation ---
        np.random.seed(42)
        num_normal = 80
        num_outliers = 8
        normal_data = np.random.multivariate_normal(mean=[0, 0], cov=[[0.5, 0.1], [0.1, 0.5]], size=num_normal)
        # More extreme outliers
        outlier_data = np.random.uniform(low=[-4, -4], high=[4, 4], size=(num_outliers, 2))  # Wider spread
        # Ensure outliers are actually outside the normal distribution
        for i in range(num_outliers):
            while np.linalg.norm(outlier_data[i] - [0, 0]) < 2.5:  # Distance check
                outlier_data[i] = np.random.uniform(low=[-4, -4], high=[4, 4], size=(1, 2))

        data = np.concatenate([normal_data, outlier_data])
        labels = np.array([0] * num_normal + [1] * num_outliers)
        shuffle_indices = np.arange(len(data))
        np.random.shuffle(shuffle_indices)
        data = data[shuffle_indices]
        labels = labels[shuffle_indices]

        dots = VGroup(*[Dot(point=[x, y, 0], radius=0.06) for x, y in data])
        self.play(Create(dots), run_time=0.8)  # Faster dot creation
        self.wait(0.2)

        # --- 2. Title and Explanation ---
        title = Text("Isolation Forest", color=WHITE).scale(0.8)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.5)  # Faster title
        self.wait(0.2)

        subtitle = Text("Finding Anomalies in Data", color=GRAY_B).scale(0.5)
        subtitle.next_to(title, DOWN, buff=0.2)
        self.play(Write(subtitle), run_time=0.5) # Faster subtitle
        self.wait(0.5)

        # --- 3. Single Tree Building ---
        explanation_text = Tex("1. Build a tree by randomly splitting.", color=WHITE).scale(0.4)
        explanation_text.to_edge(LEFT, buff=0.5).shift(UP * 1.2)
        self.play(Write(explanation_text), run_time=0.4) # Faster text

        tree, lines = self.build_tree_visual(data, max_depth=3)
        self.wait(0.2)
        self.play(FadeOut(explanation_text), run_time=0.2) # Faster fadeout

        # --- 4. Multiple Trees and Path Length ---
        num_trees = 3
        explanation_text = Tex("2. Repeat for multiple trees.", color=WHITE).scale(0.4)
        explanation_text.to_edge(LEFT, buff=0.5).shift(UP * 1.2)

        self.play(*[FadeOut(line) for line in lines], FadeOut(explanation_text), run_time=0.2) #Faster fadeout
        all_lines = []
        trees = [tree]
        for _ in range(num_trees - 1):
            new_tree, new_lines = self.build_tree_visual(data, max_depth=3)
            trees.append(new_tree)
            all_lines.extend(new_lines)
            self.wait(0.1)  # Even shorter wait

        explanation_text = Tex("3. Shorter average path = Anomaly.", color=WHITE).scale(0.4)
        explanation_text.to_edge(LEFT, buff=0.5).shift(UP * 1.2)

        self.play(Write(explanation_text), run_time=0.4) # Faster text write
        self.wait(0.5)

        self.play(FadeOut(explanation_text), run_time=0.2) #Faster fade out
        self.wait(0.2)


        # --- 5. Anomaly Score and Coloring (ONLY OUTLIERS) ---
        avg_path_lengths = [self.calculate_avg_path_length(trees, point) for point in data]
        max_path_length = max(avg_path_lengths)

        # Create a list to store the animation of outlier dots
        outlier_animations = []

        for i, dot in enumerate(dots):
            if labels[i] == 1:  # Only color outliers
                anomaly_score = avg_path_lengths[i] / max_path_length
                color = interpolate_color(BLUE, RED, anomaly_score)
                # Append the animation to the list, without playing it immediately
                outlier_animations.append(dot.animate.set_color(color))
        # Play all outlier color animations simultaneously
        self.play(*outlier_animations, run_time=0.5)
        self.wait(0.5)


        # --- 6. Highlight True Outliers ---
        self.play(*[FadeOut(line) for line in all_lines], run_time=0.3)  # Faster line fadeout
        self.wait(0.2)

        true_outliers = VGroup(*[dot for i, dot in enumerate(dots) if labels[i] == 1])
        self.play(
            *[dot.animate.scale(1.5) for dot in true_outliers], #Scale and color
            *[dot.animate.set_color(YELLOW) for dot in true_outliers],
            run_time=0.6 #Faster animation
        )
        self.wait(1)



    def build_tree_visual(self, data_points, max_depth, current_depth=0):
        if len(data_points) <= 1 or current_depth == max_depth:
            return None, []

        feature = np.random.choice(2)
        min_val, max_val = np.min(data_points[:, feature]), np.max(data_points[:, feature])
        split_val = np.random.uniform(min_val, max_val)

        left_indices = data_points[:, feature] < split_val
        left_data = data_points[left_indices]
        right_data = data_points[~left_indices]

        if feature == 0:
            split_line = Line(start=[split_val, -4, 0], end=[split_val, 4, 0], color=GRAY_B, stroke_width=2)
        else:
            split_line = Line(start=[-4, split_val, 0], end=[4, split_val, 0], color=GRAY_B, stroke_width=2)

        self.play(Create(split_line), run_time=0.2)  # Even faster line drawing
        lines = [split_line]

        left_tree, left_lines = self.build_tree_visual(left_data, max_depth, current_depth + 1)
        right_tree, right_lines = self.build_tree_visual(right_data, max_depth, current_depth + 1)
        lines.extend(left_lines)
        lines.extend(right_lines)

        return {
            'split_val': split_val,
            'feature': feature,
            'left': left_tree,
            'right': right_tree
        }, lines

    def calculate_avg_path_length(self, trees, point):
        total_path_length = 0
        for tree in trees:
            path_length = 0
            current_node = tree
            while current_node:
                path_length += 1
                if point[current_node['feature']] < current_node['split_val']:
                    current_node = current_node['left']
                else:
                    current_node = current_node['right']
                if current_node is None:
                    break
            total_path_length += path_length
        return total_path_length / len(trees) if trees else 0