import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self._tree = {}
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self._num_classes = None

    def _get_gini(self, y):
        if len(y) == 0:
            return 0.0
        classes, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return 1.0 - np.sum(proportions**2)

    def _find_best_split(self, sub_X, sub_y, feature_idx):
        feature_type = self.feature_types[feature_idx]
        current_feature_values = sub_X[:, feature_idx]
        
        best_gini = 1.0
        best_split_value = None

        if feature_type == "real":
            unique_values = np.unique(current_feature_values)
            if len(unique_values) < 2:
                return best_gini, best_split_value

            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i+1]) / 2
                mask = current_feature_values <= threshold
                y_left = sub_y[mask]
                y_right = sub_y[~mask]

                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                current_gini = (len(y_left) / len(sub_y)) * self._get_gini(y_left) + \
                               (len(y_right) / len(sub_y)) * self._get_gini(y_right)

                if current_gini < best_gini:
                    best_gini = current_gini
                    best_split_value = threshold

        elif feature_type == "categorical":
            counts = Counter(current_feature_values)
            if len(counts) < 2:
                return best_gini, best_split_value

            category_ratios = {}
            for category in counts:
                mask = current_feature_values == category
                y_category = sub_y[mask]
                if len(y_category) > 0:
                    category_ratios[category] = np.mean(y_category)
                else:
                    category_ratios[category] = 0

            sorted_categories = sorted(category_ratios.items(), key=lambda x: x[1])

            for i in range(len(sorted_categories) - 1):
                left_categories = [cat for cat, _ in sorted_categories[:i+1]]
                
                mask = np.isin(current_feature_values, left_categories)
                y_left = sub_y[mask]
                y_right = sub_y[~mask]

                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                current_gini = (len(y_left) / len(sub_y)) * self._get_gini(y_left) + \
                               (len(y_right) / len(sub_y)) * self._get_gini(y_right)

                if current_gini < best_gini:
                    best_gini = current_gini
                    best_split_value = left_categories

        return best_gini, best_split_value

    def _fit_node(self, sub_X, sub_y, node, current_depth=0):
        if len(sub_y) < self.min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self.max_depth is not None and current_depth >= self.max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        best_gini = 1.0
        best_feature_idx = None
        best_split_value = None

        for feature_idx in range(sub_X.shape[1]):
            current_gini, current_split_value = self._find_best_split(sub_X, sub_y, feature_idx)
            
            if current_split_value is not None and current_gini < best_gini:
                best_gini = current_gini
                best_feature_idx = feature_idx
                best_split_value = current_split_value

        if best_feature_idx is None or best_split_value is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "internal"
        node["feature_idx"] = best_feature_idx
        node["feature_type"] = self.feature_types[best_feature_idx]
        node["split_value"] = best_split_value

        left_node = {}
        right_node = {}
        node["left"] = left_node
        node["right"] = right_node

        if self.feature_types[best_feature_idx] == "real":
            mask = sub_X[:, best_feature_idx] <= best_split_value
        else:
            mask = np.isin(sub_X[:, best_feature_idx], best_split_value)

        X_left, y_left = sub_X[mask], sub_y[mask]
        X_right, y_right = sub_X[~mask], sub_y[~mask]

        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        self._fit_node(X_left, y_left, left_node, current_depth + 1)
        self._fit_node(X_right, y_right, right_node, current_depth + 1)

    def fit(self, X, y):
        self._num_classes = len(np.unique(y))
        self._fit_node(X, y, self._tree)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_idx"]
        feature_type = node["feature_type"]
        split_value = node["split_value"]

        if feature_type == "real":
            if x[feature_idx] <= split_value:
                return self._predict_node(x, node["left"])
            else:
                return self._predict_node(x, node["right"])
        elif feature_type == "categorical":
            if x[feature_idx] in split_value:
                return self._predict_node(x, node["left"])
            else:
                return self._predict_node(x, node["right"])

    def predict(self, X):
        predictions = np.array([self._predict_node(x, self._tree) for x in X])
        return predictions

    def get_params(self, deep=True):
        return {
            'feature_types': self.feature_types,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self