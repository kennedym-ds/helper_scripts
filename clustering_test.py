import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def topology_aware_distance(point1, point2, radius):
  """
  Calculates the distance between two points in a circular plane,
  considering the boundary.

  Args:
    point1: Coordinates of the first point.
    point2: Coordinates of the second point.
    radius: The radius of the circular plane.

  Returns:
    The topology-aware distance between the two points.
  """
  # Calculate Euclidean distance.
  euclidean_dist = np.linalg.norm(point1 - point2)

  # Calculate angular distance.
  angle1 = np.arctan2(point1[1], point1[0])
  angle2 = np.arctan2(point2[1], point2[0])
  angular_dist = radius * np.abs(angle1 - angle2)

  # Return the minimum of Euclidean and angular distance.
  return min(euclidean_dist, angular_dist)


def preprocess_data(data, radius):
  """
  Preprocesses the data to mitigate edge effects using mirroring.

  Args:
    data: The input data points.
    radius: The radius of the circular plane.

  Returns:
    The preprocessed data with mirrored points.
  """
  # Mirror points across the x and y axes.
  mirrored_data = np.concatenate([
      data,
      data * np.array([-1, 1]),
      data * np.array([1, -1]),
      data * np.array([-1, -1]),
  ])

  # Mirror points across the circular boundary
  # For each point, calculate its reflection across the circle's circumference
  mirrored_circular_data = []
  for point in data:
    # Calculate distance from origin
    distance = np.linalg.norm(point)
    
    # Only mirror points that are inside the circle
    if distance < radius:
      # Calculate the angle
      angle = np.arctan2(point[1], point[0])
      
      # Calculate the mirrored point across the circle
      # The mirrored point is at distance (2*radius - distance) from origin
      mirrored_distance = 2 * radius - distance
      mirrored_x = mirrored_distance * np.cos(angle)
      mirrored_y = mirrored_distance * np.sin(angle)
      mirrored_circular_data.append([mirrored_x, mirrored_y])
  
  # Add circular mirrored points to the dataset
  if mirrored_circular_data:
    mirrored_circular_array = np.array(mirrored_circular_data)
    mirrored_data = np.concatenate([mirrored_data, mirrored_circular_array])

  return mirrored_data


def cluster_data(data, distance_metric, linkage="ward"):
  """
  Performs hierarchical clustering on the data.

  Args:
    data: The input data points.
    distance_metric: The distance metric to use.
    linkage: The linkage criterion for hierarchical clustering.

  Returns:
    The cluster labels for each data point.
  """
  clustering = AgglomerativeClustering(
      n_clusters=None,  # Determine number of clusters automatically
      affinity="precomputed",  # Use a precomputed distance matrix
      linkage=linkage,
      distance_threshold=0.5  # Adjust as needed
  )

  # Calculate the distance matrix using the provided metric
  distance_matrix = np.array([[distance_metric(p1, p2) for p2 in data] for p1 in data])

  labels = clustering.fit_predict(distance_matrix)
  return labels


def identify_outliers(data, contamination=0.1):
  """
  Identifies outliers in the data using LOF.

  Args:
    data: The input data points.
    contamination: The proportion of outliers in the data.

  Returns:
    A boolean array indicating whether each point is an outlier.
  """
  lof = LocalOutlierFactor(contamination=contamination)
  outlier_scores = lof.fit_predict(data)
  is_outlier = outlier_scores == -1
  return is_outlier


def visualize_clusters(data, labels, is_outlier):
  """
  Visualizes the clustering results.

  Args:
    data: The input data points.
    labels: The cluster labels for each data point.
    is_outlier: A boolean array indicating outliers.
  """
  plt.figure(figsize=(8, 8))

  # Plot the clusters
  unique_labels = set(labels)
  for label in unique_labels:
    if label == -1:  # Noise points
      plt.scatter(data[labels == label, 0], data[labels == label, 1], 
                  color="black", marker="x", label="Noise")
    else:
      plt.scatter(data[labels == label, 0], data[labels == label, 1], 
                  label=f"Cluster {label}")

  # Highlight outliers
  plt.scatter(data[is_outlier, 0], data[is_outlier, 1], 
              facecolors="none", edgecolors="red", s=100, linewidths=2, label="Outliers")

  plt.xlabel("X")
  plt.ylabel("Y")
  plt.title("Clustering Results")
  plt.legend()
  plt.show()


def analyze_adversarial_data_points(data, labels, is_outlier):
  """
  Analyzes adversarial data points that could be potential security threats.
  
  Args:
    data: The input data points.
    labels: The cluster labels for each data point.
    is_outlier: A boolean array indicating outliers.
  
  Returns:
    A dictionary containing adversarial analysis results.
  """
  print("\n--- Adversarial Data Points Analysis ---")
  
  # Identify potential adversarial points
  adversarial_indicators = {
    'isolated_outliers': [],
    'boundary_points': [],
    'suspicious_clusters': [],
    'anomalous_patterns': []
  }
  
  # 1. Isolated outliers (points marked as outliers and in noise clusters)
  noise_outliers = np.where((labels == -1) & is_outlier)[0]
  adversarial_indicators['isolated_outliers'] = noise_outliers.tolist()
  
  # 2. Boundary points - points that are close to multiple cluster boundaries
  unique_labels = set(labels)
  unique_labels.discard(-1)  # Remove noise label
  
  boundary_threshold = 0.1  # Distance threshold for boundary detection
  for point_idx in range(len(data)):
    if labels[point_idx] != -1:  # Skip noise points
      point = data[point_idx]
      distances_to_clusters = []
      
      for cluster_label in unique_labels:
        if cluster_label != labels[point_idx]:
          cluster_points = data[labels == cluster_label]
          if len(cluster_points) > 0:
            min_distance = np.min([np.linalg.norm(point - cp) for cp in cluster_points])
            distances_to_clusters.append(min_distance)
      
      # If point is close to multiple other clusters, it's on a boundary
      if len(distances_to_clusters) > 0 and np.min(distances_to_clusters) < boundary_threshold:
        adversarial_indicators['boundary_points'].append(point_idx)
  
  # 3. Suspicious clusters - very small clusters that might be adversarial
  cluster_counts = {}
  for label in labels:
    if label != -1:
      cluster_counts[label] = cluster_counts.get(label, 0) + 1
  
  min_cluster_size = max(3, len(data) // 20)  # Minimum expected cluster size
  for cluster_label, count in cluster_counts.items():
    if count < min_cluster_size:
      adversarial_indicators['suspicious_clusters'].append(cluster_label)
  
  # 4. Anomalous patterns - points with unusual local density
  local_density_threshold = np.percentile([np.sum(np.linalg.norm(data - point, axis=1) < 0.5) 
                                          for point in data], 10)
  
  for point_idx, point in enumerate(data):
    local_density = np.sum(np.linalg.norm(data - point, axis=1) < 0.5)
    if local_density < local_density_threshold:
      adversarial_indicators['anomalous_patterns'].append(point_idx)
  
  # Print analysis results
  print(f"Isolated outliers found: {len(adversarial_indicators['isolated_outliers'])}")
  print(f"Boundary points found: {len(adversarial_indicators['boundary_points'])}")
  print(f"Suspicious clusters found: {len(adversarial_indicators['suspicious_clusters'])}")
  print(f"Anomalous patterns found: {len(adversarial_indicators['anomalous_patterns'])}")
  
  if adversarial_indicators['isolated_outliers']:
    print(f"Isolated outlier indices: {adversarial_indicators['isolated_outliers'][:10]}...")
  
  if adversarial_indicators['suspicious_clusters']:
    print(f"Suspicious cluster labels: {adversarial_indicators['suspicious_clusters']}")
  
  return adversarial_indicators


def conduct_risk_assessment(data, labels, is_outlier, adversarial_indicators):
  """
  Conducts a comprehensive risk assessment based on clustering and adversarial analysis.
  
  Args:
    data: The input data points.
    labels: The cluster labels for each data point.
    is_outlier: A boolean array indicating outliers.
    adversarial_indicators: Dictionary from adversarial analysis.
  
  Returns:
    A risk assessment report.
  """
  print("\n--- Risk Assessment ---")
  
  total_points = len(data)
  outlier_count = np.sum(is_outlier)
  noise_points = np.sum(labels == -1)
  
  # Calculate risk factors
  risk_factors = {
    'outlier_ratio': outlier_count / total_points,
    'noise_ratio': noise_points / total_points,
    'adversarial_ratio': len(set(
      adversarial_indicators['isolated_outliers'] + 
      adversarial_indicators['boundary_points'] + 
      adversarial_indicators['anomalous_patterns']
    )) / total_points,
    'suspicious_cluster_ratio': len(adversarial_indicators['suspicious_clusters']) / max(1, len(set(labels)) - 1)
  }
  
  # Assign risk levels
  risk_assessment = {
    'overall_risk': 'LOW',
    'specific_risks': [],
    'recommendations': []
  }
  
  # Risk level determination
  high_risk_threshold = 0.15
  medium_risk_threshold = 0.05
  
  if risk_factors['outlier_ratio'] > high_risk_threshold:
    risk_assessment['specific_risks'].append(f"HIGH: Outlier ratio ({risk_factors['outlier_ratio']:.2%}) exceeds threshold")
    risk_assessment['overall_risk'] = 'HIGH'
  elif risk_factors['outlier_ratio'] > medium_risk_threshold:
    risk_assessment['specific_risks'].append(f"MEDIUM: Elevated outlier ratio ({risk_factors['outlier_ratio']:.2%})")
    if risk_assessment['overall_risk'] == 'LOW':
      risk_assessment['overall_risk'] = 'MEDIUM'
  
  if risk_factors['adversarial_ratio'] > high_risk_threshold:
    risk_assessment['specific_risks'].append(f"HIGH: Adversarial pattern ratio ({risk_factors['adversarial_ratio']:.2%}) exceeds threshold")
    risk_assessment['overall_risk'] = 'HIGH'
  elif risk_factors['adversarial_ratio'] > medium_risk_threshold:
    risk_assessment['specific_risks'].append(f"MEDIUM: Elevated adversarial patterns ({risk_factors['adversarial_ratio']:.2%})")
    if risk_assessment['overall_risk'] == 'LOW':
      risk_assessment['overall_risk'] = 'MEDIUM'
  
  if risk_factors['suspicious_cluster_ratio'] > 0.3:
    risk_assessment['specific_risks'].append(f"MEDIUM: High suspicious cluster ratio ({risk_factors['suspicious_cluster_ratio']:.2%})")
    if risk_assessment['overall_risk'] == 'LOW':
      risk_assessment['overall_risk'] = 'MEDIUM'
  
  # Generate recommendations
  if risk_assessment['overall_risk'] == 'HIGH':
    risk_assessment['recommendations'].extend([
      "Implement additional security monitoring",
      "Investigate isolated outliers manually",
      "Consider data validation and sanitization",
      "Apply more restrictive clustering parameters"
    ])
  elif risk_assessment['overall_risk'] == 'MEDIUM':
    risk_assessment['recommendations'].extend([
      "Monitor boundary points for unusual activity",
      "Review small clusters for legitimacy",
      "Consider increasing outlier detection sensitivity"
    ])
  else:
    risk_assessment['recommendations'].append("Current clustering patterns appear normal")
  
  # Print assessment
  print(f"Overall Risk Level: {risk_assessment['overall_risk']}")
  print(f"Outlier Ratio: {risk_factors['outlier_ratio']:.2%}")
  print(f"Noise Ratio: {risk_factors['noise_ratio']:.2%}")
  print(f"Adversarial Pattern Ratio: {risk_factors['adversarial_ratio']:.2%}")
  print(f"Suspicious Cluster Ratio: {risk_factors['suspicious_cluster_ratio']:.2%}")
  
  if risk_assessment['specific_risks']:
    print("\nSpecific Risk Factors:")
    for risk in risk_assessment['specific_risks']:
      print(f"  - {risk}")
  
  print("\nRecommendations:")
  for rec in risk_assessment['recommendations']:
    print(f"  - {rec}")
  
  return risk_assessment


def main():
  # Load the data
  data = np.loadtxt("data.csv", delimiter=",") 

  # Define the radius of the circular plane
  radius = 10

  # Preprocess the data
  preprocessed_data = preprocess_data(data, radius)

  # Define the distance metric
  def distance_metric(point1, point2):
    return topology_aware_distance(point1, point2, radius)

  # Perform clustering
  labels = cluster_data(preprocessed_data, distance_metric)

  # Identify outliers
  is_outlier = identify_outliers(preprocessed_data)

  # Evaluate clustering (example using silhouette score)
  score = silhouette_score(preprocessed_data, labels, metric="precomputed")
  print("Silhouette score:", score)

  # Analyze adversarial data points
  adversarial_indicators = analyze_adversarial_data_points(preprocessed_data, labels, is_outlier)

  # Visualize clustering results
  visualize_clusters(preprocessed_data, labels, is_outlier)

  # Conduct risk assessment
  risk_assessment = conduct_risk_assessment(preprocessed_data, labels, is_outlier, adversarial_indicators)

if __name__ == "__main__":
  main()