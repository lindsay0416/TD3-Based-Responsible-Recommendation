#!/usr/bin/env python3
"""
Calculate belief central distances using pentagon geometry.
Measures PP1 distance - the distance between optimal belief center and user belief center.
"""

import json
import numpy as np
import pickle
import math
from typing import Dict, List, Tuple


class BeliefCentralDistanceCalculator:
    """
    Calculate distances between optimal belief distribution and user belief distributions
    using pentagon geometry representation.
    """
    
    def __init__(self):
        self.optimal_beliefs = None
        self.pentagon_vertices = None
        self.optimal_center = (0, 0)  # P at origin
        
    def load_optimal_beliefs(self, file_path: str = "processed_data/natural_belief_target.pkl"):
        """Load optimal belief distribution."""
        try:
            with open(file_path, 'rb') as f:
                self.optimal_beliefs = pickle.load(f)
            print(f"Loaded optimal beliefs: {self.optimal_beliefs}")
        except FileNotFoundError:
            print(f"Warning: {file_path} not found. Using default values.")
            # Default values based on your example
            self.optimal_beliefs = np.array([0.69958180867747, 0.2, 0.2, 0.2, 0.2])
        
        # Generate pentagon vertices for optimal beliefs
        self.pentagon_vertices = self._generate_pentagon_vertices()
        
    def _generate_pentagon_vertices(self) -> List[Tuple[float, float]]:
        """
        Generate pentagon vertices coordinates.
        Each vertex represents a cluster, distance from center = belief value.
        """
        vertices = []
        n_clusters = len(self.optimal_beliefs)
        
        for i in range(n_clusters):
            # Pentagon vertices at angles: 0°, 72°, 144°, 216°, 288°
            angle = (2 * math.pi * i) / n_clusters - math.pi/2  # Start from top
            radius = self.optimal_beliefs[i]
            
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            vertices.append((x, y))
            
        return vertices
    
    def calculate_user_pentagon_vertices(self, user_beliefs: List[float]) -> List[Tuple[float, float]]:
        """
        Calculate pentagon vertices for a user's belief distribution.
        Same angles as optimal, but different radii based on user beliefs.
        """
        vertices = []
        n_clusters = len(user_beliefs)
        
        for i in range(n_clusters):
            angle = (2 * math.pi * i) / n_clusters - math.pi/2  # Start from top
            radius = user_beliefs[i]
            
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            vertices.append((x, y))
            
        return vertices
    
    def calculate_pentagon_centroid(self, vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Calculate the centroid (geometric center) of pentagon vertices.
        P1 = centroid of (A1, B1, C1, D1, E1)
        """
        if not vertices:
            return (0, 0)
            
        x_sum = sum(vertex[0] for vertex in vertices)
        y_sum = sum(vertex[1] for vertex in vertices)
        n = len(vertices)
        
        return (x_sum / n, y_sum / n)
    
    def calculate_pp1_distance(self, user_beliefs: List[float]) -> float:
        """
        Calculate PP1 distance for a user.
        P = optimal belief center (0, 0)
        P1 = user belief pentagon centroid
        """
        # Calculate user pentagon vertices
        user_vertices = self.calculate_user_pentagon_vertices(user_beliefs)
        
        # Calculate user pentagon centroid (P1)
        p1_x, p1_y = self.calculate_pentagon_centroid(user_vertices)
        
        # Calculate distance from P (0,0) to P1
        pp1_distance = math.sqrt(p1_x**2 + p1_y**2)
        
        return pp1_distance
    
    def calculate_cluster_distances(self, user_beliefs: List[float]) -> List[float]:
        """
        Calculate AA1, BB1, CC1, DD1, EE1 distances.
        Distance between optimal cluster points and user cluster points.
        """
        user_vertices = self.calculate_user_pentagon_vertices(user_beliefs)
        optimal_vertices = self.pentagon_vertices
        
        cluster_distances = []
        for i in range(len(user_beliefs)):
            # Distance between optimal vertex and user vertex for cluster i
            opt_x, opt_y = optimal_vertices[i]
            user_x, user_y = user_vertices[i]
            
            distance = math.sqrt((opt_x - user_x)**2 + (opt_y - user_y)**2)
            cluster_distances.append(distance)
            
        return cluster_distances
    
    def process_all_users(self, input_file: str, output_file: str):
        """
        Process all users and add PP1 distances to the JSON file.
        """
        # Load user beliefs
        with open(input_file, 'r') as f:
            user_data = json.load(f)
        
        print(f"Processing {len(user_data)} users...")
        
        # Calculate PP1 distances for all users
        updated_data = {}
        pp1_distances = []
        
        for user_id, belief_list in user_data.items():
            pp1_distance = self.calculate_pp1_distance(belief_list)
            cluster_distances = self.calculate_cluster_distances(belief_list)
            
            updated_data[user_id] = {
                "beliefs": belief_list,
                "pp1_distance": pp1_distance,
                "cluster_distances": cluster_distances
            }
            
            pp1_distances.append(pp1_distance)
        
        # Save updated data
        with open(output_file, 'w') as f:
            json.dump(updated_data, f, indent=2)
        
        # Print statistics
        pp1_array = np.array(pp1_distances)
        print(f"\nPP1 Distance Statistics:")
        print(f"Mean: {np.mean(pp1_array):.6f}")
        print(f"Std: {np.std(pp1_array):.6f}")
        print(f"Min: {np.min(pp1_array):.6f}")
        print(f"Max: {np.max(pp1_array):.6f}")
        
        print(f"\nUpdated data saved to {output_file}")
        return updated_data


def main():
    """Main function to calculate belief central distances."""
    calculator = BeliefCentralDistanceCalculator()
    
    # Load optimal beliefs
    calculator.load_optimal_beliefs()
    
    # Process all users
    input_file = "processed_data/user_average_beliefs.json"
    output_file = "processed_data/user_average_beliefs.json"  # Update same file
    
    calculator.process_all_users(input_file, output_file)
    
    # Show sample results
    print("\nSample calculations:")
    sample_beliefs = [0.0, 0.0002460, 0.0004718, 0.0, 0.0]
    pp1 = calculator.calculate_pp1_distance(sample_beliefs)
    cluster_dist = calculator.calculate_cluster_distances(sample_beliefs)
    
    print(f"Sample user beliefs: {sample_beliefs}")
    print(f"PP1 distance: {pp1:.6f}")
    print(f"Cluster distances: {[f'{d:.6f}' for d in cluster_dist]}")


if __name__ == "__main__":
    main()