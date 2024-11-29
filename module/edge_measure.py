import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, multivariate_normal
from scipy.spatial import distance_matrix


class PointDistributionAnalyzer:
    def __init__(self, points, predefined_distribution=None, num_radial_bins=10,
                 num_angular_bins=10, parzen_bandwidth = 0.5):
        self.points = points
        self.distances = np.abs(points)  # Compute magnitudes of complex points
        self.angles = np.angle(points)  # Compute angles of complex points
        self.num_radial_bins = num_radial_bins
        self.num_angular_bins = num_angular_bins
        self.predefined_distribution = predefined_distribution
        self.parzen_bandwidth = parzen_bandwidth

    def compute_entropy(self):
        # Discretize the unit disc into a 2D histogram
        hist, xedges, yedges = np.histogram2d(self.points.real, self.points.imag, bins=50, range=[[-1, 1], [-1, 1]])

        # Normalize to form a probability density function (PDF)
        pdf = hist / np.sum(hist)

        # Compute the entropy of the distribution
        # Flatten the PDF to 1D and filter out zero probabilities to avoid log(0)
        pdf_nonzero = pdf[pdf > 0]
        entropy_value = entropy(pdf_nonzero)

        return entropy_value

    def fast_parzen_window_pt(self, method='hard', k=5, return_density=True,
                              use_adaptive_bandwidth=False, use_histogram = True, num_bins=50):
        """
        Estimate the entropy of the density using the Fast Parzen window method with random ball selection,
        supporting both hard and soft versions, with data-adaptive Gaussian kernels.

        Parameters:
        - data_complex: array-like, shape (n_samples,), complex data points in the complex plane.
        - base_bandwidth: float, the base bandwidth (scaling factor for the adaptive kernels).
        - method: str, either 'hard' or 'soft' to choose the version.
        - k: int, number of nearest neighbors to use for adaptive kernel bandwidth.
        - return_density: bool, if True, return the estimated density of the mixture distribution.

        Returns:
        - entropy: float, the entropy of the estimated density.
        - density_estimates (optional): array, the estimated density of the mixture distribution.
        """
        # Separate real and imaginary parts of complex data
        data_real = self.points.real
        data_imag = self.points.imag
        base_bandwidth = self.parzen_bandwidth

        # Combine real and imaginary parts into a 2D array
        data = np.column_stack((data_real, data_imag))
        n_samples = data.shape[0]

        # Determine bandwidths
        if use_adaptive_bandwidth:
            # Calculate distance matrix and determine adaptive bandwidths
            distances = distance_matrix(data, data)
            adaptive_bandwidths = base_bandwidth * np.sort(distances, axis=1)[:,
                                                   k]  # k-th nearest neighbor distances as bandwidth
        else:
            # Fixed bandwidth for all points
            adaptive_bandwidths = np.full(n_samples, base_bandwidth)

        # Initialize density estimates
        density_estimates = np.zeros(n_samples)

        if method == 'hard':
            # Hard version: Discard points once covered
            uncovered_indices = set(range(n_samples))  # Use set for efficient removal

            # Continue until all points are covered
            while uncovered_indices:
                # Randomly select a point from uncovered points
                center_idx = np.random.choice(list(uncovered_indices))
                center = data[center_idx]
                center_bandwidth = adaptive_bandwidths[center_idx]  # Bandwidth for the selected center

                # Calculate distances to the selected center from all points
                if use_adaptive_bandwidth:
                    distances_to_center = distances[center_idx]
                else:
                    distances_to_center = np.linalg.norm(data - center, axis=1)

                in_bandwidth = np.where(distances_to_center < center_bandwidth)[0]

                # Estimate density contributions for points within the bandwidth
                density_contributions = multivariate_normal.pdf(
                    data[in_bandwidth], mean=center, cov=center_bandwidth ** 2 * np.eye(2)
                )

                # Add density contributions to uncovered points
                density_estimates[in_bandwidth] += density_contributions / n_samples

                # Remove covered points from the uncovered set
                uncovered_indices.difference_update(in_bandwidth)

        elif method == 'soft':
            # Soft version: Points can contribute to multiple balls with different weights
            num_centers = 0

            # Continue until all points have been considered a sufficient number of times
            while num_centers < n_samples:
                # Randomly select a center
                center_idx = np.random.choice(n_samples)
                center = data[center_idx]
                center_bandwidth = adaptive_bandwidths[center_idx]  # Bandwidth for the selected center

                # Calculate distances from the center to all points
                distances_to_center = np.linalg.norm(data - center, axis=1) if not use_adaptive_bandwidth else \
                distances[center_idx]
                in_bandwidth = distances_to_center < center_bandwidth

                # Compute the kernel weights for all points within the bandwidth
                kernel_weights = multivariate_normal.pdf(
                    data[in_bandwidth], mean=center, cov=center_bandwidth ** 2 * np.eye(2)
                )

                # Add weighted contributions to the density estimates
                density_estimates[in_bandwidth] += kernel_weights / n_samples

                num_centers += 1

        # Normalize the density estimates to ensure they sum to 1
        density_estimates /= np.sum(density_estimates)

        if use_histogram:
            # Histogram-based entropy estimation
            # Create histogram of the density estimates
            # Here we once again have to partition into bins..
            # Instead of points we do the density estimates.
            hist, bin_edges = np.histogram(density_estimates, bins=num_bins, density=True)
            # Calculate the probability for each bin
            p_i = hist / np.sum(hist)
            # Filter out zero values to avoid log(0)
            p_i = p_i[p_i > 0]
            # Compute the entropy
            entropy = -np.sum(p_i * np.log(p_i))
        else:
            # Kernel-based entropy estimation
            # Filter out zero or very small density values to avoid log(0)
            nonzero_density = density_estimates[density_estimates > 1e-10]
            # Calculate entropy using the normalized density estimates
            entropy = -np.sum(nonzero_density * np.log(nonzero_density))

        if return_density:
            return entropy, density_estimates
        else:
            return entropy

    def compute_variance(self):
        # variance from center
        # Compute the variance of the distances (magnitudes)
        variance_value = np.var(self.distances)
        return variance_value

    def compute_density_weighted_variance(self):
        # variance weighted by the area of the slice of the bin
        # Divide the unit disc into concentric annular regions
        num_bins = 50
        max_radius = 1.0
        bin_edges = np.linspace(0, max_radius, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        counts, _ = np.histogram(self.distances, bins=bin_edges)

        # Calculate the density of points in each region
        areas = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)
        densities = counts / areas

        # Compute the weighted variance
        weighted_variance = np.average((bin_centers - np.mean(bin_centers)) ** 2, weights=densities)
        return weighted_variance

    def compute_radial_density_entropy(self):
        # instead of partitioning the area by squares, we do it by radial blocks
        # Divide the unit disc into concentric annular regions
        num_bins = 50
        max_radius = 1.0
        bin_edges = np.linspace(0, max_radius, num_bins + 1)
        counts, _ = np.histogram(self.distances, bins=bin_edges)

        # Calculate the density of points in each region
        areas = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)
        densities = counts / areas

        # Normalize the densities to form a probability distribution
        total_density = np.sum(densities)
        if total_density > 0:
            normalized_densities = densities / total_density
        else:
            normalized_densities = densities  # In case there are no points, avoid division by zero

        # Compute the entropy of the radial density distribution
        nonzero_densities = normalized_densities[normalized_densities > 0]
        radial_entropy = entropy(nonzero_densities)

        return radial_entropy

    def compute_polar_density_entropy(self):
        # Convert points to polar coordinates (r, theta)
        r = self.distances
        theta = self.angles

        # Create a 2D histogram in polar coordinates
        radial_edges = np.linspace(0, 1, self.num_radial_bins + 1)
        angular_edges = np.linspace(-np.pi, np.pi, self.num_angular_bins + 1)
        hist, _, _ = np.histogram2d(r, theta, bins=[radial_edges, angular_edges])

        # Normalize each bin by its area in polar coordinates
        radial_widths = np.diff(radial_edges)
        angular_widths = np.diff(angular_edges)
        areas = np.outer(radial_widths, angular_widths) * (radial_edges[:-1] + radial_edges[1:]) / 2
        hist = hist / areas

        # Normalize to form a probability density function (PDF)
        pdf = hist / np.sum(hist)

        # Compute the entropy of the distribution
        # Flatten the PDF to 1D and filter out zero probabilities to avoid log(0)
        pdf_nonzero = pdf[pdf > 0]
        polar_entropy = entropy(pdf_nonzero)

        return polar_entropy

    def compute_kl_divergence(self):
        if self.predefined_distribution is None:
            raise ValueError("Predefined distribution must be provided for KL divergence computation.")

        # Convert points to polar coordinates (r, theta)
        r = self.distances
        theta = self.angles

        # Create a 2D histogram in polar coordinates for the given points
        radial_edges = np.linspace(0, 1, self.num_radial_bins + 1)
        angular_edges = np.linspace(-np.pi, np.pi, self.num_angular_bins + 1)
        hist, _, _ = np.histogram2d(r, theta, bins=[radial_edges, angular_edges])

        # Normalize to form a probability density function (PDF) for the given points
        pdf = hist / np.sum(hist)

        # Compute the KL divergence between the given points' distribution and the predefined distribution
        pdf_nonzero = pdf[pdf > 0]
        predefined_nonzero = self.predefined_distribution[pdf > 0]
        kl_divergence = entropy(pdf_nonzero, qk=predefined_nonzero)

        return kl_divergence

    def analyze(self):
        # Compute all measures and return as a dictionary
        results = {
            "entropy": self.compute_entropy(),
            "variance": self.compute_variance(),
            "density_weighted_variance": self.compute_density_weighted_variance(),
            "radial_density_entropy": self.compute_radial_density_entropy(),
            "polar_density_entropy": self.compute_polar_density_entropy(),
            "parzen": self.fast_parzen_window_pt()[0],
            "parzen_density": self.fast_parzen_window_pt()[1]
        }
        if self.predefined_distribution is not None:
            results["kl_divergence"] = self.compute_kl_divergence()
        return results


# Function to create predefined distributions
def create_predefined_distribution(type, num_radial_bins=10, num_angular_bins=10):
    if type == "uniform":
        distribution = np.ones((num_radial_bins, num_angular_bins))
    elif type == "normal":
        # Create a 2D Gaussian distribution over the unit disc
        x, y = np.meshgrid(np.linspace(-1, 1, num_angular_bins), np.linspace(-1, 1, num_radial_bins))
        pos = np.dstack((x, y))
        rv = multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]])
        distribution = rv.pdf(pos)
    elif type == "exponential":
        # Create an exponential distribution over the unit disc
        r = np.linspace(0, 1, num_radial_bins)
        theta = np.linspace(0, 2 * np.pi, num_angular_bins)
        R, Theta = np.meshgrid(r, theta)
        distribution = np.exp(-R)
    elif type == "unit_circle":
        # Create a distribution that approximates points on the unit circle
        distribution = np.zeros((num_radial_bins, num_angular_bins))
        # Assume the unit circle is approximated by the outermost radial bin
        distribution[-1, :] = 1
    else:
        raise ValueError("Unsupported distribution type")

    distribution /= np.sum(distribution)  # Normalize to form a PDF
    return distribution


# Example usage
if __name__ == "__main__":
    # Generate random points within the unit disc
    num_points = 1000
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    radii = np.sqrt(np.random.uniform(0, 1, num_points))
    points = radii * np.exp(1j * angles)

    # Create a predefined distribution (e.g., uniform, normal, exponential, unit_circle)
    predefined_distribution = create_predefined_distribution("unit_circle", num_radial_bins=10, num_angular_bins=10)

    # Create an instance of the analyzer with the predefined distribution and analyze the points
    analyzer = PointDistributionAnalyzer(points, predefined_distribution=predefined_distribution)
    results = analyzer.analyze()

    print("Analysis Results:", results)

    # Visualize the distribution of points
    plt.figure(figsize=(8, 8))
    plt.scatter(points.real, points.imag, alpha=0.5, edgecolor='k')
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.title('Distribution of Points within the Unit Disc')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

    # Plot the histogram of distances
    plt.figure(figsize=(8, 4))
    plt.hist(analyzer.distances, bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel('Distance from Origin')
    plt.ylabel('Frequency')
    plt.title('Histogram of Distances from the Origin')
    plt.grid(True)
    plt.show()
