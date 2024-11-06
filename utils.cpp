#include <iostream>
#include <filesystem>
#include <system_error>
#include <string>
#include <unistd.h> // For access() and mknod()
#include <sys/stat.h> // For mknod()
#include <fcntl.h> // For O_* constant
#include <vector>
#include <sstream>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <optional>
#include <tinyxml2.h>
#include <map>

using namespace tinyxml2;




namespace fs = std::filesystem;


void safe_mkdir_recursive(const fs::path& directory, bool overwrite = false) {
    try {
        // Check if directory exists

        if (!fs::exists(directory)) {
            // Create directories recursively

            fs::create_directories(directory);
        } else if (overwrite) {
            // If overwrite is true, remove and recreate the directory
            
            fs::remove_all(directory);
            fs::create_directories(directory);
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

bool safe_mknode_recursive(const fs::path& destiny_dir, const std::string& node_name, bool overwrite) {
    // Create the directory safely

    if (!safe_mkdir_recursive(destiny_dir)) {
        return false;
    }

    fs::path node_path = destiny_dir / node_name;

    // Remove the node if it exists and overwrite is true

    if (overwrite && fs::exists(node_path)) {
        fs::remove(node_path);
    }

    // Check if the node already exists

    if (!fs::exists(node_path)) {
        // Create a node (FIFO in this case)
        if (mknod(node_path.c_str(), S_IFIFO | 0666, 0) == -1) {
            perror("mknod failed");
            return true; // Return true to indicate failure in creating the node
        }
        return false; // Node was created successfully
    }
    return true; // Node already exists
}


std::vector<std::vector<double>> jsonify(const Eigen::MatrixXd& array) {
    std::vector<std::vector<double>> result(array.rows(), std::vector<double>(array.cols()));
    for (int i = 0; i < array.rows(); ++i) {
        for (int j = 0; j < array.cols(); ++j) {
            result[i][j] = array(i, j);
        }
    }
    return result;
}

// Function to convert Eigen matrix to std::vector

std::vector<std::vector<double>> jsonify(const Eigen::MatrixXd& array) {
    std::vector<std::vector<double>> result(array.rows(), std::vector<double>(array.cols()));
    for (int i = 0; i < array.rows(); ++i) {
        for (int j = 0; j < array.cols(); ++j) {
            result[i][j] = array(i, j);
        }
    }
    return result;
}

// Overloaded function for std::vector

std::vector<double> jsonify(const std::vector<double>& array) {
    return array; // Return the vector as is
}

// Function to undo the JSONify process

Eigen::MatrixXd undo_jsonify(const std::vector<std::string>& array) {
    std::vector<std::vector<double>> x;

    for (const auto& elem : array) {
        std::string nums_str = elem.substr(elem.find('[') + 1);
        nums_str = nums_str.substr(0, nums_str.find(']'));
        
        std::vector<double> a;
        std::stringstream ss(nums_str);
        std::string num;

        while (std::getline(ss, num, ',')) {
            a.push_back(std::stod(num));
        }
        x.push_back(a);
    }


std::vector<double> linear_interpolate(const std::vector<double>& t, const std::vector<double>& x, const std::vector<double>& t_interp) {
    std::vector<double> x_interp(t_interp.size());
    
    for (size_t i = 0; i < t_interp.size(); ++i) {
        // Find the interval [t[j], t[j+1]] that contains t_interp[i]
        auto it = std::lower_bound(t.begin(), t.end(), t_interp[i]);
        size_t j = std::distance(t.begin(), it) - 1;
        
        // Handle edge cases
        if (j >= t.size() - 1) {
            x_interp[i] = x.back(); // Use last value if out of bounds
            continue;
        } else if (j < 0) {
            x_interp[i] = x.front(); // Use first value if out of bounds
            continue;
        }

        // Perform linear interpolation
        double alpha = (t_interp[i] - t[j]) / (t[j + 1] - t[j]);
        x_interp[i] = (1 - alpha) * x[j] + alpha * x[j + 1];
    }
    
    return x_interp;
}

// Function to calculate Mean Squared Error of interpolated values

double interpol_mse(const std::vector<double>& t_1, const Eigen::MatrixXd& x_1, const std::vector<double>& t_2, const Eigen::MatrixXd& x_2, size_t n_interp_samples = 1000) {
    if (t_1 == t_2) {
        return (x_1 - x_2).squaredNorm() / x_1.rows(); // Return mean squared error if both time vectors are identical
    }

    assert(x_1.cols() == x_2.cols());

    double t_min = std::max(t_1.front(), t_2.front());
    double t_max = std::min(t_1.back(), t_2.back());

    std::vector<double> t_interp(n_interp_samples);
    for (size_t i = 0; i < n_interp_samples; ++i) {
        t_interp[i] = t_min + i * (t_max - t_min) / (n_interp_samples - 1);
    }

    Eigen::MatrixXd err(n_interp_samples, x_1.cols());
    for (size_t dim = 0; dim < x_1.cols(); ++dim) {
        // Interpolate x_1 and x_2 along the current dimension
        std::vector<double> x1_interp = linear_interpolate(t_1, std::vector<double>(x_1.col(dim).data(), x_1.col(dim).data() + x_1.rows()), t_interp);
        std::vector<double> x2_interp = linear_interpolate(t_2, std::vector<double>(x_2.col(dim).data(), x_2.col(dim).data() + x_2.rows()), t_interp);
        
        for (size_t i = 0; i < n_interp_samples; ++i) {
            err(i, dim) = x1_interp[i] - x2_interp[i];
        }
    }

    // Calculate the mean squared error
    double mse = 0.0;
    for (size_t i = 0; i < n_interp_samples; ++i) {
        mse += err.row(i).squaredNorm();
    }
    return mse / n_interp_samples; // Return mean squared error
}

// Function to calculate Euclidean distance between two points

std::optional<bool> euclidean_dist(const Eigen::VectorXd& x, const Eigen::VectorXd& y, std::optional<double> thresh = std::nullopt) {
    // Ensure x and y have the same dimension
    if (x.size() != y.size()) {
        std::cerr << "Error: Points x and y must have the same dimensions." << std::endl;
        return std::nullopt; // Return nullopt to indicate an error
    }

    double dist = (x - y).norm(); // Calculate the Euclidean distance

    if (!thresh.has_value()) {
        return dist; // Return distance if no threshold is provided
    }

    return dist < thresh.value(); // Return whether the distance is smaller than the threshold
}

// Function to convert Euler angles (roll, pitch, yaw) to a quaternion

Eigen::Vector4d euler_to_quaternion(double roll, double pitch, double yaw) {
    double qx = std::sin(roll / 2) * std::cos(pitch / 2) * std::cos(yaw / 2) - std::cos(roll / 2) * std::sin(pitch / 2) * std::sin(yaw / 2);
    double qy = std::cos(roll / 2) * std::sin(pitch / 2) * std::cos(yaw / 2) + std::sin(roll / 2) * std::cos(pitch / 2) * std::sin(yaw / 2);
    double qz = std::cos(roll / 2) * std::cos(pitch / 2) * std::sin(yaw / 2) - std::sin(roll / 2) * std::sin(pitch / 2) * std::cos(yaw / 2);
    double qw = std::cos(roll / 2) * std::cos(pitch / 2) * std::cos(yaw / 2) + std::sin(roll / 2) * std::sin(pitch / 2) * std::sin(yaw / 2);

    return Eigen::Vector4d(qw, qx, qy, qz);
}

std::vector<double> quaternion_to_euler(const Eigen::Vector4d& q) {
    double roll = std::atan2(2.0 * (q(0) * q(1) + q(2) * q(3)),
                              1.0 - 2.0 * (q(1) * q(1) + q(2) * q(2)));
    double pitch = std::asin(2.0 * (q(0) * q(2) - q(3) * q(1)));
    double yaw = std::atan2(2.0 * (q(0) * q(3) + q(1) * q(2)),
                             1.0 - 2.0 * (q(2) * q(2) + q(3) * q(3)));
    return {roll, pitch, yaw};
}

// Function to normalize a quaternion to unit modulus

Eigen::Vector4d unit_quat(const Eigen::Vector4d& q) {
    double q_norm = q.norm(); // Calculate the norm of the quaternion
    if (q_norm == 0) {
        return Eigen::Vector4d(1, 0, 0, 0); // Return the identity quaternion if norm is zero
    }
    return q / q_norm; // Normalize the quaternion
}

// Function to convert quaternion to rotation matrix

Eigen::Matrix3d q_to_rot_mat(const Eigen::Vector4d& q) {
    Eigen::Matrix3d rot_mat;
    double q0 = q(0), q1 = q(1), q2 = q(2), q3 = q(3);
    rot_mat << 1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2),
               2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1),
               2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2);
    return rot_mat;
}

// Function to rotate vector v by quaternion q

Eigen::Vector3d v_dot_q(const Eigen::Vector3d& v, const Eigen::Vector4d& q) {
    Eigen::Matrix3d rot_mat = q_to_rot_mat(q);
    return rot_mat * v; // Rotate vector v using rotation matrix
}

// Function to convert a quaternion to a rotation matrix

Eigen::Matrix3d q_to_rot_mat(const Eigen::Vector4d& q) {
    double qw = q(0), qx = q(1), qy = q(2), qz = q(3);
    Eigen::Matrix3d rot_mat;

    rot_mat << 1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
               2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
               2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy);

    return rot_mat;
}

// Function to apply the rotation of quaternion r to quaternion q

Eigen::Vector4d q_dot_q(const Eigen::Vector4d& q, const Eigen::Vector4d& r) {
    double qw = q(0), qx = q(1), qy = q(2), qz = q(3);
    double rw = r(0), rx = r(1), ry = r(2), rz = r(3);

    double t0 = rw * qw - rx * qx - ry * qy - rz * qz;
    double t1 = rw * qx + rx * qw - ry * qz + rz * qy;
    double t2 = rw * qy + rx * qz + ry * qw - rz * qx;
    double t3 = rw * qz - rx * qy + ry * qx + rz * qw;

    return Eigen::Vector4d(t0, t1, t2, t3); // Return the resulting quaternion
}

// Function to calculate a quaternion from a rotation matrix

Eigen::Vector4d rotation_matrix_to_quat(const Eigen::Matrix3d& rot) {
    double qw, qx, qy, qz;
    double trace = rot.trace();

    if (trace > 0) {
        double s = 0.5 / sqrt(trace + 1.0);
        qw = 0.25 / s;
        qx = (rot(2, 1) - rot(1, 2)) * s;
        qy = (rot(0, 2) - rot(2, 0)) * s;
        qz = (rot(1, 0) - rot(0, 1)) * s;
    } else {
        if (rot(0, 0) > rot(1, 1) && rot(0, 0) > rot(2, 2)) {
            double s = 2.0 * sqrt(1.0 + rot(0, 0) - rot(1, 1) - rot(2, 2));
            qw = (rot(2, 1) - rot(1, 2)) / s;
            qx = 0.25 * s;
            qy = (rot(0, 1) + rot(1, 0)) / s;
            qz = (rot(0, 2) + rot(2, 0)) / s;
        } else if (rot(1, 1) > rot(2, 2)) {
            double s = 2.0 * sqrt(1.0 + rot(1, 1) - rot(0, 0) - rot(2, 2));
            qw = (rot(0, 2) - rot(2, 0)) / s;
            qx = (rot(0, 1) + rot(1, 0)) / s;
            qy = 0.25 * s;
            qz = (rot(1, 2) + rot(2, 1)) / s;
        } else {
            double s = 2.0 * sqrt(1.0 + rot(2, 2) - rot(0, 0) - rot(1, 1));
            qw = (rot(1, 0) - rot(0, 1)) / s;
            qx = (rot(0, 2) + rot(2, 0)) / s;
            qy = (rot(1, 2) + rot(2, 1)) / s;
            qz = 0.25 * s;
        }
    }

    return Eigen::Vector4d(qw, qx, qy, qz); // Return the quaternion as a vector
}

// Function to correct quaternion flips

Eigen::Vector4d undo_quaternion_flip(const Eigen::Vector4d& q_past, const Eigen::Vector4d& q_current) {
    double dist_current = (q_past - q_current).norm();
    double dist_flipped = (q_past + q_current).norm();

    if (dist_current > dist_flipped) {
        return -q_current; // Flip the quaternion
    }
    return q_current; // Return as is
}

Eigen::Matrix4d skew_symmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix4d skew_mat;
    skew_mat << 0, -v[0], -v[1], -v[2],
                v[0], 0, v[2], -v[1],
                v[1], -v[2], 0, v[0],
                v[2], v[1], -v[0], 0;
    return skew_mat;
}

// Function to compute the inverse of a quaternion

Eigen::Vector4d quaternion_inverse(const Eigen::Vector4d& q) {
    return Eigen::Vector4d(q[0], -q[1], -q[2], -q[3]) / q.squaredNorm();
}

// Function to normalize a quaternion

Eigen::Vector4d unit_quat(const Eigen::Vector4d& q) {
    return q / q.norm();
}

// Function to apply quaternion multiplication

Eigen::Vector4d q_dot_q(const Eigen::Vector4d& q, const Eigen::Vector4d& r) {
    double qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    double rw = r[0], rx = r[1], ry = r[2], rz = r[3];

    double t0 = rw * qw - rx * qx - ry * qy - rz * qz;
    double t1 = rw * qx + rx * qw - ry * qz + rz * qy;
    double t2 = rw * qy + rx * qz + ry * qw - rz * qx;
    double t3 = rw * qz - rx * qy + ry * qx + rz * qw;

    return Eigen::Vector4d(t0, t1, t2, t3);
}

// Function to decompose a quaternion into a z rotation and an xy rotation

std::pair<Eigen::Vector4d, Eigen::Vector4d> decompose_quaternion(const Eigen::Vector4d& q) {
    double w = q[0], x = q[1], y = q[2], z = q[3];

    Eigen::Vector4d qz;
    if (x == 0 && y == 0 && z == 0) {
        qz = unit_quat(Eigen::Vector4d(w, 0, 0, z)); // Handle zero vector case
    } else {
        qz = unit_quat(Eigen::Vector4d(w, 0, 0, z));
    }

    Eigen::Vector4d qxy = q_dot_q(q, quaternion_inverse(qz));

    return std::make_pair(qxy, qz);
}

// Function to convert a rotation matrix to Euler angles

Eigen::Vector3d rotation_matrix_to_euler(const Eigen::Matrix3d& r_mat) {
    double sy = std::sqrt(r_mat(0, 0) * r_mat(0, 0) + r_mat(1, 0) * r_mat(1, 0));

    bool singular = sy < 1e-6;

    double x, y, z;
    if (!singular) {
        x = std::atan2(r_mat(2, 1), r_mat(2, 2));
        y = std::atan2(-r_mat(2, 0), sy);
        z = std::atan2(r_mat(1, 0), r_mat(0, 0));
    } else {
        x = std::atan2(-r_mat(1, 2), r_mat(1, 1));
        y = std::atan2(-r_mat(2, 0), sy);
        z = 0;
    }

    return Eigen::Vector3d(x, y, z);
}



// Helper function to compute histogram
std::pair<std::vector<int>, std::vector<double>> compute_histogram(const Eigen::VectorXd& data, int bins) {
    double min_val = data.minCoeff();
    double max_val = data.maxCoeff();
    double bin_width = (max_val - min_val) / bins;

    std::vector<int> hist(bins, 0);
    std::vector<double> bin_edges(bins + 1);
    for (int i = 0; i <= bins; ++i) {
        bin_edges[i] = min_val + i * bin_width;
    }

    for (int i = 0; i < data.size(); ++i) {
        int bin = std::min(static_cast<int>((data[i] - min_val) / bin_width), bins - 1);
        hist[bin]++;
    }
    return {hist, bin_edges};
}

// Main function for pruning dataset with optional plotting
std::vector<int> prune_dataset(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double x_cap, int bins, double thresh, bool plot, const std::vector<std::string>& labels = {}) {
    int n_samples = x.rows();
    int y_dims = y.cols();
    std::vector<int> pruned_indices;

    

    // Prune by velocity cap
    for (int i = 0; i < x.cols(); ++i) {
        for (int j = 0; j < n_samples; ++j) {
            if (std::abs(x(j, i)) > x_cap) {
                pruned_indices.push_back(j);
            }
        }
    }

    std::sort(pruned_indices.begin(), pruned_indices.end());
    pruned_indices.erase(std::unique(pruned_indices.begin(), pruned_indices.end()), pruned_indices.end());

    // Prune by error histogram (dimension-wise)
    for (int i = 0; i < y_dims; ++i) {
        Eigen::VectorXd y_col = y.col(i);
        auto [hist, bin_edges] = compute_histogram(y_col, bins);

        int total_count = y_col.size();
        for (int j = 0; j < bins; ++j) {
            if (static_cast<double>(hist[j]) / total_count < thresh) {
                for (int k = 0; k < n_samples; ++k) {
                    if (y_col[k] >= bin_edges[j] && y_col[k] < bin_edges[j + 1]) {
                        pruned_indices.push_back(k);
                    }
                }
            }
        }
    }

    // Prune by error norm
    Eigen::VectorXd y_norm = y.rowwise().norm();
    auto [hist_norm, bin_edges_norm] = compute_histogram(y_norm, bins);
    int total_count = y_norm.size();
    for (int j = 0; j < bins; ++j) {
        if (static_cast<double>(hist_norm[j]) / total_count < thresh) {
            for (int k = 0; k < n_samples; ++k) {
                if (y_norm[k] >= bin_edges_norm[j] && y_norm[k] < bin_edges_norm[j + 1]) {
                    pruned_indices.push_back(k);
                }
            }
        }
    }

    std::sort(pruned_indices.begin(), pruned_indices.end());
    pruned_indices.erase(std::unique(pruned_indices.begin(), pruned_indices.end()), pruned_indices.end());

  

    std::vector<int> kept_indices;
    for (int i = 0; i < n_samples; ++i) {
        if (std::find(pruned_indices.begin(), pruned_indices.end(), i) == pruned_indices.end()) {
            kept_indices.push_back(i);
        }
    }

    return kept_indices;
}



std::vector<int> distance_maximizing_points_1d(const Eigen::VectorXd &points, int n_train_points) {
    std::vector<int> closest_points(n_train_points, 0);

    // Create a histogram with `n_train_points` bins
    double min = points.minCoeff();
    double max = points.maxCoeff();
    double bin_width = (max - min) / n_train_points;

    // Digitize points to assign bins
    std::vector<int> hist_indices(points.size());
    for (int i = 0; i < points.size(); i++) {
        hist_indices[i] = std::min(static_cast<int>((points[i] - min) / bin_width), n_train_points - 1);
    }

    for (int i = 0; i < n_train_points; i++) {
        // Get points in the current bin
        std::vector<double> bin_values;
        for (int j = 0; j < points.size(); j++) {
            if (hist_indices[j] == i) {
                bin_values.push_back(points[j]);
            }
        }

        if (!bin_values.empty()) {
            double median = bin_values[bin_values.size() / 2]; // Simple median computation
            auto it = std::find(points.data(), points.data() + points.size(), median);
            closest_points[i] = std::distance(points.data(), it);
        } else {
            closest_points[i] = rand() % points.size(); // Random selection if bin is empty
        }
    }

    return closest_points;
}


std::vector<int> distance_maximizing_points_2d(const Eigen::MatrixXd &points, int n_train_points, bool plot=false) {
    int n_clusters = (n_train_points > 30) ? std::max(n_train_points / 10, 30) : n_train_points;
    int n_samples = n_train_points / n_clusters;

    std::vector<int> kmeans_labels(points.rows());

    // Apply KMeans clustering
    // Assuming you use an external KMeans library here
    KMeans kmeans(n_clusters);
    kmeans.Fit(points, kmeans_labels);

    std::vector<int> closest_points;
    for (int i = 0; i < n_clusters; i++) {
        std::vector<int> cluster_indices;
        for (int j = 0; j < points.rows(); j++) {
            if (kmeans_labels[j] == i) cluster_indices.push_back(j);
        }

        // Randomly pick points from each cluster
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, cluster_indices.size() - 1);
        for (int s = 0; s < n_samples; s++) {
            closest_points.push_back(cluster_indices[dis(gen)]);
        }
    }

    return closest_points;
}


std::vector<int> distance_maximizing_points(const MatrixXd& x_values, const VectorXd& center, int n_train_points) {
    if (x_values.cols() == 1) {
        return distance_maximizing_points_1d(x_values, n_train_points);
    }
    
    if (x_values.cols() >= 2) {
        return distance_maximizing_points_2d(x_values, n_train_points);
    }

    // PCA computation (3D case)
    MatrixXd centered = x_values.rowwise() - center.transpose();
    MatrixXd cov_matrix = centered.transpose() * centered / double(x_values.rows() - 1);
    SelfAdjointEigenSolver<MatrixXd> eigensolver(cov_matrix);
    MatrixXd pca_axes = eigensolver.eigenvectors().rightCols(3);

    MatrixXd points_pca = centered * pca_axes;
    Vector3d p_min = points_pca.colwise().minCoeff();
    Vector3d p_max = points_pca.colwise().maxCoeff();

    std::vector<Vector3d> centroids = { center };

    // Generate centroids (pyramids and cuboids as required)
    if (n_train_points >= 15) {
        centroids.push_back(Vector3d(p_max[0], center[1], center[2]));
        centroids.push_back(Vector3d(center[0], p_max[1], center[2]));
        centroids.push_back(Vector3d(center[0], center[1], p_max[2]));
        centroids.push_back(Vector3d(p_min[0], center[1], center[2]));
        centroids.push_back(Vector3d(center[0], p_min[1], center[2]));
        centroids.push_back(Vector3d(center[0], center[1], p_min[2]));
    }

    // Find closest points to centroids
    std::vector<int> closest_points(centroids.size(), -1);
    for (size_t i = 0; i < centroids.size(); ++i) {
        Vector3d centroid = centroids[i];
        double min_dist = std::numeric_limits<double>::max();
        int min_idx = -1;

        for (int j = 0; j < points_pca.rows(); ++j) {
            double dist = (points_pca.row(j) - centroid.transpose()).norm();
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = j;
            }
        }

        closest_points[i] = min_idx;
    }

    return closest_points;
}



std::vector<int> sample_random_points(const MatrixXd& points, std::vector<int>& used_idx, int points_to_sample) {
    int bins = std::min(10, static_cast<int>(points.rows() / 10));
    bins = std::max(bins, 2);

    // Find free points by excluding used indices
    std::vector<int> free_points;
    for (int i = 0; i < points.rows(); ++i) {
        if (std::find(used_idx.begin(), used_idx.end(), i) == used_idx.end()) {
            free_points.push_back(i);
        }
    }

    int n_samples = std::min(points_to_sample, static_cast<int>(free_points.size()));

    // Compute histogram for free points
    std::vector<std::vector<int>> assignments = compute_histogram_bins(points, bins, free_points);

    // Compute inverse histogram-based probabilities
    std::vector<double> probs(free_points.size(), 1.0);
    double total_prob = 0.0;
    for (size_t i = 0; i < free_points.size(); ++i) {
        int bin_count = 1;
        for (int j = 0; j < points.cols(); ++j) {
            bin_count *= assignments[j][i] + 1; // Simple approximation for bin count
        }
        probs[i] = 1.0 / bin_count;  // Assign inverse count as probability weight
        total_prob += probs[i];
    }

    // Normalize probabilities
    for (double& prob : probs) {
        prob /= total_prob;
    }

    // Randomly sample from free points using computed probabilities
    std::vector<int> selected_points;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());

    while (selected_points.size() < static_cast<size_t>(n_samples)) {
        int sampled_index = dist(gen);
        int point_index = free_points[sampled_index];
        
        if (std::find(selected_points.begin(), selected_points.end(), point_index) == selected_points.end()) {
            selected_points.push_back(point_index);
        }
    }

    // Add selected points to used indices
    used_idx.insert(used_idx.end(), selected_points.begin(), selected_points.end());
    return used_idx;
}


// Helper function to convert attribute list to a map
map<string, string> parseAttributes(XMLNode* node) {
    map<string, string> attributesMap;
    if (node == nullptr) return attributesMap;

    XMLElement* element = node->ToElement();
    if (element) {
        const XMLAttribute* attr = element->FirstAttribute();
        while (attr) {
            attributesMap[attr->Name()] = attr->Value();
            attr = attr->Next();
        }
    }
    return attributesMap;
}

// Function to parse a xacro file and return a map of attributes
map<string, string> parseXacroFile(const string& xacroFilePath) {
    XMLDocument doc;
    map<string, string> attribDict;

    // Load the xacro file
    XMLError loadResult = doc.LoadFile(xacroFilePath.c_str());
    if (loadResult != XML_SUCCESS) {
        cerr << "Error loading file " << xacroFilePath << endl;
        return attribDict;
    }

    // Traverse root elements and get attributes
    XMLNode* root = doc.RootElement();
    if (root == nullptr) {
        cerr << "Error: No root element in XML file" << endl;
        return attribDict;
    }

    // Iterate over child elements of the root
    for (XMLNode* node = root->FirstChild(); node; node = node->NextSibling()) {
        XMLElement* element = node->ToElement();
        if (element) {
            const char* name = element->Attribute("name");
            const char* value = element->Attribute("value");

            // If 'value' attribute exists, add it to the map
            if (value && name) {
                attribDict[name] = value;
            }

            // If there are child nodes, store them in the map
            vector<map<string, string>> childAttributes;
            for (XMLNode* child = node->FirstChild(); child; child = child->NextSibling()) {
                childAttributes.push_back(parseAttributes(child));
            }

            // Store child attributes in the map if they exist
            if (!childAttributes.empty() && name) {
                // Convert child attributes to string (or use a different data structure for complex needs)
                attribDict[name] = "Child attributes present";  // This is a placeholder
            }
        }
    }

    return attribDict;
}


Eigen::MatrixXd makeBxMatrix(int x_dims, const std::vector<int>& y_feats) {
    // Initialize Bx matrix with zeros
    Eigen::MatrixXd bx = Eigen::MatrixXd::Zero(x_dims, y_feats.size());

    // Set the specified elements to 1
    for (size_t i = 0; i < y_feats.size(); ++i) {
        bx(y_feats[i], i) = 1;
    }

    return bx;
}


// Function to create the Bz matrix
Eigen::MatrixXd makeBzMatrix(int x_dims, int u_dims, const std::vector<int>& x_feats, const std::vector<int>& u_feats) {
    Eigen::MatrixXd bz = Eigen::MatrixXd::Zero(x_feats.size(), x_dims);
    
    for (size_t i = 0; i < x_feats.size(); ++i) {
        bz(i, x_feats[i]) = 1;
    }

    Eigen::MatrixXd bzu = Eigen::MatrixXd::Zero(u_feats.size(), u_dims);
    
    for (size_t i = 0; i < u_feats.size(); ++i) {
        bzu(i, u_feats[i]) = 1;
    }

    bz.conservativeResize(bz.rows() + bzu.rows(), bz.cols() + u_dims);
    bz.bottomRows(bzu.rows()) = bzu;

    return bz;
}

// Function to calculate the mean squared error for quaternion state
double quaternionStateMSE(const Eigen::VectorXd& x, const Eigen::VectorXd& x_ref, const Eigen::VectorXd& mask) {
    Eigen::Vector4d q_error = qDotQ(x.segment<4>(3), quaternionInverse(x_ref.segment<4>(3)));
    Eigen::VectorXd e(12);
    
    e.head<3>() = x.head<3>() - x_ref.head<3>();
    e.segment<3>(3) = q_error.tail<3>();
    e.segment<3>(6) = x.segment<3>(7) - x_ref.segment<3>(7);
    e.tail<3>() = x.tail<3>() - x_ref.tail<3>();

    return std::sqrt((e.array() * mask.array()).matrix().squaredNorm());
}



std::vector<Eigen::MatrixXd> separateVariables(const Eigen::MatrixXd& traj) {
    // Check that the input trajectory has the correct dimensions
    if (traj.rows() == 0 || traj.cols() != 13) {
        throw std::invalid_argument("Input trajectory must be an Nx13 matrix.");
    }

    // Separate the trajectory into components
    Eigen::MatrixXd p_traj = traj.leftCols<3>();      // Position trajectory Nx3
    Eigen::MatrixXd a_traj = traj.middleCols<4>(3);   // Quaternion trajectory Nx4
    Eigen::MatrixXd v_traj = traj.middleCols<3>(7);    // Velocity trajectory Nx3
    Eigen::MatrixXd r_traj = traj.rightCols<3>();      // Body rate trajectory Nx3

    // Store components in a vector
    return {p_traj, a_traj, v_traj, r_traj};
}
