///
/// @file
/// @brief 2D particle filter class.
///

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include "helper_functions.h"

struct Particle
{
    int id;
    double x;
    double y;
    double yaw;
    double weight;
};

class ParticleFilter
{
  public:
    // Constructor
    ParticleFilter() : is_initialized_(false), num_particles_(0), particles_() {}

    // Destructor
    ~ParticleFilter() {}

    // Flag, if filter is initialized
    bool is_initialized_;

    // Number of particles to draw
    int num_particles_;

    // Set of current particles
    std::vector<Particle> particles_;

    /// @brief Initializes particle filter by initializing particles to Gaussian
    ///        distribution around first position and all the weights to 1.
    /// @param x Initial x position [m] (simulated estimate from GPS)
    /// @param y Initial y position [m]
    /// @param theta Initial orientation [rad]
    /// @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
    ///              standard deviation of yaw [rad]]
    void Initialize(const double x, const double y, const double theta, const double std[]);

    /// @brief Predicts the state for the next time step
    ///        using the process model.
    /// @param delta_t Time between time step t and t+1 in measurements [s]
    /// @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
    ///        standard deviation of yaw [rad]]
    /// @param velocity Velocity of car from t to t+1 [m/s]
    /// @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
    void Predict(const double delta_t, const double std_pos[], const double velocity, const double yaw_rate);

    /// @brief Updates the weights for each particle based on the likelihood of the
    ///        observed measurements.
    /// @param sensor_range Range [m] of sensor
    /// @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
    ///        standard deviation of bearing [rad]]
    /// @param observations Vector of landmark observations
    /// @param map Map class containing map landmarks
    void UpdateWeights(const double sensor_range,
                       const double std_landmark[],
                       const std::vector<LandmarkObs>& observations,
                       const Map& map_landmarks);

    /// @brief Finds which observations correspond to which landmarks (likely by using
    ///        a nearest-neighbors data association).
    /// @param predicted Vector of predicted landmark observations
    /// @param observations Vector of landmark observations
    void AssociateObservationsWithLandmarks(const Map& predicted, std::vector<LandmarkObs>& observations);

    /// @brief Computes the weight using multi-variant Gaussian distribution
    double ComputeWeight(const double std_landmark[],
                         const Map& map_landmarks,
                         const std::vector<LandmarkObs>& observation);

    /// @brief Performs a coordinate transformation from vehicle to map system
    void TransformToMapCoordinateSystem(const Particle& particle, std::vector<LandmarkObs>& observations);

    /// @brief Re-samples from the updated set of particles to form
    ///   the new set of particles.
    void Resample();

    /// @brief Writes particle positions to a file.
    /// @param filename File to write particle positions to.
    void Write(std::string filename);

    /// @brief Returns whether particle filter is initialized yet or not.
    const bool IsInitialized() const { return is_initialized_; }
};

#endif  // PARTICLE_FILTER_H_
