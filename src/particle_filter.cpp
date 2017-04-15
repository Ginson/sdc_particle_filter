///
/// @file
/// @brief 2D particle filter class.
///

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

#include "particle_filter.h"

static const auto kPi = 3.14159265358979323846;

void ParticleFilter::Initialize(const double gps_x, const double gps_y, const double yaw, const double std[])
{
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, yaw and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    std::default_random_engine gen;

    // Fetch the std deviations
    auto std_x = std[0];
    auto std_y = std[1];
    auto std_yaw = std[2];

    // This line creates a normal (Gaussian) distribution for x
    std::normal_distribution<double> dist_x(gps_x, std_x);
    std::normal_distribution<double> dist_y(gps_y, std_y);
    std::normal_distribution<double> dist_yaw(yaw, std_yaw);

    num_particles_ = 50;
    is_initialized_ = true;

    // First, simply fill a number of particles and weights into our vectors
    // Init particles with Gaussian noise
    auto given_id = 0;
    for (auto i = 0; i < num_particles_; ++i)
    {
        Particle particle;
        particle.id = given_id;

        auto sample_x = dist_x(gen);
        auto sample_y = dist_y(gen);
        auto sample_yaw = dist_yaw(gen);

        particle.x = sample_x;
        particle.y = sample_y;
        particle.yaw = sample_yaw;

        particle.weight = 1.0;

        particles_.push_back(particle);
        given_id++;
    }
}

void ParticleFilter::Predict(const double delta_t, const double std_pos[], const double velocity, const double yaw_rate)
{
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    std::default_random_engine gen;

    auto std_x = std_pos[0];
    auto std_y = std_pos[1];
    auto std_yaw = std_pos[2];

    for (auto& particle : particles_)
    {
        auto v_per_yaw_rate = (velocity / yaw_rate);
        auto yaw_rate_times_delta_t = yaw_rate * delta_t;

        particle.x = particle.x + (v_per_yaw_rate * (sin(particle.yaw + yaw_rate_times_delta_t) - sin(particle.yaw)));
        particle.y = particle.y + (v_per_yaw_rate * (cos(particle.yaw) - cos(particle.yaw + yaw_rate_times_delta_t)));
        particle.yaw = particle.yaw + (yaw_rate * delta_t);

        // Add some noise
        std::normal_distribution<double> dist_x(particle.x, std_x);
        std::normal_distribution<double> dist_y(particle.y, std_y);
        std::normal_distribution<double> dist_yaw(particle.yaw, std_yaw);

        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.yaw = dist_yaw(gen);
    }
}

void ParticleFilter::UpdateWeights(const double sensor_range,
                                   const double std_landmark[],
                                   const std::vector<LandmarkObs>& observations,
                                   const Map& map_landmarks)
{
    // TODO: Update the weights of each particle using a multi-variant Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
    //   for the fact that the map's y-axis actually points downwards.)
    //   http://planning.cs.uiuc.edu/node99.html

    // Actually this is not used in my approach as we do not predict measurements,
    // but directly transform the given observations into map space.
    sensor_range;

    for (auto& particle : particles_)
    {
        auto current_observations = observations;

        // Transform car to map coordinate system
        TransformToMapCoordinateSystem(particle, current_observations);

        // Data association
        AssociateObservationsWithLandmarks(map_landmarks, current_observations);

        // Computes the weight for the particle
        auto weight = ComputeWeight(std_landmark, map_landmarks, current_observations);

        // Assign weight to particle
        particle.weight = weight;
    }
}

void ParticleFilter::TransformToMapCoordinateSystem(const Particle& particle, std::vector<LandmarkObs>& observations)
{
    auto yaw = particle.yaw;
    auto cos_of_yaw = cos(yaw);
    auto sin_of_yaw = sin(yaw);

    for (auto& observation : observations)
    {
        auto x = observation.x * cos_of_yaw - observation.y * sin_of_yaw + particle.x;
        auto y = observation.x * sin_of_yaw + observation.y * cos_of_yaw + particle.y;

        observation.x = x;
        observation.y = y;
    }
}

void ParticleFilter::AssociateObservationsWithLandmarks(const Map& predictions, std::vector<LandmarkObs>& observations)
{
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    for (auto& observation : observations)
    {
        auto min_distance = std::numeric_limits<double>::max();
        for (const auto& prediction : predictions.landmark_list)
        {
            auto x = observation.x - prediction.x_f;
            auto y = observation.y - prediction.y_f;
            auto dist = sqrt(pow(x, 2) + pow(y, 2));

            if (dist < min_distance)
            {
                min_distance = dist;
                observation.id = prediction.id_i;
            }
        }
    }
}

double ParticleFilter::ComputeWeight(const double std_landmark[],
                                     const Map& map_landmarks,
                                     const std::vector<LandmarkObs>& observations)
{
    auto cov_x = std_landmark[0];
    auto cov_y = std_landmark[1];

    // Some helper for latter computations, does only have to be computed once.
    auto inverse_factor = 1 / (cov_x * cov_y);
    auto normalizer = sqrt(2 * kPi * cov_x * cov_y);

    // Initially set to 1 as we start off from new to compute the weight
    double weight = 1.0;
    for (const auto& observation : observations)
    {
        auto x_i = observation;

        // Here we get the landmark whose id matches the observation id
        const int id_to_find = observation.id;
        auto u_i =
            std::find_if(map_landmarks.landmark_list.begin(),
                         map_landmarks.landmark_list.end(),
                         [id_to_find](const Map::single_landmark_s& item) -> bool { return item.id_i == id_to_find; });

        // The computation of the multi-variant Gaussian distribution is divided into chunks of small computations to
        // avoid having to use Eigen or another library, since we are only in the 2-D case.
        auto diff_x = x_i.x - u_i->x_f;
        auto diff_y = x_i.y - u_i->y_f;

        auto term_for_x = diff_x * inverse_factor * cov_y;
        auto term_for_y = diff_y * inverse_factor * cov_x;

        auto inner_term = diff_x * term_for_x + diff_y * term_for_y;

        auto term = exp(-0.5 * inner_term);

        auto w = term / normalizer;
        weight *= w;
    }

    return weight;
}

void ParticleFilter::Resample()
{
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::default_random_engine gen;

    std::vector<double> all_weights;
    for (auto& particle : particles_)
    {
        all_weights.push_back(particle.weight);
    }

    // Create a distribution equivalent to the weights per particle.
    std::discrete_distribution<> d(all_weights.begin(), all_weights.end());
    std::vector<Particle> particles_resampled;

    for (int n = 0; n < num_particles_; ++n)
    {
        auto to_be_sampled = d(gen);
        Particle resampled_particle = particles_[to_be_sampled];
        particles_resampled.push_back(resampled_particle);
    }

    // Set the internal list of particles to the re-sampled ones
    particles_ = particles_resampled;
}

void ParticleFilter::Write(std::string filename)
{
    // You don't need to modify this file.
    std::ofstream dataFile;
    dataFile.open(filename, std::ios::app);
    for (int i = 0; i < num_particles_; ++i)
    {
        dataFile << particles_[i].x << " " << particles_[i].y << " " << particles_[i].yaw << "\n";
    }
    dataFile.close();
}
