/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

#include "particle_filter.h"

void ParticleFilter::init(double gps_x, double gps_y, double yaw, double std[])
{
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, yaw and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    std::default_random_engine gen;

    // Fetch the std devs
    auto std_x = std[0];
    auto std_y = std[1];
    auto std_yaw = std[2];

    // This line creates a normal (Gaussian) distribution for x
    std::normal_distribution<double> dist_x(gps_x, std_x);
    std::normal_distribution<double> dist_y(gps_y, std_y);
    std::normal_distribution<double> dist_yaw(yaw, std_yaw);

    num_particles_ = 20;
    is_initialized_ = true;

    // First, simply fill a number of particles and weights into our vectors
    for (auto i = 0; i < num_particles_; ++i)
    {
        particles_.push_back(Particle());
        weights_.push_back(0.0);
    }

    for (auto& weight : weights_)
    {
        weight = 1.0;
    }

    // Init particles with Gaussian noise
    auto given_id = 0;
    for (auto& particle : particles_)
    {
        particle.id = given_id;

        auto sample_x = dist_x(gen);
        auto sample_y = dist_y(gen);
        auto sample_yaw = dist_yaw(gen);

        particle.x = sample_x;
        particle.y = sample_y;
        particle.yaw = sample_yaw;

        particle.weight = 1.0;

        given_id++;
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
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
        particle.x =
            particle.x + ((velocity / yaw_rate) * (sin(particle.yaw + (yaw_rate * delta_t)) - sin(particle.yaw)));
        particle.y =
            particle.y + ((velocity / yaw_rate) * (cos(particle.yaw) - cos(particle.yaw + (yaw_rate * delta_t))));
        particle.yaw = particle.yaw + (yaw_rate * delta_t);

        std::normal_distribution<double> dist_x(particle.x, std_x);
        std::normal_distribution<double> dist_y(particle.y, std_y);
        std::normal_distribution<double> dist_yaw(particle.yaw, std_yaw);

        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.yaw = dist_yaw(gen);
    }
}

void ParticleFilter::dataAssociation(Map predictions, std::vector<LandmarkObs>& observations)
{
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    for (auto& observation : observations)
    {
        auto min_distance = std::numeric_limits<double>::max();
        for (auto& prediction : predictions.landmark_list)
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

void ParticleFilter::updateWeights(double sensor_range,
                                   double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks)
{
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
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

    const auto kPi = 3.14159265358979323846;

    for (auto& particle : particles_)
    {
        // Transform car to map coordinate system
        for (auto& observation : observations)
        {
            auto x = observation.x;
            auto y = observation.y;

            auto yaw = particle.yaw;
            auto x_t = particle.x;
            auto y_t = particle.y;

            observation.x = x * cos(yaw) - y * sin(yaw) + x_t;
            observation.y = x * sin(yaw) + y * cos(yaw) + y_t;
        }

        dataAssociation(map_landmarks, observations);

        auto cov_x = std_landmark[0];
        auto cov_y = std_landmark[1];

        double weight = 1.0;
        for (auto observation : observations)
        {
            auto x_i = observation;

            const int id_to_find = observation.id;
            auto u_i = std::find_if(
                map_landmarks.landmark_list.begin(),
                map_landmarks.landmark_list.end(),
                [id_to_find](const Map::single_landmark_s& item) -> bool { return item.id_i == id_to_find; });

            auto diff_x = x_i.x - u_i->x_f;
            auto diff_y = x_i.y - u_i->y_f;

            auto inverse_factor = 1 / (cov_x * cov_y);

            auto t_x = diff_x * inverse_factor * cov_y;
            auto t_y = diff_y * inverse_factor * cov_x;

            auto inner_term = diff_x * t_x + diff_y * t_y;

            auto term = exp(-0.5 * inner_term);

            auto normalizer = sqrt(2 * kPi * cov_x * cov_y);

            auto w = term / normalizer;
            weight *= w;
        }
        particle.weight = weight;
    }

    // Update weights
    sensor_range;
    std_landmark;
    observations;
    map_landmarks;
}

void ParticleFilter::resample()
{
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
}

void ParticleFilter::write(std::string filename)
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
