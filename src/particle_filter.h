/**
 * particle_filter.h
 * 2D particle filter class.
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include <string>
#include <vector>
#include "helper_functions.h"

struct Particle {
  int id;
  double x;
  double y;
  double theta;
  double weight;
  std::vector<int> associations;
  std::vector<double> sense_x;
  std::vector<double> sense_y;
};


class ParticleFilter {  
 public:
  // Constructor
  // @param num_particles Number of particles
  ParticleFilter() : num_particles(0), is_initialized(false) {}

  // Destructor
  ~ParticleFilter() {}

  /**
   * init Initializes particle filter by initializing particles to Gaussian
   *   distribution around first position and all the weights to 1.
   * @param x Initial x position [m] (simulated estimate from GPS)
   * @param y Initial y position [m]
   * @param theta Initial orientation [rad]
   * @param std[] Array of dimension 3 [standard deviation of x [m], 
   *   standard deviation of y [m], standard deviation of yaw [rad]]
   */
  void init(double x, double y, double theta, double std[]);

  /**
   * prediction Predicts the state for the next time step
   *   using the process model.
   * @param delta_t Time between time step t and t+1 in measurements [s]
   * @param std_pos[] Array of dimension 3 [standard deviation of x [m], 
   *   standard deviation of y [m], standard deviation of yaw [rad]]
   * @param velocity Velocity of car from t to t+1 [m/s]
   * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
   */
  void prediction(double delta_t, double std_pos[], double velocity, 
                  double yaw_rate);
  
  /**
   * dataAssociation Finds which observations correspond to which landmarks 
   *   (likely by using a nearest-neighbors data association).
   * @param predicted Vector of predicted landmark observations
   * @param observations Vector of landmark observations
   */
  void dataAssociation(std::vector<LandmarkObs> predicted, 
                       std::vector<LandmarkObs>& observations);
  
  /**
   * updateWeights Updates the weights for each particle based on the likelihood
   *   of the observed measurements. 
   * @param sensor_range Range [m] of sensor
   * @param std_landmark[] Array of dimension 2
   *   [Landmark measurement uncertainty [x [m], y [m]]]
   * @param observations Vector of landmark observations
   * @param map Map class containing map landmarks
   */
  void updateWeights(double sensor_range, double std_landmark[], 
                     const std::vector<LandmarkObs> &observations,
                     const Map &map_landmarks);
  
  /**
   * resample Resamples from the updated set of particles to form
   *   the new set of particles.
   */
  void resample();

  /**
   * Set a particles list of associations, along with the associations'
   *   calculated world x,y coordinates
   * This can be a very useful debugging tool to make sure transformations 
   *   are correct and assocations correctly connected
   */
  void SetAssociations(Particle& particle, const std::vector<int>& associations,
                       const std::vector<double>& sense_x, 
                       const std::vector<double>& sense_y);

  /**
   * initialized Returns whether particle filter is initialized yet or not.
   */
  const bool initialized() const {
    return is_initialized;
  }
  
  double dist(double x1, double y1, double x2, double y2) {
  return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
	}
  double multi_gaussian(double std_x,double std_y,double x1,double y1,double mu1,double mu2)
  {
    double varx = std_x*std_x;
    double vary = std_y*std_y;
    double numer_x = (x1 - mu1)*(x1 - mu1);
    double numer_y = (y1 - mu2)*(y1 - mu2);
    double multiplier = 1/(2*M_PI*std_x*std_y);
    double probability = ((numer_x)/(2 * varx)) + ((numer_y)/(2 * vary));
    double exponent = exp(-probability);
    return multiplier*exponent;
  }

  /**
   * Used for obtaining debugging information related to particles.
   */
  std::string getAssociations(Particle best);
  std::string getSenseCoord(Particle best, std::string coord);

  // Set of current particles
  std::vector<Particle> particles;

 private:
  // Number of particles to draw
  int num_particles; 
  
  // Flag, if filter is initialized
  bool is_initialized;
  
  // Vector of weights of all particles
  std::vector<double> weights; 
};

#endif  // PARTICLE_FILTER_H_