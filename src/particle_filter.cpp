/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cassert>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
   std::random_device rd{};
   std::mt19937 gen{rd()};
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::normal_distribution<double> gaussian_x{0,std[0]};
  std::normal_distribution<double> gaussian_y{0,std[1]};
  std::normal_distribution<double> gaussian_theta{0,std[2]};

  num_particles = 1000;  // TODO: Set the number of particles
  is_initialized = true;
  for(int i =0;i<num_particles;i++)
  {
    Particle particle;
    particle.x = x + gaussian_x(gen);
    particle.y = y + gaussian_y(gen);
    particle.theta = theta + gaussian_theta(gen);
    particles.push_back(particle);
    weights.push_back(1.0);
  }
  
  std::cout<<"Initialized"<<std::endl;
 
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  std::random_device rd{};
   std::mt19937 gen{rd()};
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::normal_distribution<double> gaussian_x{0,std_pos[0]};
  std::normal_distribution<double> gaussian_y{0,std_pos[1]};
  std::normal_distribution<double> gaussian_theta{0,std_pos[2]};
  for(int i = 0;i<num_particles;i++)
  {
    particles[i].x = particles[i].x +(velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta))+ gaussian_x(gen);
    particles[i].y = particles[i].y +(velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t))+ gaussian_y(gen);
    particles[i].theta = particles[i].theta + yaw_rate*delta_t + gaussian_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  //based on the distance metric between each predicted measurement and each and every observations we associate the predicted measurement 
  //to a particular observation landmark id ;
 
  for(int i =0;i<predicted.size();i++)
  {
    double min_dist;
    bool initial = true;
    for(int j =0;j<observations.size();j++)
    {
      double dist = ParticleFilter::dist(predicted[i].x,predicted[i].y,observations[i].x,observations[i].y);
      if(initial)
      {
        min_dist = dist;
        observations[j].id = predicted[i].id;
        initial = false;
      }
      else
      {
        if(dist<=min_dist)
        {
          min_dist = dist;
          observations[j].id = predicted[i].id;
        }
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  // for each particle and for each observation , transform the observation based on the paticle's map coordinates and then 
  // associating the measruement to a particular landmark id based on the sensor range and also based on the nearest nieghbour distance.
  // then calculating the probability associated with this observation measurement and correspoinding map landmark using gaussian dist.
  // then updating the weight of the particle by multiplying all the observations weights probabilities.
  // then updating the assciation vector of particle and also the sense x and sense y.
  
  for(int p = 0;p<particles.size();p++)
  {
    double weight  = 1;
    vector<int> id;
    vector<double> sense_x;
    vector<double> sense_y;
    for(int ob = 0;ob<observations.size();ob++)
    {
      //Linear transformation of the car frame observations.
      LandmarkObs k;
      double min_dist = std::numeric_limits<double>::infinity();
      double map_x = particles[p].x + cos(particles[p].theta)*observations[ob].x - sin(particles[p].theta)*observations[ob].y; 
      double map_y = particles[p].y + sin(particles[p].theta)*observations[ob].x + cos(particles[p].theta)*observations[ob].y;
      for(int ml = 0;ml<map_landmarks.landmark_list.size();ml++)
      {
        LandmarkObs l;
        l.x = map_landmarks.landmark_list[ml].x_f;
        l.y = map_landmarks.landmark_list[ml].y_f;
        l.id = map_landmarks.landmark_list[ml].id_i;
        double dist_obs_map_landmark = ParticleFilter::dist(l.x,l.y,map_x,map_y); // calculating the distance to a map landmark from a transformed observations coordinate
        double dist_particle_map_landmark = ParticleFilter::dist(l.x,l.y,particles[p].x,particles[p].y); // calculating the distance from the particle to the landmark 
        if(dist_obs_map_landmark<=shortest_distance && dist_particle_map_landmark<=sensor_range)
        {
          min_dist = dist_obs_map_landmark;
          k.x = l.x;
          k.y = l.y;
          k.id = l.id;
        }
      }
      sense_x.push_back(k.x);
      sense_y.push_back(k.y);
      id.push_back(k.id);
      weight*=ParticleFilter::multi_gaussian(std_landmark[0],std_landmark[1],map_x,map_y,
                               k.x,
                               k.y);
    }
    
    particles[p].weight = weight;
    weights[p] = weight;
    ParticleFilter::SetAssociations(particles[p],id,sense_x,sense_y);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(weights.begin(),weights.end()); // defining a discrete distribution of weights where the probability of picking is proportional to the weights given
  
  int num_samples = 1000;
  vector<Particle> resampled_particles;
  for(int i =0;i<num_samples;i++)
  {
    resampled_particles.push_back(particles[d(gen)]);
  }
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}