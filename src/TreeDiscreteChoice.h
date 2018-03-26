/*-------------------------------------------------------------------------------
 This file is part of Ranger.

 Ranger is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Ranger is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with Ranger. If not, see <http://www.gnu.org/licenses/>.

 Written by:

 Marvin N. Wright
 Institut für Medizinische Biometrie und Statistik
 Universität zu Lübeck
 Ratzeburger Allee 160
 23562 Lübeck

 http://www.imbs-luebeck.de
 wright@imbs.uni-luebeck.de
 #-------------------------------------------------------------------------------*/

#ifndef TREEDISCRETECHOICE_H_
#define TREEDISCRETECHOICE_H_

#include "globals.h"
#include "Tree.h"

class TreeDiscreteChoice: public Tree {
public:
  TreeDiscreteChoice();
  TreeDiscreteChoice(const std::unordered_map<size_t, std::vector<size_t>>& agentID_to_sampleIDs);

  // Create from loaded forest
  TreeDiscreteChoice(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_varIDs,
      std::vector<double>& split_values, std::vector<bool>* is_ordered_variable);

  virtual ~TreeDiscreteChoice();

  void initInternal();

  void appendToFileInternal(std::ofstream& file);

  double getPrediction(size_t sampleID) const {
    size_t terminal_nodeID = prediction_terminal_nodeIDs[sampleID];
    return (split_values[terminal_nodeID]);
  }

  size_t getPredictionTerminalNodeID(size_t sampleID) const {
    return prediction_terminal_nodeIDs[sampleID];
  }

private:
  bool splitNodeInternal(size_t nodeID, std::vector<size_t>& possible_split_varIDs);
  void createEmptyNodeInternal();

  double computePredictionAccuracyInternal();

  // Called by splitNodeInternal(). Sets split_varIDs and split_values.
  bool findBestSplit(size_t nodeID, std::vector<size_t>& possible_split_varIDs);
  void findBestSplitValue(size_t nodeID, size_t varID, size_t num_samples_node,
      double& best_value, size_t& best_varID, double& best_decrease, int split_finder_idx);
  int split_finder(size_t nodeID, size_t varID, size_t num_samples_node);


  double computePredictionMSE();

  void cleanUpInternal() {
    if (counter != 0) {
      delete[] counter;
    }
    if (sums != 0) {
      delete[] sums;
    }
  }

  size_t* counter;
  double* sums;

  void post_bootstrap_init();
  void splitNode_post_process();
  void grow_post_process();

  double dcrf_numItems;
  double dcrf_numAgents;

  void bootstrap();
  void bootstrapWithoutReplacement();

  std::unordered_set<size_t> unique_agentIDs;
  std::vector<size_t> agentIDs; // repeats allowed

  //   agentID -> [sampleIDs] 
  std::unordered_map<size_t, std::vector<size_t>> agentID_to_sampleIDs; 
  // agentID -> sampleID with Y=1
  std::unordered_map<size_t, size_t> agentID_to_choiceID; 
  //   agentID -> # of times it appears in bootstrap sample
  std::unordered_map<size_t, size_t> agentID_to_N;
  // log-lik contribution of each node
  std::vector<double> llik;
  // estimated utility at each node
  std::vector<double> util;
  // parent nodeID -> left [0] or right [1] leaf values
  std::vector<std::vector<double>> child_util;

  void compute_partition_func(std::unordered_map<size_t, double>& agent_Z, const std::vector<double>& curr_util);

  double compute_log_likelihood(const std::unordered_map<size_t, double>& agent_Z, 
                                const std::vector<double>& curr_util,
                                const std::vector<size_t>& agentIDs);

  double compute_temp_log_likelihood(const std::unordered_map<size_t, double>& agent_Z, 
                                     const std::vector<double>& curr_util,
                                     const std::vector<size_t>& agent_ids,
                                     double V_L, double V_R, double V_star,
                                     const std::unordered_set<size_t>& right_sIDs,
                                     std::unordered_map<size_t, size_t>& n_l,
                                     std::unordered_map<size_t, size_t>& n_r,
                                     size_t nodeID) ;

  void compute_full_gradient(std::unordered_map<size_t,double>& leafID_to_partial, 
                             const std::unordered_map<size_t, double>& agent_Z, 
                             const std::vector<double>& curr_util);

  double backtracking(const std::unordered_map<size_t,double>& leafID_to_partial,
                      std::unordered_map<size_t, double>& agent_Z, 
                      std::vector<double>& curr_util, double prev_llik);

  size_t num_splits;
  size_t debug;
  size_t timing;

  DISALLOW_COPY_AND_ASSIGN(TreeDiscreteChoice);
};

#endif /* TREEDISCRETECHOICE_H_ */
