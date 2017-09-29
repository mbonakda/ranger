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

  // Create from loaded forest
  TreeDiscreteChoice(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_varIDs,
      std::vector<double>& split_values, std::vector<bool>* is_ordered_variable);

  virtual ~TreeDiscreteChoice();

  void initInternal();

  double estimate(size_t nodeID);
  void computePermutationImportanceInternal(std::vector<std::vector<size_t>>* permutations);
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
  void reshape();
  void goldilocks_opt(const std::set<size_t> &leaves, const std::vector<std::pair<size_t, size_t>> &edges);

  double computePredictionAccuracyInternal();

  // Called by splitNodeInternal(). Sets split_varIDs and split_values.
  bool findBestSplit(size_t nodeID, std::vector<size_t>& possible_split_varIDs);
  void findBestSplitValue(size_t nodeID, size_t varID, double sum_node, size_t num_samples_node,
      double& best_value, size_t& best_varID, double& best_decrease);

  void addImpurityImportance(size_t nodeID, size_t varID, double decrease);

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

  double dcrf_numItems;
  double dcrf_numAgents;

  std::unordered_set<size_t> agentIDs;
  std::unordered_set<size_t> itemIDs;

  //   agentID -> [sampleIDs] 
  std::unordered_map<size_t, std::vector<size_t>> agentID_to_sampleIDs; 
  //   sampleID -> agentID 
  std::unordered_map<size_t, size_t> sampleID_to_agentIDs; 
  // log-lik contribution of each node
  std::vector<double> llik;
  // estimated utility at each node
  std::vector<double> util;
  // agentID -> all leafIDs that currently have a sample from that agent
  std::unordered_map<size_t, std::vector<size_t>> agentID_to_leafIDs; 
  // sampleID -> leafID
  std::unordered_map<size_t, size_t> sampleID_to_leafID; 

  DISALLOW_COPY_AND_ASSIGN(TreeDiscreteChoice);
};

#endif /* TREEDISCRETECHOICE_H_ */
