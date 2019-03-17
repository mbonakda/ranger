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

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

#include <ctime>

#include "utility.h"
#include "TreeRegression.h"
#include "Data.h"

TreeRegression::TreeRegression() :
    counter(0), sums(0) {
}

TreeRegression::TreeRegression(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_varIDs,
    std::vector<double>& split_values, std::vector<bool>* is_ordered_variable) :
    Tree(child_nodeIDs, split_varIDs, split_values, is_ordered_variable), counter(0), sums(0) {
}

TreeRegression::~TreeRegression() {
  // Empty on purpose
}

void TreeRegression::initInternal() {
  // Init counters if not in memory efficient mode
  if (!memory_saving_splitting) {
    size_t max_num_splits = data->getMaxNumUniqueValues();

    // Use number of random splits for extratrees
    if (splitrule == EXTRATREES && num_random_splits > max_num_splits) {
      max_num_splits = num_random_splits;
    }

    counter = new size_t[max_num_splits];
    sums = new double[max_num_splits];
  }
}

double TreeRegression::estimate(size_t nodeID) {

// Mean of responses of samples in node
  double sum_responses_in_node = 0;
  size_t num_samples_in_node = sampleIDs[nodeID].size();
  for (size_t i = 0; i < sampleIDs[nodeID].size(); ++i) {
    sum_responses_in_node += data->get(sampleIDs[nodeID][i], dependent_varID);
  }
  return (sum_responses_in_node / (double) num_samples_in_node);
}

void TreeRegression::appendToFileInternal(std::ofstream& file) {
// Empty on purpose
}

bool TreeRegression::splitNodeInternal(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {

// Check node size, stop if maximum reached
  if (sampleIDs[nodeID].size() <= min_node_size) {
    split_values[nodeID] = estimate(nodeID);
    return true;
  }

  // Check if node is pure and set split_value to estimate and stop if pure
  bool pure = true;
  double pure_value = 0;
  for (size_t i = 0; i < sampleIDs[nodeID].size(); ++i) {
    double value = data->get(sampleIDs[nodeID][i], dependent_varID);
    if (i != 0 && value != pure_value) {
      pure = false;
      break;
    }
    pure_value = value;
  }
  if (pure) {
    split_values[nodeID] = pure_value;
    return true;
  }

  // Find best split, stop if no decrease of impurity
  bool stop;
  if (splitrule == MAXSTAT) {
    stop = findBestSplitMaxstat(nodeID, possible_split_varIDs);
  } else if (splitrule == EXTRATREES) {
    stop = findBestSplitExtraTrees(nodeID, possible_split_varIDs);
  } else {
    stop = findBestSplit(nodeID, possible_split_varIDs);
  }

  if (stop) {
    split_values[nodeID] = estimate(nodeID);
    return true;
  }

  return false;
}

void TreeRegression::createEmptyNodeInternal() {
// Empty on purpose
}

double TreeRegression::computePredictionAccuracyInternal() {

  size_t num_predictions = prediction_terminal_nodeIDs.size();
  double sum_of_squares = 0;
  for (size_t i = 0; i < num_predictions; ++i) {
    size_t terminal_nodeID = prediction_terminal_nodeIDs[i];
    double predicted_value = split_values[terminal_nodeID];
    double real_value = data->get(oob_sampleIDs[i], dependent_varID);
    if (predicted_value != real_value) {
      sum_of_squares += (predicted_value - real_value) * (predicted_value - real_value);
    }
  }
  return (1.0 - sum_of_squares / (double) num_predictions);
}

bool TreeRegression::findBestSplit(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {

  size_t num_samples_node = sampleIDs[nodeID].size();
  double best_decrease = -1;
  size_t best_varID = 0;
  double best_value = 0;

  // Compute sum of responses in node
  double sum_node = 0;
  for (auto& sampleID : sampleIDs[nodeID]) {
    sum_node += data->get(sampleID, dependent_varID);
  }

  // For all possible split variables
  for (auto& varID : possible_split_varIDs) {

    // Find best split value, if ordered consider all values as split values, else all 2-partitions
    if ((*is_ordered_variable)[varID]) {

      // Use memory saving method if option set
      if (memory_saving_splitting) {
        findBestSplitValueSmallQ(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease);
      } else {
        // Use faster method for both cases
        double q = (double) num_samples_node / (double) data->getNumUniqueDataValues(varID);
        if (q < Q_THRESHOLD) {
          findBestSplitValueSmallQ(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease);
        } else {
          findBestSplitValueLargeQ(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease);
        }
      }
    } else {
      findBestSplitValueUnordered(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease);
    }
  }

// Stop if no good split found
  if (best_decrease < 0) {
    return true;
  }

// Save best values
  split_varIDs[nodeID] = best_varID;
  split_values[nodeID] = best_value;

// Compute decrease of impurity for this node and add to variable importance if needed
  if (importance_mode == IMP_GINI) {
    addImpurityImportance(nodeID, best_varID, best_decrease);
  }
  return false;
}

void TreeRegression::findBestSplitValueSmallQ(size_t nodeID, size_t varID, double sum_node, size_t num_samples_node,
    double& best_value, size_t& best_varID, double& best_decrease) {

  // Create possible split values
  std::vector<double> possible_split_values;
  data->getAllValues(possible_split_values, sampleIDs[nodeID], varID);

  // Try next variable if all equal for this
  if (possible_split_values.size() < 2) {
    return;
  }

  // Initialize with 0 if not in memory efficient mode, use pre-allocated space
  // -1 because no split possible at largest value
  size_t num_splits = possible_split_values.size() - 1;
  double* sums_right;
  size_t* n_right;
  if (memory_saving_splitting) {
    sums_right = new double[num_splits]();
    n_right = new size_t[num_splits]();
  } else {
    sums_right = sums;
    n_right = counter;
    std::fill(sums_right, sums_right + num_splits, 0);
    std::fill(n_right, n_right + num_splits, 0);
  }

  // Sum in right child and possbile split
  for (auto& sampleID : sampleIDs[nodeID]) {
    double value = data->get(sampleID, varID);
    double response = data->get(sampleID, dependent_varID);

    // Count samples until split_value reached
    for (size_t i = 0; i < num_splits; ++i) {
      if (value > possible_split_values[i]) {
        ++n_right[i];
        sums_right[i] += response;
      } else {
        break;
      }
    }
  }

  // Compute decrease of impurity for each possible split
  for (size_t i = 0; i < num_splits; ++i) {

    // Stop if one child empty
    size_t n_left = num_samples_node - n_right[i];
    if (n_left == 0 || n_right[i] == 0) {
      continue;
    }

    double sum_right = sums_right[i];
    double sum_left = sum_node - sum_right;
    double decrease = sum_left * sum_left / (double) n_left + sum_right * sum_right / (double) n_right[i];

    // If better than before, use this
    if (decrease > best_decrease) {
      best_value = (possible_split_values[i] + possible_split_values[i + 1]) / 2;
      best_varID = varID;
      best_decrease = decrease;
    }
  }

  if (memory_saving_splitting) {
    delete[] sums_right;
    delete[] n_right;
  }
}

void TreeRegression::findBestSplitValueLargeQ(size_t nodeID, size_t varID, double sum_node, size_t num_samples_node,
    double& best_value, size_t& best_varID, double& best_decrease) {

  // Set counters to 0
  size_t num_unique = data->getNumUniqueDataValues(varID);
  std::fill(counter, counter + num_unique, 0);
  std::fill(sums, sums + num_unique, 0);

  for (auto& sampleID : sampleIDs[nodeID]) {
    size_t index = data->getIndex(sampleID, varID);

    sums[index] += data->get(sampleID, dependent_varID);
    ++counter[index];
  }

  size_t n_left = 0;
  double sum_left = 0;

  // Compute decrease of impurity for each split
  for (size_t i = 0; i < num_unique - 1; ++i) {

    // Stop if nothing here
    if (counter[i] == 0) {
      continue;
    }

    n_left += counter[i];
    sum_left += sums[i];

    // Stop if right child empty
    size_t n_right = num_samples_node - n_left;
    if (n_right == 0) {
      break;
    }

    double sum_right = sum_node - sum_left;
    double decrease = sum_left * sum_left / (double) n_left + sum_right * sum_right / (double) n_right;

    // If better than before, use this
    if (decrease > best_decrease) {
      // Find next value in this node
      size_t j = i + 1;
      while(j < num_unique && counter[j] == 0) {
        ++j;
      }

      // Use mid-point split
      best_value = (data->getUniqueDataValue(varID, i) + data->getUniqueDataValue(varID, j)) / 2;
      best_varID = varID;
      best_decrease = decrease;
    }
  }
}

void TreeRegression::findBestSplitValueUnordered(size_t nodeID, size_t varID, double sum_node, size_t num_samples_node,
    double& best_value, size_t& best_varID, double& best_decrease) {

// Create possible split values
  std::vector<double> factor_levels;
  data->getAllValues(factor_levels, sampleIDs[nodeID], varID);

// Try next variable if all equal for this
  if (factor_levels.size() < 2) {
    return;
  }

// Number of possible splits is 2^num_levels
  size_t num_splits = (1 << factor_levels.size());

// Compute decrease of impurity for each possible split
// Split where all left (0) or all right (1) are excluded
// The second half of numbers is just left/right switched the first half -> Exclude second half
  for (size_t local_splitID = 1; local_splitID < num_splits / 2; ++local_splitID) {

    // Compute overall splitID by shifting local factorIDs to global positions
    size_t splitID = 0;
    for (size_t j = 0; j < factor_levels.size(); ++j) {
      if ((local_splitID & (1 << j))) {
        double level = factor_levels[j];
        size_t factorID = floor(level) - 1;
        splitID = splitID | (1 << factorID);
      }
    }

    // Initialize
    double sum_right = 0;
    size_t n_right = 0;

    // Sum in right child
    for (auto& sampleID : sampleIDs[nodeID]) {
      double response = data->get(sampleID, dependent_varID);
      double value = data->get(sampleID, varID);
      size_t factorID = floor(value) - 1;

      // If in right child, count
      // In right child, if bitwise splitID at position factorID is 1
      if ((splitID & (1 << factorID))) {
        ++n_right;
        sum_right += response;
      }
    }
    size_t n_left = num_samples_node - n_right;

    // Sum of squares
    double sum_left = sum_node - sum_right;
    double decrease = sum_left * sum_left / (double) n_left + sum_right * sum_right / (double) n_right;

    // If better than before, use this
    if (decrease > best_decrease) {
      best_value = splitID;
      best_varID = varID;
      best_decrease = decrease;
    }
  }
}

bool TreeRegression::findBestSplitMaxstat(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {

  size_t num_samples_node = sampleIDs[nodeID].size();

  // Compute ranks
  std::vector<double> response;
  response.reserve(num_samples_node);
  for (auto& sampleID : sampleIDs[nodeID]) {
    response.push_back(data->get(sampleID, dependent_varID));
  }
  std::vector<double> ranks = rank(response);

  // Save split stats
  std::vector<double> pvalues;
  pvalues.reserve(possible_split_varIDs.size());
  std::vector<double> values;
  values.reserve(possible_split_varIDs.size());
  std::vector<double> candidate_varIDs;
  candidate_varIDs.reserve(possible_split_varIDs.size());

  // Compute p-values
  for (auto& varID : possible_split_varIDs) {

    // Get all observations
    std::vector<double> x;
    x.reserve(num_samples_node);
    for (auto& sampleID : sampleIDs[nodeID]) {
      x.push_back(data->get(sampleID, varID));
    }

    // Order by x
    std::vector<size_t> indices = order(x, false);
    //std::vector<size_t> indices = orderInData(data, sampleIDs[nodeID], varID, false);

    // Compute maximally selected rank statistics
    double best_maxstat;
    double best_split_value;
    maxstat(ranks, x, indices, best_maxstat, best_split_value, minprop, 1 - minprop);
    //maxstatInData(scores, data, sampleIDs[nodeID], varID, indices, best_maxstat, best_split_value, minprop, 1 - minprop);

    if (best_maxstat > -1) {
      // Compute number of samples left of cutpoints
      std::vector<size_t> num_samples_left = numSamplesLeftOfCutpoint(x, indices);
      //std::vector<size_t> num_samples_left = numSamplesLeftOfCutpointInData(data, sampleIDs[nodeID], varID, indices);

      // Compute p-values
      double pvalue_lau92 = maxstatPValueLau92(best_maxstat, minprop, 1 - minprop);
      double pvalue_lau94 = maxstatPValueLau94(best_maxstat, minprop, 1 - minprop, num_samples_node, num_samples_left);

      // Use minimum of Lau92 and Lau94
      double pvalue = std::min(pvalue_lau92, pvalue_lau94);

      // Save split stats
      pvalues.push_back(pvalue);
      values.push_back(best_split_value);
      candidate_varIDs.push_back(varID);
    }
  }

  double adjusted_best_pvalue = std::numeric_limits<double>::max();
  size_t best_varID = 0;
  double best_value = 0;

  if (pvalues.size() > 0) {
    // Adjust p-values with Benjamini/Hochberg
    std::vector<double> adjusted_pvalues = adjustPvalues(pvalues);

    // Use smallest p-value
    double min_pvalue = std::numeric_limits<double>::max();
    for (size_t i = 0; i < pvalues.size(); ++i) {
      if (pvalues[i] < min_pvalue) {
        min_pvalue = pvalues[i];
        best_varID = candidate_varIDs[i];
        best_value = values[i];
        adjusted_best_pvalue = adjusted_pvalues[i];
      }
    }
  }

  // Stop if no good split found (this is terminal node).
  if (adjusted_best_pvalue > alpha) {
    return true;
  } else {
    // If not terminal node save best values
    split_varIDs[nodeID] = best_varID;
    split_values[nodeID] = best_value;
    return false;
  }
}

bool TreeRegression::findBestSplitExtraTrees(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {

  size_t num_samples_node = sampleIDs[nodeID].size();
  double best_decrease = -1;
  size_t best_varID = 0;
  double best_value = 0;

  // Compute sum of responses in node
  double sum_node = 0;
  for (auto& sampleID : sampleIDs[nodeID]) {
    sum_node += data->get(sampleID, dependent_varID);
  }

  // For all possible split variables
  for (auto& varID : possible_split_varIDs) {

    // Find best split value, if ordered consider all values as split values, else all 2-partitions
    if ((*is_ordered_variable)[varID]) {
      findBestSplitValueExtraTrees(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease);
    } else {
      findBestSplitValueExtraTreesUnordered(nodeID, varID, sum_node, num_samples_node, best_value, best_varID,
          best_decrease);
    }
  }

  // Stop if no good split found
  if (best_decrease < 0) {
    return true;
  }

  // Save best values
  split_varIDs[nodeID] = best_varID;
  split_values[nodeID] = best_value;

  // Compute decrease of impurity for this node and add to variable importance if needed
  if (importance_mode == IMP_GINI) {
    addImpurityImportance(nodeID, best_varID, best_decrease);
  }
  return false;
}

void TreeRegression::findBestSplitValueExtraTrees(size_t nodeID, size_t varID, double sum_node, size_t num_samples_node,
    double& best_value, size_t& best_varID, double& best_decrease) {

  // Get min/max values of covariate in node
  double min;
  double max;
  data->getMinMaxValues(min, max, sampleIDs[nodeID], varID);

  // Try next variable if all equal for this
  if (min == max) {
    return;
  }

  // Create possible split values: Draw randomly between min and max
  std::vector<double> possible_split_values;
  std::uniform_real_distribution<double> udist(min, max);
  possible_split_values.reserve(num_random_splits);
  for (size_t i = 0; i < num_random_splits; ++i) {
    possible_split_values.push_back(udist(random_number_generator));
  }

  // Initialize with 0m if not in memory efficient mode, use pre-allocated space
  size_t num_splits = possible_split_values.size();
  double* sums_right;
  size_t* n_right;
  if (memory_saving_splitting) {
    sums_right = new double[num_splits]();
    n_right = new size_t[num_splits]();
  } else {
    sums_right = sums;
    n_right = counter;
    std::fill(sums_right, sums_right + num_splits, 0);
    std::fill(n_right, n_right + num_splits, 0);
  }

  // Sum in right child and possbile split
  for (auto& sampleID : sampleIDs[nodeID]) {
    double value = data->get(sampleID, varID);
    double response = data->get(sampleID, dependent_varID);

    // Count samples until split_value reached
    for (size_t i = 0; i < num_splits; ++i) {
      if (value > possible_split_values[i]) {
        ++n_right[i];
        sums_right[i] += response;
      } else {
        break;
      }
    }
  }

  // Compute decrease of impurity for each possible split
  for (size_t i = 0; i < num_splits; ++i) {

    // Stop if one child empty
    size_t n_left = num_samples_node - n_right[i];
    if (n_left == 0 || n_right[i] == 0) {
      continue;
    }

    double sum_right = sums_right[i];
    double sum_left = sum_node - sum_right;
    double decrease = sum_left * sum_left / (double) n_left + sum_right * sum_right / (double) n_right[i];

    // If better than before, use this
    if (decrease > best_decrease) {
      best_value = possible_split_values[i];
      best_varID = varID;
      best_decrease = decrease;
    }
  }

  if (memory_saving_splitting) {
    delete[] sums_right;
    delete[] n_right;
  }
}

void TreeRegression::findBestSplitValueExtraTreesUnordered(size_t nodeID, size_t varID, double sum_node,
    size_t num_samples_node, double& best_value, size_t& best_varID, double& best_decrease) {

  size_t num_unique_values = data->getNumUniqueDataValues(varID);

  // Get all factor indices in node
  std::vector<bool> factor_in_node(num_unique_values, false);
  for (auto& sampleID : sampleIDs[nodeID]) {
    size_t index = data->getIndex(sampleID, varID);
    factor_in_node[index] = true;
  }

  // Vector of indices in and out of node
  std::vector<size_t> indices_in_node;
  std::vector<size_t> indices_out_node;
  indices_in_node.reserve(num_unique_values);
  indices_out_node.reserve(num_unique_values);
  for (size_t i = 0; i < num_unique_values; ++i) {
    if (factor_in_node[i]) {
      indices_in_node.push_back(i);
    } else {
      indices_out_node.push_back(i);
    }
  }

  // Generate num_random_splits splits
  for (size_t i = 0; i < num_random_splits; ++i) {
    std::vector<size_t> split_subset;
    split_subset.reserve(num_unique_values);

    // Draw random subsets, sample all partitions with equal probability
    if (indices_in_node.size() > 1) {
      size_t num_partitions = (2 << (indices_in_node.size() - 1)) - 2; // 2^n-2 (don't allow full or empty)
      std::uniform_int_distribution<size_t> udist(1, num_partitions);
      size_t splitID_in_node = udist(random_number_generator);
      for (size_t j = 0; j < indices_in_node.size(); ++j) {
        if ((splitID_in_node & (1 << j)) > 0) {
          split_subset.push_back(indices_in_node[j]);
        }
      }
    }
    if (indices_out_node.size() > 1) {
      size_t num_partitions = (2 << (indices_out_node.size() - 1)) - 1; // 2^n-1 (allow full or empty)
      std::uniform_int_distribution<size_t> udist(0, num_partitions);
      size_t splitID_out_node = udist(random_number_generator);
      for (size_t j = 0; j < indices_out_node.size(); ++j) {
        if ((splitID_out_node & (1 << j)) > 0) {
          split_subset.push_back(indices_out_node[j]);
        }
      }
    }

    // Assign union of the two subsets to right child
    size_t splitID = 0;
    for (auto& idx : split_subset) {
      splitID |= 1 << idx;
    }

    // Initialize
    double sum_right = 0;
    size_t n_right = 0;

    // Sum in right child
    for (auto& sampleID : sampleIDs[nodeID]) {
      double response = data->get(sampleID, dependent_varID);
      double value = data->get(sampleID, varID);
      size_t factorID = floor(value) - 1;

      // If in right child, count
      // In right child, if bitwise splitID at position factorID is 1
      if ((splitID & (1 << factorID))) {
        ++n_right;
        sum_right += response;
      }
    }
    size_t n_left = num_samples_node - n_right;

    // Sum of squares
    double sum_left = sum_node - sum_right;
    double decrease = sum_left * sum_left / (double) n_left + sum_right * sum_right / (double) n_right;

    // If better than before, use this
    if (decrease > best_decrease) {
      best_value = splitID;
      best_varID = varID;
      best_decrease = decrease;
    }
  }
}

void TreeRegression::addImpurityImportance(size_t nodeID, size_t varID, double decrease) {

  double sum_node = 0;
  for (auto& sampleID : sampleIDs[nodeID]) {
    sum_node += data->get(sampleID, dependent_varID);
  }
  double best_decrease = decrease - sum_node * sum_node / (double) sampleIDs[nodeID].size();

// No variable importance for no split variables
  size_t tempvarID = varID;
  for (auto& skip : *no_split_variables) {
    if (varID >= skip) {
      --tempvarID;
    }
  }
  (*variable_importance)[tempvarID] += best_decrease;
}

void TreeRegression::reshape() {
  // 1. level-order traversal of shape-constrained nodes
  std::vector<size_t> sc_nodes;
  std::queue<size_t>  q;
  q.push(0);
  while(q.empty() == false) {

    size_t curr_node    = q.front();
    size_t left_nodeID  = child_nodeIDs[0][curr_node]; 
    size_t right_nodeID = child_nodeIDs[1][curr_node]; 
    q.pop();

    // only add nodes with children
    auto sc_itr = sc_variable_IDs.find(split_varIDs[curr_node]);
    if(sc_itr != sc_variable_IDs.end() && left_nodeID != 0 && right_nodeID != 0) {
      sc_nodes.push_back(curr_node);
    }

    if(left_nodeID != 0) {
      q.push(left_nodeID);
    }
    if(right_nodeID != 0) {
      q.push(right_nodeID);
    }
  }

  std::reverse(sc_nodes.begin(), sc_nodes.end()); // bottom-up ordering
  num_sc_nodes = sc_nodes.size();

  // shape-constrained node id -> vector of leaf (node id, value) pairs
  optmap node_to_left;
  optmap node_to_right;

  // 2. for each optimization node, need vector of left/right leaf IDs and values
  over_num_constraints = 0;
  for (auto& nn : sc_nodes) {
    std::vector<std::pair<size_t, double>> left  = get_leaves(child_nodeIDs[0][nn], node_to_left, node_to_right);
    node_to_left[nn] = left;
    std::vector<std::pair<size_t, double>> right = get_leaves(child_nodeIDs[1][nn], node_to_left, node_to_right);
    node_to_right[nn] = right;
    over_num_constraints += left.size() * right.size();
  }

  /*
  // 3. perform exact optimization 
  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<std::pair<size_t, size_t>> intersections;
  for (auto& nn : sc_nodes) {
    std::vector<std::pair<size_t, size_t>>   tmp_int = find_intersections( nn, node_to_left[nn], node_to_right[nn], dim_intervals );
    intersections.insert(intersections.end(), tmp_int.begin(), tmp_int.end() );
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  time_goldiInt = std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count();

  goldi_num_constraints = intersections.size();
  //std::cout << "num goldilocks constraints: " << goldi_num_constraints << std::endl;

  std::set<size_t> leaf_ids;
  for (auto& nn : sc_nodes) {
    for(auto& ii: node_to_left[nn]) {
      leaf_ids.insert(ii.first);
    }
    for(auto& ii: node_to_right[nn]) {
      leaf_ids.insert(ii.first);
    }
  }

  goldilocks_opt(leaf_ids, intersections);
  */

  // 3. perform bottom-up over-constrained optimization
  for (auto& nn : sc_nodes) {
    over_constr_opt(nn, node_to_left[nn], node_to_right[nn]);
  }


  /*
  // 3. perform under-constrained optimization
  t1 = std::chrono::high_resolution_clock::now();
  under_constr_opt(intersections, dim_intervals);
  t2 = std::chrono::high_resolution_clock::now();
  time_underInt = std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count();
  */

  /*
  //std::cout << "total shape-constrained leaves," << leaf_ids.size() << std::endl;

  std::cout << leaf_ids.size() << " final leaves";
  for( auto it = leaf_ids.begin(); it != leaf_ids.end(); ++it ) {
  std::cout << "," << *it;
  }	
  std::cout << std::endl;
  */

}

void TreeRegression::goldilocks_opt(const std::set<size_t> & leaves, const std::vector<std::pair<size_t, size_t>> & id_edges) {

  double DIVIDE_MULT = 1e3;

  std::unordered_map<size_t, size_t> id_to_idx; 
  auto v = new_array_ptr<double,1>(leaves.size());

  size_t idx = 0;
  for( auto l : leaves ) {
    (*v)[idx]    = split_values[l]/DIVIDE_MULT;
    id_to_idx[l] = idx;
    idx++;
  }

  std::vector<std::pair<size_t, size_t>> idx_edges;
  for( auto e : id_edges ) {
    //std::cout << "constraint," << e.first << "," << e.second << std::endl;
    idx_edges.push_back(std::pair<size_t,size_t>(id_to_idx[e.first], id_to_idx[e.second]));
  }


  size_t num_vars = v->size() + 2; // 2 dummy variables
  auto c          = new_array_ptr<double,1>(num_vars);
  (*c)[0] = 0;
  (*c)[1] = 1;
  for( size_t ii = 2; ii < c->size(); ++ii ) {
      (*c)[ii] = -1*(*v)[ii-2];
  }

  size_t num_constraints = idx_edges.size();

  auto rows   = new_array_ptr<int,1>(2*num_constraints);
  for(size_t ii = 0; ii < rows->size(); ++ii) {
    (*rows)[ii] = int(ii/2); // each row appears twice for each constraint
  }

  auto cols   = new_array_ptr<int,1>(2*num_constraints);
  auto values = new_array_ptr<double,1>(2*num_constraints);
  for( int ii = 0; ii < idx_edges.size(); ++ii ) {
    (*cols)[(2*ii)]         = idx_edges[ii].first + 2;
    (*values)[2*ii]         = -1;
    (*cols)[(2*ii)+1]       = idx_edges[ii].second + 2;
    (*values)[(2*ii)+1]     = 1;
  }

  auto A = Matrix::sparse(num_constraints, num_vars, rows, cols, values);


  Model::t M     = new Model("rrf"); auto _M = finally([&]() { M->dispose(); });
  //M->setLogHandler([=](const std::string & msg) { std::cout << msg << std::flush; } );


  Variable::t x0  = M->variable("x0", 1, Domain::equalsTo(1.));
  Variable::t x1  = M->variable("x1", 1, Domain::greaterThan(0.));
  Variable::t x2  = M->variable("x2", num_vars-2, Domain::unbounded());

  Variable::t z1 = Var::vstack(x0, x1, x2);

  Constraint::t qc = M->constraint("qc", z1, Domain::inRotatedQCone());
  M->constraint("mono", Expr::mul(A,z1),Domain::greaterThan(0.));

  M->objective("obj", ObjectiveSense::Minimize, Expr::dot(c,z1));
  try {
    auto t1 = std::chrono::high_resolution_clock::now();
    M->solve();
    auto t2 = std::chrono::high_resolution_clock::now();

    /*
    std::cout << "mosek solve took "
      << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count()
      << " seconds\n";
      */

    //std::cout << "mosek status = " << M->getPrimalSolutionStatus() << std::endl;

    ndarray<double,1> xlvl   = *(x2->level());
    for( auto p : id_to_idx ) {
      double new_val = xlvl[p.second]*DIVIDE_MULT; 
      double old_val = split_values[p.first];
      //std::cout << "values," << split_values[p.first] << "," << xlvl[p.second]*1e6 << "," << fabs(new_val - old_val) <<  std::endl;
      split_values[p.first] = new_val;
    }

  }  catch(const FusionException &e) {
    std::cout << "caught an exception" << std::endl;
  }

}

