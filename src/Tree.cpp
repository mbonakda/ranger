/*-------------------------------------------------------------------------------
 * std::cout << "number of overconstrained intersections = " << num_over_constr << std::endl;
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

#include <iterator>

#include "Tree.h"
#include "utility.h"


Tree::Tree() :
    dependent_varID(0), mtry(0), num_samples(0), num_samples_oob(0), is_ordered_variable(0), no_split_variables(0), min_node_size(
        0), deterministic_varIDs(0), split_select_varIDs(0), split_select_weights(0), case_weights(0), oob_sampleIDs(0), holdout(
        false), keep_inbag(false), data(0), variable_importance(0), importance_mode(DEFAULT_IMPORTANCE_MODE), sample_with_replacement(
        true), sample_fraction(1), memory_saving_splitting(false), splitrule(DEFAULT_SPLITRULE), alpha(DEFAULT_ALPHA), minprop(
        DEFAULT_MINPROP), num_random_splits(DEFAULT_NUM_RANDOM_SPLITS), lowest_sc_depth(-1) {
}

Tree::Tree(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_varIDs,
    std::vector<double>& split_values, std::vector<bool>* is_ordered_variable) :
    dependent_varID(0), mtry(0), num_samples(0), num_samples_oob(0), is_ordered_variable(is_ordered_variable), no_split_variables(
        0), min_node_size(0), deterministic_varIDs(0), split_select_varIDs(0), split_select_weights(0), case_weights(0), split_varIDs(
        split_varIDs), split_values(split_values), child_nodeIDs(child_nodeIDs), oob_sampleIDs(0), holdout(false), keep_inbag(
        false), data(0), variable_importance(0), importance_mode(DEFAULT_IMPORTANCE_MODE), sample_with_replacement(
        true), sample_fraction(1), memory_saving_splitting(false), splitrule(DEFAULT_SPLITRULE), alpha(DEFAULT_ALPHA), minprop(
        DEFAULT_MINPROP), num_random_splits(DEFAULT_NUM_RANDOM_SPLITS) {
}

Tree::~Tree() {
}

void Tree::init(Data* data, uint mtry, size_t dependent_varID, size_t num_samples, uint seed,
    std::vector<size_t>* deterministic_varIDs, std::vector<size_t>* split_select_varIDs,
    std::vector<double>* split_select_weights, ImportanceMode importance_mode, uint min_node_size,
    std::vector<size_t>* no_split_variables, bool sample_with_replacement, std::vector<bool>* is_unordered,
    bool memory_saving_splitting, SplitRule splitrule, std::vector<double>* case_weights, bool keep_inbag,
    double sample_fraction, double alpha, double minprop, bool holdout, uint num_random_splits,
    std::vector<size_t>* sc_variable_IDs, int maxTreeHeight, bool speedy) {

  this->data = data;
  this->mtry = mtry;
  this->dependent_varID = dependent_varID;
  this->num_samples = num_samples;
  this->memory_saving_splitting = memory_saving_splitting;

  // Create root node, assign bootstrap sample and oob samples
  child_nodeIDs.push_back(std::vector<size_t>());
  child_nodeIDs.push_back(std::vector<size_t>());
  createEmptyNode();

  // Initialize random number generator and set seed
  random_number_generator.seed(seed);

  this->deterministic_varIDs = deterministic_varIDs;
  this->split_select_varIDs = split_select_varIDs;
  this->split_select_weights = split_select_weights;
  this->importance_mode = importance_mode;
  this->min_node_size = min_node_size;
  this->no_split_variables = no_split_variables;
  this->is_ordered_variable = is_unordered;
  this->sample_with_replacement = sample_with_replacement;
  this->splitrule = splitrule;
  this->case_weights = case_weights;
  this->keep_inbag = keep_inbag;
  this->sample_fraction = sample_fraction;
  this->holdout = holdout;
  this->alpha = alpha;
  this->minprop = minprop;
  this->num_random_splits = num_random_splits;

  for( int ii = 0; ii < sc_variable_IDs->size(); ++ii ) { 
      this->sc_variable_IDs.insert(sc_variable_IDs->at(ii));
  }

  max_tree_height = maxTreeHeight;
  this->speedy    = speedy;

  initInternal();
}

// TODO: pass in vector of intersections as referece, make it a member function and dont pass dim_intervals
std::vector<std::pair<size_t, size_t>> Tree::find_intersections( size_t nodeID, const std::vector<std::pair<size_t, double>> & l_leaves, const std::vector<std::pair<size_t, double>>& r_leaves,
                                                           const std::vector<std::vector<std::pair<double, double>>> & dim_intervals  ) {
    std::vector<std::pair<size_t, size_t>> result;
    for( int l_idx = 0; l_idx < l_leaves.size(); ++l_idx ) {
        size_t l_leafID = l_leaves[l_idx].first;
        std::vector<std::pair<double, double>> l_cell = dim_intervals[l_leafID]; // TODO: avoid copy
        for( int r_idx = 0; r_idx < r_leaves.size(); ++r_idx ) {
            size_t r_leafID = r_leaves[r_idx].first;
            std::vector<std::pair<double, double>> r_cell = dim_intervals[r_leafID]; // TODO: avoid copy
            bool intersect = true;
            for( int d_idx = 0; d_idx < l_cell.size(); ++d_idx )  {

              // skip shape-constrained dimensions
              if(sc_variable_IDs.find(d_idx) != sc_variable_IDs.end()) {
                continue;
              }

              double i0 = std::max( l_cell[d_idx].first, r_cell[d_idx].first);
              double i1 = std::min( l_cell[d_idx].second, r_cell[d_idx].second);
              if(i0 >= i1) {
                intersect = false;

                /*
                std::cout << "does not intersection. dimension = " << d_idx << std::endl;
                std::cout << "left: ";
                for( size_t i = 0; i < dim_intervals[l_leafID].size(); ++i ) {
                  std::cout << "[" << i << ": (" << dim_intervals[l_leafID][i].first << "," << dim_intervals[l_leafID][i].second << ")] ";
                }
                std::cout << std::endl;

                std::cout << "right: ";
                for( size_t i = 0; i < dim_intervals[r_leafID].size(); ++i ) {
                  std::cout << "[" << i << ": (" << dim_intervals[r_leafID][i].first << "," << dim_intervals[r_leafID][i].second << ")] ";
                }
                std::cout << std::endl;
                */

                break;
              }
            }
            if( intersect ) {
                result.push_back(std::pair<size_t, size_t>(l_leafID, r_leafID));

                /*
                std::cout << "INTERSECTS" << std::endl;
                std::cout << "left: ";
                for( size_t i = 0; i < dim_intervals[l_leafID].size(); ++i ) {
                    std::cout << "[" << i << ": (" << dim_intervals[l_leafID][i].first << "," << dim_intervals[l_leafID][i].second << ")] ";
                }
                std::cout << std::endl;

                std::cout << "right: ";
                for( size_t i = 0; i < dim_intervals[r_leafID].size(); ++i ) {
                    std::cout << "[" << i << ": (" << dim_intervals[r_leafID][i].first << "," << dim_intervals[r_leafID][i].second << ")] ";
                }
                std::cout << std::endl;
                */
            }
        }
    }

    //std::cout << "size of intersection: " << result.size() << std::endl;
    return(result);

}

void Tree::grow(std::vector<double>* variable_importance) {

  this->variable_importance = variable_importance;

  // Bootstrap, dependent if weighted or not and with or without replacement
  if (case_weights->empty()) {
    if (sample_with_replacement) {
      bootstrap();
    } else {
      bootstrapWithoutReplacement();
    }
  } else {
    if (sample_with_replacement) {
      bootstrapWeighted();
    } else {
      bootstrapWithoutReplacementWeighted();
    }
  }
  
  post_bootstrap_init();

  auto t1 = std::chrono::high_resolution_clock::now();
// While not all nodes terminal, split next node
  size_t num_open_nodes = 1;
  size_t i = 0;
  while (num_open_nodes > 0) {
    bool is_terminal_node = splitNode(i);
    if (is_terminal_node) {
      --num_open_nodes;
    } else {
      ++num_open_nodes;
    }
    ++i;
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "timing,growTree," << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;


  t1 = std::chrono::high_resolution_clock::now();
  grow_post_process();
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "timing,growPostProcess," << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;
  
  //time_growTrees = std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count();
  //std::cout << "number of nodes," << sampleIDs.size() << std::endl;

  if(!sc_variable_IDs.empty()) {
     reshape();

     /*
     std::cout << "log,tree.height,under.num.constraints,goldi.num.constraints,over.num.constraints,num.sc.nodes,lowest.sc.depth,num.nodes,time.grow,time.goldiInt,time.underInt" << std::endl;
     std::cout << "log," << tree_height << "," << under_num_constraints << "," << goldi_num_constraints << "," 
       << over_num_constraints << "," << num_sc_nodes << "," << lowest_sc_depth  << "," << sampleIDs.size() 
       << "," << time_growTrees << "," << time_goldiInt << "," << time_underInt << std::endl;
       */
  }
// Delete sampleID vector to save memory
  dim_intervals.clear();
  sampleIDs.clear();
  cleanUpInternal();
}

std::vector<size_t> sort_indexes(const std::vector<double> &v, std::vector<double> & sorted_v, bool increasing) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  if( increasing ) {
	  sort(idx.begin(), idx.end(),
			  [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
  } else {
	  sort(idx.begin(), idx.end(),
			  [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
  }

  for( int ii = 0; ii < v.size(); ++ii ) {
	  sorted_v[ii] = v[idx[ii]];
  }
	
  return idx;
}

void Tree::over_constr_opt(size_t node, const std::vector<std::pair<size_t, double>> & left, const std::vector<std::pair<size_t, double>>& right) {

    //std::cout << "optimization size," << left.size() << "," << right.size() << "," << left.size() + right.size() << std::endl;

    // construct vectors for left and right nodeIDs [note use split_values for up to date values]
    std::vector<size_t> left_ids, right_ids;
    std::vector<double> left_vals, right_vals;
    for( auto& vv : left ) {
        left_ids.push_back(vv.first);
        left_vals.push_back(split_values[vv.first]);
    }

    for( auto& vv : right ) {
        right_ids.push_back(vv.first);
        right_vals.push_back(split_values[vv.first]);
    }

	std::vector<double> sorted_right(right_vals.size());
	std::vector<double> sorted_left(left_vals.size());

    // sort vectors and maintain list of sorted indices
	std::vector <size_t> right_idx = sort_indexes(right_vals, sorted_right, true);
	std::vector <size_t> left_idx  = sort_indexes(left_vals, sorted_left, false);

    
	/*
	std::cout << "before left values:" << std::endl;
	for( int ii = 0; ii < sorted_left.size(); ++ii ) {
		std::cout << sorted_left[ii] << ","; 
	}
	std::cout << std::endl;

	std::cout << "before right values:" << std::endl;
	for( int ii = 0; ii < sorted_right.size(); ++ii ) {
		std::cout << sorted_right[ii] << ","; 
	}
	std::cout << std::endl;
	*/

    // run algorithm
    size_t il = 0, ir = 0; 
    if( sorted_left[il] <= sorted_right[ir] ) {
        return;
    }

    double nn  = 2;
    double avg = (sorted_left[il] + sorted_right[ir]) / nn; 
    il++; ir++;

    bool l_end = (il == sorted_left.size());
    bool r_end = (ir == sorted_right.size());
	bool end   = l_end && r_end;

	bool l_violate = false;
	bool r_violate = false;
	bool violation = false;

	if( ! end ) {

		if( !l_end ) 
			l_violate = avg < sorted_left[il];
		if( !r_end ) 
			r_violate = avg > sorted_right[ir];
		violation = l_violate || r_violate;


		while( violation ) {
			if( l_violate && !l_end ) {
				avg = (nn / (nn+1))*avg + (1/(nn+1))*sorted_left[il];
				il++; nn++;
			}

			if( !r_end ) {
				r_violate = avg > sorted_right[ir];
				if( r_violate  ) {
					avg = (nn / (nn+1))*avg + (1/(nn+1))*sorted_right[ir];
					ir++; nn++;
				}
			}

			l_end = (il == sorted_left.size());
			r_end = (ir == sorted_right.size());
			end   = l_end && r_end;

			if( end ) break;

			if( !l_end ) 
				l_violate = avg < sorted_left[il];
			else
				l_violate = false;
			if( !r_end ) 
				r_violate = avg > sorted_right[ir];
			else
				r_violate = false;
			violation = l_violate || r_violate;
		}
	}

	for( size_t ii = 0; ii < il; ++ii ) {
		split_values[left_ids[left_idx[ii]]] = avg;
	}

	for( size_t ii = 0; ii < ir; ++ii ) {
		split_values[right_ids[right_idx[ii]]] = avg;
	}

	/*
	std::cout << "after left values:" << std::endl;
	for( int ii = 0; ii < sorted_left.size(); ++ii ) {
		std::cout << sorted_left[ii] << ","; 
	}
	std::cout << std::endl;

	std::cout << "after right values:" << std::endl;
	for( int ii = 0; ii < sorted_right.size(); ++ii ) {
		std::cout << sorted_right[ii] << ","; 
	}
	std::cout << std::endl;
	*/
}

std::vector<std::pair<size_t, double>> Tree::get_leaves(size_t node_id, const optmap & leftmap, const optmap & rightmap) {

    std::vector<std::pair<size_t, double>> result;

    size_t left_id    = child_nodeIDs[0][node_id];
    size_t right_id   = child_nodeIDs[1][node_id];

    if( left_id == 0 && right_id == 0 ) {
        std::pair<size_t, double> res_data;
        res_data.first  = node_id;
        res_data.second = split_values[node_id]; // probably don't need these stored in another data structure
        result.push_back(res_data);
        return(result);
    }

    std::vector<std::pair<size_t, double>> leftData, rightData;

    optmap::const_iterator got_left  = leftmap.find(node_id);
    if( got_left != leftmap.end() ) {
        leftData = got_left->second;
    } else {
        leftData          = get_leaves(left_id, leftmap, rightmap);
    }

    optmap::const_iterator got_right = rightmap.find(node_id);
    if( got_right != rightmap.end() ) {
        rightData = got_right->second;
    } else { 
        rightData         = get_leaves(right_id, leftmap, rightmap);
    }
       
    result.insert(result.end(), leftData.begin(), leftData.end());
    result.insert(result.end(), rightData.begin(), rightData.end());

    return(result);
}

void Tree::predict(const Data* prediction_data, bool oob_prediction) {

  size_t num_samples_predict;
  if (oob_prediction) {
    num_samples_predict = num_samples_oob;
  } else {
    num_samples_predict = prediction_data->getNumRows();
  }

  prediction_terminal_nodeIDs.resize(num_samples_predict, 0);

// For each sample start in root, drop down the tree and return final value
  for (size_t i = 0; i < num_samples_predict; ++i) {
    size_t sample_idx;
    if (oob_prediction) {
      sample_idx = oob_sampleIDs[i];
    } else {
      sample_idx = i;
    }
    size_t nodeID = 0;
    while (1) {

      // Break if terminal node
      if (child_nodeIDs[0][nodeID] == 0 && child_nodeIDs[1][nodeID] == 0) {
        break;
      }

      // Move to child
      size_t split_varID = split_varIDs[nodeID];
      double value = prediction_data->get(sample_idx, split_varID);
      if ((*is_ordered_variable)[split_varID]) {
        if (value <= split_values[nodeID]) {
          // Move to left child
          nodeID = child_nodeIDs[0][nodeID];
        } else {
          // Move to right child
          nodeID = child_nodeIDs[1][nodeID];
        }
      } else {
        size_t factorID = floor(value) - 1;
        size_t splitID = floor(split_values[nodeID]);

        // Left if 0 found at position factorID
        if (!(splitID & (1 << factorID))) {
          // Move to left child
          nodeID = child_nodeIDs[0][nodeID];
        } else {
          // Move to right child
          nodeID = child_nodeIDs[1][nodeID];
        }
      }
    }

    prediction_terminal_nodeIDs[i] = nodeID;
  }
}

void Tree::computePermutationImportance(std::vector<double>* forest_importance, std::vector<double>* forest_variance) {

  size_t num_independent_variables = data->getNumCols() - no_split_variables->size();

// Compute normal prediction accuracy for each tree. Predictions already computed..
  double accuracy_normal = computePredictionAccuracyInternal();

  prediction_terminal_nodeIDs.clear();
  prediction_terminal_nodeIDs.resize(num_samples_oob, 0);

// Reserve space for permutations, initialize with oob_sampleIDs
  std::vector<size_t> permutations(oob_sampleIDs);

// Randomly permute for all independent variables
  for (size_t i = 0; i < num_independent_variables; ++i) {

    // Skip no split variables
    size_t varID = i;
    for (auto& skip : *no_split_variables) {
      if (varID >= skip) {
        ++varID;
      }
    }

    // Permute and compute prediction accuracy again for this permutation and save difference
    permuteAndPredictOobSamples(varID, permutations);
    double accuracy_permuted = computePredictionAccuracyInternal();
    double accuracy_difference = accuracy_normal - accuracy_permuted;
    (*forest_importance)[i] += accuracy_difference;

    // Compute variance
    if (importance_mode == IMP_PERM_BREIMAN) {
      (*forest_variance)[i] += accuracy_difference * accuracy_difference;
    } else if (importance_mode == IMP_PERM_LIAW) {
      (*forest_variance)[i] += accuracy_difference * accuracy_difference * num_samples_oob;
    }
  }
}

void Tree::appendToFile(std::ofstream& file) {

// Save general fields
  saveVector2D(child_nodeIDs, file);
  saveVector1D(split_varIDs, file);
  saveVector1D(split_values, file);

// Call special functions for subclasses to save special fields.
  appendToFileInternal(file);
}

void Tree::createPossibleSplitVarSubset(std::vector<size_t>& result) {

// Always use deterministic variables
  std::copy(deterministic_varIDs->begin(), deterministic_varIDs->end(), std::inserter(result, result.end()));

// Randomly add non-deterministic variables (according to weights if needed)
  if (split_select_weights->empty()) {
    drawWithoutReplacementSkip(result, random_number_generator, data->getNumCols(), *no_split_variables, mtry);
  } else {
    size_t num_draws = mtry - result.size();
    drawWithoutReplacementWeighted(result, random_number_generator, *split_select_varIDs, num_draws,
        *split_select_weights);
  }
}

bool Tree::splitNode(size_t nodeID) {

// Select random subset of variables to possibly split at
  std::vector<size_t> possible_split_varIDs;
  createPossibleSplitVarSubset(possible_split_varIDs);

// Call subclass method, sets split_varIDs and split_values
  bool stop = splitNodeInternal(nodeID, possible_split_varIDs);

  if(node_depth[nodeID] == max_tree_height) {
    stop = true;
  }


  if (stop) {
    if( node_depth[nodeID] > tree_height ) {
      tree_height = node_depth[nodeID];
    }
    // Terminal node
    return true;
  }

  size_t split_varID = split_varIDs[nodeID];
  double split_value = split_values[nodeID];
  leafIDs.erase(nodeID);

// Create child nodes
  size_t left_child_nodeID = sampleIDs.size();
  leafIDs.insert(left_child_nodeID);
  child_nodeIDs[0][nodeID] = left_child_nodeID;
  createEmptyNode();
  dim_intervals.push_back(dim_intervals[nodeID]);
  dim_intervals[left_child_nodeID][split_varID].second = split_value;
  node_depth[left_child_nodeID] = node_depth[nodeID]+1;

  size_t right_child_nodeID = sampleIDs.size();
  leafIDs.insert(right_child_nodeID);
  child_nodeIDs[1][nodeID] = right_child_nodeID;
  createEmptyNode();
  dim_intervals.push_back(dim_intervals[nodeID]);
  dim_intervals[right_child_nodeID][split_varID].first = split_value;
  node_depth[right_child_nodeID] = node_depth[nodeID]+1;

  auto sc_itr      = sc_variable_IDs.find(split_varID);
  bool is_sc_split = sc_itr != sc_variable_IDs.end();
  if(is_sc_split && lowest_sc_depth == -1) {
    lowest_sc_depth = node_depth[nodeID];
  }
// For each sample in node, assign to left or right child
  if ((*is_ordered_variable)[split_varID]) {

    // Ordered: left is <= splitval and right is > splitval
    for (auto& sampleID : sampleIDs[nodeID]) {
      if (data->get(sampleID, split_varID) <= split_value) {
        sampleIDs[left_child_nodeID].push_back(sampleID);
        sampleID_to_leafID[sampleID] = left_child_nodeID;
      } else {
        sampleIDs[right_child_nodeID].push_back(sampleID);
        sampleID_to_leafID[sampleID] = right_child_nodeID;
      }
    }

  } else {
    // Unordered: If bit at position is 1 -> right, 0 -> left
    for (auto& sampleID : sampleIDs[nodeID]) {
      double level = data->get(sampleID, split_varID);
      size_t factorID = floor(level) - 1;
      size_t splitID = floor(split_value);

      // Left if 0 found at position factorID
      if (!(splitID & (1 << factorID))) {
        sampleIDs[left_child_nodeID].push_back(sampleID);
        sampleID_to_leafID[sampleID] = left_child_nodeID;
      } else {
        sampleIDs[right_child_nodeID].push_back(sampleID);
        sampleID_to_leafID[sampleID] = right_child_nodeID;
      }
    }
  }

  splitNode_post_process();
// No terminal node
  return false;
}

void Tree::createEmptyNode() {
  split_varIDs.push_back(0);
  split_values.push_back(0);
  child_nodeIDs[0].push_back(0);
  child_nodeIDs[1].push_back(0);
  sampleIDs.push_back(std::vector<size_t>());
  node_depth.push_back(0);

  if( dim_intervals.size() == 0 ) {
      dim_intervals.push_back(std::vector<std::pair<double,double>>());
      size_t num_vars = data->getVariableNames().size();
      for( int i = 0; i < num_vars; ++i )  {
        dim_intervals[0].push_back(std::pair<double,double>(-1 * std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()));
      }
  } 


  createEmptyNodeInternal();
}

size_t Tree::dropDownSamplePermuted(size_t permuted_varID, size_t sampleID, size_t permuted_sampleID) {

// Start in root and drop down
  size_t nodeID = 0;
  while (child_nodeIDs[0][nodeID] != 0 || child_nodeIDs[1][nodeID] != 0) {

    // Permute if variable is permutation variable
    size_t split_varID = split_varIDs[nodeID];
    size_t sampleID_final = sampleID;
    if (split_varID == permuted_varID) {
      sampleID_final = permuted_sampleID;
    }

    // Move to child
    double value = data->get(sampleID_final, split_varID);
    if ((*is_ordered_variable)[split_varID]) {
      if (value <= split_values[nodeID]) {
        // Move to left child
        nodeID = child_nodeIDs[0][nodeID];
      } else {
        // Move to right child
        nodeID = child_nodeIDs[1][nodeID];
      }
    } else {
      size_t factorID = floor(value) - 1;
      size_t splitID = floor(split_values[nodeID]);

      // Left if 0 found at position factorID
      if (!(splitID & (1 << factorID))) {
        // Move to left child
        nodeID = child_nodeIDs[0][nodeID];
      } else {
        // Move to right child
        nodeID = child_nodeIDs[1][nodeID];
      }
    }

  }
  return nodeID;
}

void Tree::permuteAndPredictOobSamples(size_t permuted_varID, std::vector<size_t>& permutations) {

// Permute OOB sample
//std::vector<size_t> permutations(oob_sampleIDs);
  std::shuffle(permutations.begin(), permutations.end(), random_number_generator);

// For each sample, drop down the tree and add prediction
  for (size_t i = 0; i < num_samples_oob; ++i) {
    size_t nodeID = dropDownSamplePermuted(permuted_varID, oob_sampleIDs[i], permutations[i]);
    prediction_terminal_nodeIDs[i] = nodeID;
  }
}

void Tree::bootstrap() {

// Use fraction (default 63.21%) of the samples
  size_t num_samples_inbag = (size_t) num_samples * sample_fraction;

// Reserve space, reserve a little more to be save)
  sampleIDs[0].reserve(num_samples_inbag);
  oob_sampleIDs.reserve(num_samples * (exp(-sample_fraction) + 0.1));

  std::uniform_int_distribution<size_t> unif_dist(0, num_samples - 1);

// Start with all samples OOB
  inbag_counts.resize(num_samples, 0);

  std::unordered_set<size_t> ids;
// Draw num_samples samples with replacement (num_samples_inbag out of n) as inbag and mark as not OOB
  for (size_t s = 0; s < num_samples_inbag; ++s) {
    size_t draw = unif_dist(random_number_generator);
    sampleIDs[0].push_back(draw);

    auto itr = ids.find(draw);
    if( itr == ids.end() ) {
      Sample tmp_sample(draw);
      ids.insert(draw);
    }

    node_depth.push_back(0);
    tree_height = 0;

    ++inbag_counts[draw];
  }


// Save OOB samples
  for (size_t s = 0; s < inbag_counts.size(); ++s) {
    if (inbag_counts[s] == 0) {
      oob_sampleIDs.push_back(s);
    }
  }
  num_samples_oob = oob_sampleIDs.size();

  if (!keep_inbag) {
    inbag_counts.clear();
  }
}

void Tree::bootstrapWeighted() {

// Use fraction (default 63.21%) of the samples
  size_t num_samples_inbag = (size_t) num_samples * sample_fraction;

// Reserve space, reserve a little more to be save)
  sampleIDs[0].reserve(num_samples_inbag);
  oob_sampleIDs.reserve(num_samples * (exp(-sample_fraction) + 0.1));

  std::discrete_distribution<> weighted_dist(case_weights->begin(), case_weights->end());

// Start with all samples OOB
  inbag_counts.resize(num_samples, 0);

  std::unordered_set<size_t> ids;
// Draw num_samples samples with replacement (n out of n) as inbag and mark as not OOB
  for (size_t s = 0; s < num_samples_inbag; ++s) {
    size_t draw = weighted_dist(random_number_generator);
    sampleIDs[0].push_back(draw);

    auto itr = ids.find(draw);
    if( itr == ids.end() ) {
      Sample tmp_sample(draw);
      ids.insert(draw);
    }
    ++inbag_counts[draw];
  }
  node_depth.push_back(0);
  tree_height = 0;

  // Save OOB samples. In holdout mode these are the cases with 0 weight.
  if (holdout) {
    for (size_t s = 0; s < (*case_weights).size(); ++s) {
      if ((*case_weights)[s] == 0) {
        oob_sampleIDs.push_back(s);
      }
    }
  } else {
    for (size_t s = 0; s < inbag_counts.size(); ++s) {
      if (inbag_counts[s] == 0) {
        oob_sampleIDs.push_back(s);
      }
    }
  }
  num_samples_oob = oob_sampleIDs.size();

  if (!keep_inbag) {
    inbag_counts.clear();
  }
}

void Tree::bootstrapWithoutReplacement() {

// Use fraction (default 63.21%) of the samples
  size_t num_samples_inbag = (size_t) num_samples * sample_fraction;
  shuffleAndSplit(sampleIDs[0], oob_sampleIDs, num_samples, num_samples_inbag, random_number_generator);
  num_samples_oob = oob_sampleIDs.size();

  if (keep_inbag) {
    // All observation are 0 or 1 times inbag
    inbag_counts.resize(num_samples, 1);
    for (size_t i = 0; i < oob_sampleIDs.size(); i++) {
      inbag_counts[oob_sampleIDs[i]] = 0;
    }
  }
}

void Tree::bootstrapWithoutReplacementWeighted() {

// Use fraction (default 63.21%) of the samples
  size_t num_samples_inbag = (size_t) num_samples * sample_fraction;
  drawWithoutReplacementWeighted(sampleIDs[0], random_number_generator, num_samples - 1, num_samples_inbag,
      *case_weights);

// All observation are 0 or 1 times inbag
  inbag_counts.resize(num_samples, 0);
  for (auto& sampleID : sampleIDs[0]) {
    inbag_counts[sampleID] = 1;
  }

// Save OOB samples. In holdout mode these are the cases with 0 weight.
  if (holdout) {
    for (size_t s = 0; s < (*case_weights).size(); ++s) {
      if ((*case_weights)[s] == 0) {
        oob_sampleIDs.push_back(s);
      }
    }
  } else {
    for (size_t s = 0; s < inbag_counts.size(); ++s) {
      if (inbag_counts[s] == 0) {
        oob_sampleIDs.push_back(s);
      }
    }
  }
  num_samples_oob = oob_sampleIDs.size();

  if (!keep_inbag) {
    inbag_counts.clear();
  }
}

