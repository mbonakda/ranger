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
 Matt Bonakdarpour
 #-------------------------------------------------------------------------------*/

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
#include <iomanip>

#include <ctime>

#include "utility.h"
#include "TreeDiscreteChoice.h"
#include "Data.h"

#include <limits>
typedef std::numeric_limits< double > dbl;

const size_t DEBUG = 0;

TreeDiscreteChoice::TreeDiscreteChoice() :
    counter(0), sums(0), dcrf_numItems(0), dcrf_numAgents(0), num_splits(0), debug(DEBUG)  {
}

TreeDiscreteChoice::TreeDiscreteChoice(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_varIDs,
    std::vector<double>& split_values, std::vector<bool>* is_ordered_variable) :
    Tree(child_nodeIDs, split_varIDs, split_values, is_ordered_variable), counter(0), sums(0), dcrf_numItems(0), dcrf_numAgents(0), num_splits(0), debug(DEBUG) {
}

TreeDiscreteChoice::~TreeDiscreteChoice() {
  // Empty on purpose
}

void TreeDiscreteChoice::initInternal() {
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

  num_splits = 0;
}

void TreeDiscreteChoice::post_bootstrap_init() {
  size_t agentID_varID = data->getVariableID("agentID");
  Tree::post_bootstrap_init();
  for (auto& sampleID : sampleIDs[0]) {
    size_t a_id = data->get(sampleID, agentID_varID);
    agentIDs.insert(a_id);
    auto itr = agentID_to_sampleIDs.find(a_id);
    if( itr == agentID_to_sampleIDs.end() ) {
      agentID_to_sampleIDs.emplace(a_id, std::vector<size_t>());
    }
    agentID_to_sampleIDs[a_id].push_back(sampleID);
    sampleID_to_agentIDs[sampleID] = a_id;

    auto atol_itr = agentID_to_leafIDs.find(a_id);
    if(atol_itr == agentID_to_leafIDs.end()){
      agentID_to_leafIDs.emplace(a_id, std::vector<size_t>());
    }
    agentID_to_leafIDs[a_id].push_back(0);
  }


  // assuming every agent considers the same number of items
  auto itr = agentID_to_sampleIDs.begin();
  auto vec = itr->second;
  dcrf_numItems = vec.size();
  dcrf_numAgents = agentIDs.size();


  // utility at root node does not affect log-lik. initialize to zero.
  util.push_back(0);
  // TODO: this likelihood calculation is incorrect when bootstrapping
  llik.push_back((dcrf_numAgents*util[0])-dcrf_numAgents*log((double)dcrf_numItems*exp(util[0])));

}

void TreeDiscreteChoice::appendToFileInternal(std::ofstream& file) {
// Empty on purpose
}

void TreeDiscreteChoice::splitNode_post_process() {
    // get leaf IDs
    std::set<size_t>  leafIDs;
    double util_sum = 0;
    for(auto a_id: agentIDs) {
        for(auto s_id: agentID_to_sampleIDs[a_id]) {
            size_t leaf_id = sampleID_to_leafID[s_id];
            auto itr = leafIDs.find(leaf_id);
            if(itr == leafIDs.end())  {
                util_sum += util[leaf_id];
                leafIDs.insert(leaf_id);
            }
        }
    }
    for( auto& l_id : leafIDs ) {
        util[l_id] -= util_sum / leafIDs.size();
    }
}

bool TreeDiscreteChoice::splitNodeInternal(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {

  //TODO: do better here
  size_t agentID_varID = data->getVariableID("agentID");
  possible_split_varIDs.erase(std::remove(possible_split_varIDs.begin(), possible_split_varIDs.end(), agentID_varID), possible_split_varIDs.end());

  // Check node size, stop if maximum reached
  if (sampleIDs[nodeID].size() <= min_node_size) {
    split_values[nodeID] = util[nodeID];
    return true;
  }
  
  
  if(node_depth[nodeID] == max_tree_height) {
    split_values[nodeID] = util[nodeID];
    return true;
  }

  // Find best split, stop if no decrease of impurity
  bool stop;
  stop = findBestSplit(nodeID, possible_split_varIDs);


  // TODO: this is incorrect for greedy MLE -- future iterations might split this node
  if (stop) {
    split_values[nodeID] = util[nodeID];
    return true;
  } 

  // TODO: this is temporary. need to change for greedy MLE
  // assumes left-right order of creation in Tree.cpp
  util.push_back(child_util[0][nodeID]);
  util.push_back(child_util[1][nodeID]);
  num_splits += 1;
  return false;
}

void TreeDiscreteChoice::createEmptyNodeInternal() {
  // put this here instead of constructor because Tree() construction calls this function
  if(child_util.size() == 0) {
      child_util.push_back(std::vector<double>());
      child_util.push_back(std::vector<double>());
  }
  child_util[0].push_back(0.0);
  child_util[1].push_back(0.0);
}

double TreeDiscreteChoice::computePredictionAccuracyInternal() {
  return 0;
}

bool TreeDiscreteChoice::findBestSplit(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {

  size_t num_samples_node = sampleIDs[nodeID].size();
  double best_increase = -1;
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
      findBestSplitValue(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_increase);
    } else {
      std::cout << "ERROR - can only handle ordered covariates for now" << std::endl;
      exit(0);
    }
  }

  // Stop if no good split found
  if (best_increase <= 0) {
      return true;
  }

  // Save best values
  split_varIDs[nodeID] = best_varID;
  split_values[nodeID] = best_value;

  return false;
}

void TreeDiscreteChoice::findBestSplitValue(size_t nodeID, size_t varID, double sum_node, size_t num_samples_node,
    double& best_value, size_t& best_varID, double& best_increase) {

  // Set counters to 0
  size_t num_unique = data->getNumUniqueDataValues(varID);
  std::fill(counter, counter + num_unique, 0);
  std::fill(sums, sums + num_unique, 0);

  // number of positive choices in current leaf, and potential left/right leaf
  size_t c_star      = 0, c_l = 0, c_r = 0; 
  size_t num_samples = 0;

  size_t agentID_varID = data->getVariableID("agentID");

  // agentID -> num samples for pre-split node
  std::unordered_map<size_t, size_t> n_star; 
  // agentID -> num samples for left/right leaves
  std::unordered_map<size_t, size_t> n_l, n_r; 
  // value index --> agentID --> numSamples
  std::unordered_map<size_t, std::unordered_map<size_t, size_t> > idx_agent_n; 
  // all agentIDs in this node
  std::unordered_set<size_t> node_agentIDs;
  // agentID --> partition func
  std::unordered_map<size_t, double> agent_Z;
  // idx --> sampleIDs
  std::unordered_map<size_t, std::unordered_set<size_t>> idx_to_sID;
  // sampleIDs in left/right leaves
  std::unordered_set<size_t> left_sIDs, right_sIDs;


  // setup initial state for optimization before iterating through split values
  // TODO: some things can be moved up a level. do not need to
  //       compute these for every split variable (they don't change)!
  for (auto& sampleID : sampleIDs[nodeID]) {
    size_t index                  = data->getIndex(sampleID, varID);
    size_t agentID                = data->get(sampleID, agentID_varID);

    double response               = data->get(sampleID, dependent_varID);
    sums[index]                  += response;
    c_star                       += response;

    n_star[agentID]              += 1; // TODO: up a level
    n_r[agentID]                 += 1; // assume all samples start in right leaf TODO: up a level
    idx_agent_n[index][agentID]  += 1;  
    num_samples                  += 1; // TODO: up a level

    node_agentIDs.insert(agentID); // TODO: up a level

    auto itr = idx_to_sID.find(index); // TODO: up a few levels (shouldn't change for the entire fitting process)
    if( itr == idx_to_sID.end() ) {
      idx_to_sID.emplace(index, std::unordered_set<size_t>());
    }
    idx_to_sID[index].insert(sampleID);

    right_sIDs.insert(sampleID);
    ++counter[index];
  }

  // TODO: up a level
  // compute partition funcs for each agent
  for(auto a_id: node_agentIDs) {
    for(auto s_id: agentID_to_sampleIDs[a_id]) {
      size_t leaf_id = sampleID_to_leafID[s_id];
      agent_Z[a_id] += exp(util[leaf_id]);
    }
  }

  size_t n_left = 0;
  double sum_left = 0, sum_right = 0; // TODO: do we need these?

  
  // compute current likelihood
  double curr_llik = compute_log_likelihood(agent_Z, util, node_agentIDs); // TODO: up a level 

  double V_star    = util[nodeID];

  // iterate through all possible split values
  for (size_t i = 0; i < num_unique - 1; ++i) {
	  
    // note: warm starting to previous V_L/V_R by moving this outside of the loop 
    //       causes numerical issues
	double curr_VL = util[nodeID];
	double curr_VR = util[nodeID];

    /***********************************************************************************
    // update constants for optimization and check for short circuits
    ***********************************************************************************/
    // Stop if nothing here
    if (counter[i] == 0) {
      continue;
    }
    
    c_l += sums[i];
    c_r  = c_star - c_l;
    for(auto a_to_n :idx_agent_n[i] ) {
      n_l[a_to_n.first] += a_to_n.second;
      n_r[a_to_n.first] = n_star[a_to_n.first] - n_l[a_to_n.first];
    }

    for(auto sID : idx_to_sID[i]) {
      right_sIDs.erase(sID);
      left_sIDs.insert(sID);
    }

    n_left         += counter[i];
    sum_left       += sums[i];
    sum_right       = c_star - sum_left;

    // Stop if right child empty
    size_t n_right  = num_samples_node - n_left;
    if (n_right == 0) {
      break;
    }

    // Stop if agent pure split
    // (assumes all agents have same number of items to choose from)
    bool agent_pure = true;
    for(auto a_id: node_agentIDs) {
        if( !agent_pure ) { break; } // short circuit
        for(auto sample_id : agentID_to_sampleIDs[a_id]) {
            size_t leaf_id = sampleID_to_leafID[sample_id];
            if( leaf_id != nodeID) {
                agent_pure = false;
                break;
            }
            if( leaf_id == nodeID ) {
                const bool is_r = right_sIDs.find(sample_id) != right_sIDs.end();
                if(is_r) {
                    if(n_r[a_id] != dcrf_numItems) {
                        agent_pure = false;
                        break;
                    }
                } else {
                    if(n_l[a_id] != dcrf_numItems) {
                        agent_pure = false;
                        break;
                    }
                }
            }
        }
    }
    
    if(agent_pure)  {
      continue;
    }
    /***********************************************************************************/

    /*****************************************************************
    // Maximum Likelihood
    *****************************************************************/
    double llik;
    if( i == 0 ) {
        llik    = curr_llik;
    } else { // warm start the VL/VR to previous iteration
        llik = compute_temp_log_likelihood(agent_Z, util, node_agentIDs, 
                curr_VL, curr_VR, V_star,
                right_sIDs, n_l, n_r, nodeID);
    }

    double step_norm       = 0;
    size_t num_newton_iter = 0;
    double dVL = 0, dVR=0; // partial derivatives
    double prev_llik;
	if(debug)  {
		std::cout << "considering split,nodeID=" << nodeID 
			      << "\tcovariate=" << data->getVariableNames()[varID]  
			      << "\tcovariate value=" << data->getUniqueDataValue(varID, i) 
				  << std::endl;
	}
    do { 
      prev_llik        = llik;

      num_newton_iter += 1;
      if(debug)
          std::cout << "newton iteration = " << num_newton_iter 
                    << "\tprev_llik=" << prev_llik 
                    << "\tcurr_VL=" << curr_VL
                    << "\tcurr_VR=" << curr_VR 
                    << std::endl;

      dVL = 0;
      dVR = 0;
      double dVL2 = 0, dVR2 = 0, dVLVR = 0;

      /*****************************************************************
      // compute direction of step
       *****************************************************************/
      if(nodeID == 0) {
        dVL += static_cast<double>(c_l) - static_cast<double>(c_r);
      } else {
        dVL += static_cast<double>(c_l);
        dVR += static_cast<double>(c_r);
      }
      for(auto a_id: node_agentIDs) {

        double Z_curr = agent_Z[a_id] 
                            - n_l[a_id]*exp(V_star) + n_l[a_id]*exp(curr_VL) 
                            - n_r[a_id]*exp(V_star) + n_r[a_id]*exp(curr_VR);

        if( nodeID == 0) {
          double   mm  = ( n_l[a_id]*exp(curr_VL) - n_r[a_id]*exp(curr_VR) );
          double   pp  = ( n_l[a_id]*exp(curr_VL) + n_r[a_id]*exp(curr_VR) );
          dVL      -=  mm / Z_curr;
          dVR       =  -dVL;
          dVL2     -=  ( Z_curr*pp  - mm*mm ) / (Z_curr*Z_curr);
        } else {
          dVL      -= n_l[a_id]*exp(curr_VL) / Z_curr;
          dVR      -= n_r[a_id]*exp(curr_VR) / Z_curr;

          dVL2     -= ( n_l[a_id]*exp(curr_VL) * ( Z_curr - ( n_l[a_id]*exp(curr_VL) ) ) ) / ( Z_curr * Z_curr );
          dVR2     -= ( n_r[a_id]*exp(curr_VR) * ( Z_curr - ( n_r[a_id]*exp(curr_VR) ) ) ) / ( Z_curr * Z_curr );
          dVLVR    += ( n_l[a_id]*exp(curr_VL) * n_r[a_id]*exp(curr_VR) ) / (Z_curr * Z_curr);
		  if(debug > 1) { 
			  std::cout << std::fixed 
				  << "\t\tagentID=" << a_id 
				  << "\tn_l=" << n_l[a_id] 
				  << "\tn_r=" << n_r[a_id] 
				  << "\texp(curr_VL)=" << exp(curr_VL)
				  << "\texp(curr_VR)=" << exp(curr_VR)
				  << "\tZ_curr=" << Z_curr 
				  << "\tdVL=" << dVL  
				  << "\tdVR=" << dVR
				  << "\tc_l=" << c_l
				  << "\tc_r=" << c_r
				  << "\tprecision=" << dbl::max_digits10
				  << std::endl;
		  }
        }

      }

      double dtmnt = 1.0 / (dVL2*dVR2 - dVLVR*dVLVR);

      double delta_VL = 0, delta_VR = 0;
      if(nodeID == 0) { // univariate newton
        delta_VL = -(1.0 / dVL2)*dVL;
        delta_VR = -delta_VL;
      } else { // newton otherwise
        delta_VL      = -dtmnt*((dVR2) *dVL - dVLVR*dVR);
        delta_VR      = -dtmnt*(-dVLVR*dVL + (dVL2)*dVR);
      }

      step_norm = sqrt( (delta_VL*delta_VL) + (delta_VR*delta_VR) );
      /*****************************************************************/

      double temp_VL = curr_VL + delta_VL;
      double temp_VR = curr_VR + delta_VR;

      llik = compute_temp_log_likelihood(agent_Z, util, node_agentIDs, 
                                         temp_VL, temp_VR, V_star,
                                         right_sIDs, n_l, n_r, nodeID);

      if(debug) {
        std::cout << "newton step" << std::endl;
        std::cout << "\tprev_llik=" << prev_llik 
          << "\tllik=" << llik 
          << "\tdiff=" << llik - prev_llik 
          << "\tdelta_VL=" << delta_VL 
          << "\tdelta_VR=" << delta_VR 
          << "\tdVL=" << dVL 
          << "\tdVR=" << dVR 
          << "\tdVL2=" << dVL2
          << "\tdVR2=" << dVR2 
          << "\tdVLVR=" << dVLVR
          << "\t1/dtmnt=" << dtmnt
          << "\ttemp_VL=" << temp_VL
          << "\tcurr_VL=" << curr_VL
          << "\ttemp_VR=" << temp_VR
          << "\tcurr_VR=" << curr_VR
          << std::endl;
      }

      /*********************************************************************
       * line search
      ************************************************************************/
      double stepsize  = 1;
      double alpha     = 0.3;
      double beta      = 0.8;

      double threshold;
      if( nodeID == 0 ) {
        threshold = stepsize*alpha*dVL*delta_VL;
      } else {
        double thresh_factor = dVL*delta_VL + dVR*delta_VR;
        threshold            = stepsize*alpha*thresh_factor;
      }

      size_t num_lineSearch_iters = 0;
      while(llik - prev_llik < threshold && step_norm > 1e-6 && fabs(dVL) > 1e-10 && fabs(dVR) > 1e-10){
          num_lineSearch_iters += 1;
		  if(debug) {
			  std::cout << "line search iteration = " << num_lineSearch_iters << std::endl;
		  }
		  if(num_lineSearch_iters > 100) {
              if(debug)
                  std::cout << "line search break" << std::endl;
			  break;
		  }

          stepsize   = stepsize * beta;
          temp_VL    = curr_VL + stepsize*delta_VL;
          temp_VR    = curr_VR + stepsize*delta_VR;

          llik = compute_temp_log_likelihood(agent_Z, util, node_agentIDs, 
                                             temp_VL, temp_VR, V_star,
                                             right_sIDs, n_l, n_r, nodeID);

          if( nodeID == 0 ) {
              threshold = -stepsize*alpha*dVL*delta_VL;
          } else {
              double thresh_factor = dVL*delta_VL + dVR*delta_VR;
              threshold            = stepsize*alpha*thresh_factor;
          }
          if(debug > 1) {
              std::cout << "\t\tprev_llik=" << prev_llik 
                  << "\tllik=" << llik 
                  << "\tstepsize=" << stepsize
                  << "\tstepnorm=" << step_norm
                  << "\tdiff=" << llik - prev_llik 
                  << "\tthreshold=" << threshold 
                  << "\tdelta_VL=" << delta_VL 
                  << "\tdelta_VR=" << delta_VR 
                  << "\tdVL=" << dVL 
                  << "\tdVR=" << dVR 
                  << "\tdVL2=" << dVL2
                  << "\tdVR2=" << dVR2 
                  << "\tdVLVR=" << dVLVR
                  << "\t1/dtmnt=" << dtmnt
                  << "\ttemp_VL=" << temp_VL
                  << "\tcurr_VL=" << curr_VL
                  << "\ttemp_VR=" << temp_VR
                  << "\tcurr_VR=" << curr_VR
                  << "\tnodeID=" << nodeID
                  << std::endl;
          }
      } // end line search

      if(threshold == 0 && step_norm !=0) {
          std::cout << "line search failed" << std::endl;
      }

      // don't accept if we didn't cross threshold
      if( llik - prev_llik > threshold && ( threshold != 0 ) ) {
        curr_VL = temp_VL;
        curr_VR = temp_VR;
      } else {
          llik = prev_llik;
      }
   } while( (llik - prev_llik) > 1e-4  && step_norm > 1e-6); 
    /*****************************************************************/
	if(debug) {
		std::cout << "after newton,curr_VL=" << curr_VL 
				  << "\tcurr_VR=" << curr_VR
				  << "\tllik=" << llik
				  << std::endl;
	}
    double increase = llik - curr_llik;
    // If better than before, use this
    if (increase > best_increase) {
      //std::cout << "new best increase=" << increase << "\tindex=" << i << std::endl;
      // Find next value in this node
      size_t j = i + 1;
      while(j < num_unique && counter[j] == 0) {
        ++j;
      }

      // Use mid-point split
      best_value    = (data->getUniqueDataValue(varID, i) + data->getUniqueDataValue(varID, j)) / 2;
      best_varID    = varID;
      best_increase = increase;
      child_util[0][nodeID] = curr_VL;
      child_util[1][nodeID] = curr_VR;
    } else {
      //std::cout << "index=" << i << " not good enough with increase=" << increase << std::endl;
    }
  }
}

// adjust all leaves 
void TreeDiscreteChoice::grow_post_process(){

    std::unordered_map<size_t, double> agent_Z;
    std::unordered_map<size_t, double> leafID_to_partial;
    std::set<size_t>                   leafIDs;
    std::vector<double>                curr_util = util;


    // get leaf IDs
    for(auto a_id: agentIDs) {
        for(auto s_id: agentID_to_sampleIDs[a_id]) {
            size_t leaf_id = sampleID_to_leafID[s_id];
            leafIDs.insert(leaf_id);
        }
    }

    // compute current state
    compute_partition_func(agent_Z, curr_util);
    double curr_llik = compute_log_likelihood(agent_Z, curr_util, agentIDs);

    // gradient descent
    double prev_llik;
    do {

        prev_llik       = curr_llik;
        compute_full_gradient(leafID_to_partial, agent_Z, curr_util, leafIDs);

        // armijo line search
        curr_llik = backtracking(leafID_to_partial, agent_Z, curr_util, leafIDs, prev_llik);

    } while ( curr_llik - prev_llik > 1e-5 );

    util = curr_util;
    
    for(auto & l_id : leafIDs) {
      split_values[l_id] = util[l_id];
    }
}

double TreeDiscreteChoice::backtracking(const std::unordered_map<size_t,double>& leafID_to_partial,std::unordered_map<size_t, double>& agent_Z, 
                                      std::vector<double>& curr_util, const std::set<size_t>& leafIDs, double prev_llik) {

    std::vector<double> temp_util = curr_util;
    double stepsize          = 1;
    double alpha             = 0.3;
    double beta              = 0.8;
    double grad_norm_squared = 0;

    double max_grad_component = 0;
    for( auto l_id : leafIDs ) {
        temp_util[l_id]         += stepsize*leafID_to_partial.at(l_id); 
        grad_norm_squared       += leafID_to_partial.at(l_id)*leafID_to_partial.at(l_id);
        if( fabs(leafID_to_partial.at(l_id)) > max_grad_component) {
          max_grad_component = leafID_to_partial.at(l_id);
        }
    }
    double grad_norm = sqrt(grad_norm_squared);

    compute_partition_func(agent_Z, temp_util);
    double curr_llik = compute_log_likelihood(agent_Z, temp_util, agentIDs);

    size_t num_iter = 0;
    while(curr_llik - prev_llik < alpha*grad_norm_squared*stepsize  && grad_norm > 1e-4) {
        num_iter += 1;
        temp_util  = curr_util;
        stepsize   = stepsize * beta;
        for( auto l_id : leafIDs ) {
            temp_util[l_id] += stepsize*leafID_to_partial.at(l_id); 
        }
        compute_partition_func(agent_Z, temp_util);
        curr_llik = compute_log_likelihood(agent_Z, temp_util, agentIDs);

        if(num_iter > 1000) {
            std::cout << "global adjustment line search failed" << std::endl;
        }
    }

    if(curr_llik - prev_llik > alpha*grad_norm_squared*stepsize) { // passed
      curr_util = temp_util;
    } else { // did not pass
      curr_llik = prev_llik;
    }
    return curr_llik;
}

void TreeDiscreteChoice::compute_partition_func(std::unordered_map<size_t, double>& agent_Z, const std::vector<double>& curr_util) {
    for(auto a_id: agentIDs) {
        agent_Z[a_id] = 0;
        for(auto s_id: agentID_to_sampleIDs[a_id]) {
            size_t leaf_id = sampleID_to_leafID[s_id];
            agent_Z[a_id] += exp(curr_util[leaf_id]);
        }
    }
    return;
}

double TreeDiscreteChoice::compute_temp_log_likelihood(const std::unordered_map<size_t, double>& agent_Z, 
                                                       const std::vector<double>& curr_util,
                                                       const std::unordered_set<size_t>& agent_ids,
                                                       double V_L, double V_R, double V_star,
                                                       const std::unordered_set<size_t>& right_sIDs,
                                                       std::unordered_map<size_t, size_t>& n_l,
                                                       std::unordered_map<size_t, size_t>& n_r,
                                                       size_t nodeID) 
{
    double llik = 0;
    for(auto agent_id: agent_ids) {

        // update partition function
        double Z_curr = agent_Z.at(agent_id) 
                         - n_l[agent_id]*exp(V_star) + n_l[agent_id]*exp(V_L) 
                         - n_r[agent_id]*exp(V_star) + n_r[agent_id]*exp(V_R);

        for(auto sample_id: agentID_to_sampleIDs[agent_id]) {
            double response  = data->get(sample_id, dependent_varID);
            size_t leaf_id   = sampleID_to_leafID[sample_id];
            if( leaf_id == nodeID ) {
                const bool is_r = right_sIDs.find(sample_id) != right_sIDs.end();
                if(is_r) {
                    llik += response*(V_R - log(Z_curr));
                } else {
                    llik += response*(V_L - log(Z_curr));
                }
            } else {
                llik += response*(curr_util[leaf_id] - log(Z_curr));
            }
        }
    }
    return llik;
}

double TreeDiscreteChoice::compute_log_likelihood(const std::unordered_map<size_t, double>& agent_Z, 
                                                  const std::vector<double>& curr_util,
                                                  const std::unordered_set<size_t>& agent_ids) 
{
    double llik = 0;
    for(auto a_id: agent_ids) {
        for(auto s_id: agentID_to_sampleIDs[a_id]) {
            double response  = data->get(s_id, dependent_varID);
            double Z         = agent_Z.at(a_id);
            size_t l_id      = sampleID_to_leafID[s_id];
            llik             += response*(curr_util[l_id] - log(Z));
        }
    }
    return llik;
}


void TreeDiscreteChoice::compute_full_gradient(std::unordered_map<size_t,double>& leafID_to_partial, const std::unordered_map<size_t, double>& agent_Z, const std::vector<double>& curr_util, const std::set<size_t>& leafIDs) {

    for(auto & l_id : leafIDs) {
      leafID_to_partial[l_id] = 0;
    }
    // compute gradient
    for (auto a_id: agentIDs) {
        for(auto s_id: agentID_to_sampleIDs[a_id]) {
            size_t leaf_id              = sampleID_to_leafID[s_id];
            double response             = data->get(s_id, dependent_varID);
            double util                 = curr_util.at(leaf_id);
            double Z                    = agent_Z.at(a_id);
            leafID_to_partial[leaf_id] += response - exp(util)/Z;
        }
    }
    return;
}
