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

/*
 * 1 = descriptive tree stats
 * 2 = high-level optimization stats
 * 3 = low-level optimization stats
 */
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

double TreeDiscreteChoice::estimate(size_t nodeID) {

// Mean of responses of samples in node
  double sum_responses_in_node = 0;
  size_t num_samples_in_node = sampleIDs[nodeID].size();
  for (size_t i = 0; i < sampleIDs[nodeID].size(); ++i) {
    sum_responses_in_node += data->get(sampleIDs[nodeID][i], dependent_varID);
  }
  return (sum_responses_in_node / (double) num_samples_in_node);
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
        if(debug >=1) {
            std::cout << "[nodeID=" << l_id << ",value=" << util[l_id] << "] - ";
        }
    }
    if(debug >=1) {
        std::cout << std::endl;
    }
}

bool TreeDiscreteChoice::splitNodeInternal(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {

  //TODO: do better here
  size_t agentID_varID = data->getVariableID("agentID");
  possible_split_varIDs.erase(std::remove(possible_split_varIDs.begin(), possible_split_varIDs.end(), agentID_varID), possible_split_varIDs.end());

  // Check node size, stop if maximum reached
  if (sampleIDs[nodeID].size() <= min_node_size) {
    if(debug >= 1)
      std::cout << "nodeID=" << nodeID << "\tSTOP: reached min node size" << std::endl;
    split_values[nodeID] = util[nodeID];
    return true;
  }
  
  
  if(node_depth[nodeID] == max_tree_height) {
    split_values[nodeID] = util[nodeID];
    if(debug >= 1)
      std::cout << "nodeID=" << nodeID<< "\tSTOP: reached max tree height" << std::endl;
    return true;
  }

  /*
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
    split_values[nodeID] = util[nodeID];
    std::cout << "PURE NODE" << std::endl;
    return true;
  }
  */

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
    if(debug >= 1) 
      std::cout << "nodeID=" << nodeID << "\tSTOP: no increase in log-likelihood. nodeSize=" << sampleIDs[nodeID].size() << std::endl;
    return true;
  }

  if(debug >= 1)
      std::cout << "nodeID=" << nodeID 
                << "\tbestSplitVar=" << data->getVariableNames()[best_varID] 
                << "\tbestSplitValue=" << best_value 
                << "\tV_L=" << child_util[0][nodeID]
                << "\tV_R=" << child_util[1][nodeID]
                << std::endl;
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

  size_t c_star = 0, c_l = 0, c_r = 0; // number of positive choices in current leaf, and potential left/right leaf
  size_t num_samples = 0;

  size_t agentID_varID = data->getVariableID("agentID");

  //TODO: clean-up variable names
  // agentID -> num samples for pre-split node
  std::unordered_map<size_t, size_t> n_star; 
  // index --> agent --> numSamples
  std::unordered_map<size_t, std::unordered_map<size_t, size_t> > idx_agent_n; 
  // all agentIDs in this node
  std::unordered_set<size_t> node_agentIDs;
  // agentID --> partition func
  std::unordered_map<size_t, double> agent_Z;
  // idx --> sampleIDs
  std::unordered_map<size_t, std::unordered_set<size_t>> idx_to_sID;
  // sampleIDs in left/right leaves
  std::unordered_set<size_t> left_sIDs, right_sIDs;
  // agentID -> num samples for left/right leaves
  std::unordered_map<size_t, size_t> n_l, n_r; 


  for (auto& sampleID : sampleIDs[nodeID]) {
    size_t index                  = data->getIndex(sampleID, varID);
    size_t agentID                = data->get(sampleID, agentID_varID);

    double response               = data->get(sampleID, dependent_varID);
    sums[index]                  += response;
    c_star                       += response;

    n_star[agentID]              += 1;
    n_r[agentID]                 += 1;// assume all samples start in right leaf
    idx_agent_n[index][agentID]  += 1;  
    num_samples                  += 1;

    node_agentIDs.insert(agentID);

    auto itr = idx_to_sID.find(index);
    if( itr == idx_to_sID.end() ) {
      idx_to_sID.emplace(index, std::unordered_set<size_t>());
    }
    idx_to_sID[index].insert(sampleID);
    right_sIDs.insert(sampleID);
    ++counter[index];
  }

  bool is_pure_node = (c_star == num_samples) || (c_star == 0);

  
  // compute partition funcs for each agent
  for(auto a_id: node_agentIDs) {
    for(auto s_id: agentID_to_sampleIDs[a_id]) {
      size_t leaf_id = sampleID_to_leafID[s_id];
      agent_Z[a_id] += exp(util[leaf_id]);
    }
  }

  size_t n_left = 0;
  double sum_left = 0, sum_right = 0;

  // compute current likelihood
  double curr_llik = 0;
  for(auto a_id: node_agentIDs) {
    std::vector<size_t> a_sIDs = agentID_to_sampleIDs[a_id];
    //TODO: only need the agent's sampleIDs with response = 1
    for(auto s_id : a_sIDs) {
      double response = data->get(s_id, dependent_varID);
      if( response != 0 ) {
        size_t l_id = sampleID_to_leafID[s_id];
        curr_llik  += util[l_id] - log(agent_Z[a_id]);
        break;
      } 
    }
  }



  double V_star  = util[nodeID];

  for (size_t i = 0; i < num_unique - 1; ++i) {


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

    // Stop if right child empty
    n_left         += counter[i];
    sum_left       += sums[i];
    sum_right       = c_star - sum_left;
    size_t n_right  = num_samples_node - n_left;
    if (n_right == 0) {
      break;
    }

    double curr_VL = util[nodeID];
    double curr_VR = util[nodeID];

    // compute current likelihood
    bool agent_pure_l = true, agent_pure_r = true;
    double llik = 0;
    for(auto a_id: node_agentIDs) {

      std::vector<size_t> a_sIDs = agentID_to_sampleIDs[a_id];
      double Z_curr = agent_Z[a_id] - n_l[a_id]*exp(V_star) + n_l[a_id]*exp(curr_VL) - n_r[a_id]*exp(V_star) + n_r[a_id]*exp(curr_VR);
      //TODO: only need the agent's sampleIDs with response = 1
      for(auto s_id : a_sIDs) {
        double response = data->get(s_id, dependent_varID);
        if( response != 0 ) {
          size_t l_id = sampleID_to_leafID[s_id];
          if( l_id == nodeID ) {
            const bool is_r = right_sIDs.find(s_id) != right_sIDs.end();
            if(is_r) {
                if(n_r[a_id] != dcrf_numItems) {
                    agent_pure_r = false;
                }
              llik += curr_VR - log(Z_curr);
            } else {
                if(n_l[a_id] != dcrf_numItems) {
                    agent_pure_l = false;
                }
              llik += curr_VL - log(Z_curr);
            }
          } else {
            llik += util[l_id] - log(Z_curr);
          }
          break;
        } 
      }
    }
    
    bool agent_pure = agent_pure_l || agent_pure_r;
    if(agent_pure) 
        continue;

    if( debug >= 2 ) { 
      std::cout << "considering split,nodeID=" << nodeID << "\tcovariate=" << data->getVariableNames()[varID]  << "\tcovariate value=" << data->getUniqueDataValue(varID, i) 
          << "\tn_left=" << n_left << "\tsum_left=" << sum_left << "\tn_right=" << n_right << "\tsum_right=" << sum_right << std::endl;
      std::cout << "newton + line search -----------------" << std::endl;
      std::cout << "\tpre_split_llik=" << curr_llik << "\tinitial llik=" << llik << "\tinit_VL=" << curr_VL << "\tinit_VR=" << curr_VR << std::endl;
    }

    /*****************************************************************
    // Maximum Likelihood
    *****************************************************************/
    double grad_norm = 0;
    double step_norm = 0;
    double prev_llik;
    size_t num_newton_iter = 0;
    double dVL = 0, dVR=0;
    do { 
      num_newton_iter += 1;
      prev_llik = llik;
      // compute entries for gradient + hessian
      dVL = 0;
      dVR = 0;
      //double dVL2 = 0;
      double dVL2 = 0, dVR2 = 0, dVLVR = 0;

      if(nodeID == 0) {
        dVL += static_cast<double>(c_l) - static_cast<double>(c_r);
      } else {
        dVL += static_cast<double>(c_l);
        dVR += static_cast<double>(c_r);
      }
      for(auto a_id: node_agentIDs) {

        double Z_curr = agent_Z[a_id] - n_l[a_id]*exp(V_star) + n_l[a_id]*exp(curr_VL) - n_r[a_id]*exp(V_star) + n_r[a_id]*exp(curr_VR);

        // constrained optimization update
         
        if( nodeID == 0) {
          double   mm  = ( n_l[a_id]*exp(curr_VL) - n_r[a_id]*exp(curr_VR) );
          double   pp  = ( n_l[a_id]*exp(curr_VL) + n_r[a_id]*exp(curr_VR) );
          dVL      -=  mm / Z_curr;
          dVL2     -=  ( Z_curr*pp  - mm*mm ) / (Z_curr*Z_curr);

          if(debug >= 3) { 
            std::cout << "\t\tagentID=" << a_id 
              << "\tn_l=" << n_l[a_id] 
              << "\tn_r=" << n_r[a_id] 
              << "\tZ_curr=" << Z_curr 
              << "\tdVL=" << dVL  
              << "\tdVL2=" << dVL2 
              << "\tmm=" << mm 
              << "\tpp=" << pp 
              << "\tmm/Z_curr=" << mm/Z_curr
              << "\tc_l=" << c_l
              << "\tc_r=" << c_r
              << std::endl;
          }

        } else {
          dVL      -= n_l[a_id]*exp(curr_VL) / Z_curr;
          dVR      -= n_r[a_id]*exp(curr_VR) / Z_curr;

          dVL2  -= ( n_l[a_id]*exp(curr_VL) * ( Z_curr - ( n_l[a_id]*exp(curr_VL) ) ) ) / ( Z_curr * Z_curr );
          dVR2  -= ( n_r[a_id]*exp(curr_VR) * ( Z_curr - ( n_r[a_id]*exp(curr_VR) ) ) ) / ( Z_curr * Z_curr );
          dVLVR += ( n_l[a_id]*exp(curr_VL) * n_r[a_id]*exp(curr_VR) ) / (Z_curr * Z_curr);

          //std::cout.precision(dbl::max_digits10);
          if(debug >= 3) { 
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

      if(nodeID == 0) {
        dVR = -dVL;
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

      double temp_VL = curr_VL + delta_VL;
      double temp_VR = curr_VR + delta_VR;

      // compute current likelihood
      llik = 0;
      for(auto a_id: node_agentIDs) {
        std::vector<size_t> a_sIDs = agentID_to_sampleIDs[a_id];
        double Z_curr = agent_Z[a_id] - n_l[a_id]*exp(V_star) + n_l[a_id]*exp(temp_VL) - n_r[a_id]*exp(V_star) + n_r[a_id]*exp(temp_VR);
        //std::cout << "agentID=" << a_id << "\tZ_curr=" << Z_curr << std::endl;
        //TODO: only need the agent's sampleIDs with response = 1
        for(auto s_id : a_sIDs) {
          double response = data->get(s_id, dependent_varID);
          if( response != 0 ) {
            size_t l_id = sampleID_to_leafID[s_id];
            if( l_id == nodeID ) {
              const bool is_r = right_sIDs.find(s_id) != right_sIDs.end();
              if(is_r) {
                llik += temp_VR - log(Z_curr);
              } else {
                llik += temp_VL - log(Z_curr);
              }
            } else {
              llik += util[l_id] - log(Z_curr);
            }
            break;
          } 
        }
      }
      
      /*********************************************************************
       * line search
      ************************************************************************/
      double stepsize  = 1;
      double alpha     = 0.3;
      double beta      = 0.8;
      grad_norm = dVL*dVL + dVR*dVR;

      double threshold;
      if( nodeID == 0 ) {
        threshold = stepsize*alpha*dVL*delta_VL;
      } else {
        double thresh_factor = dVL*delta_VL + dVR*delta_VR;
        threshold = stepsize*alpha*thresh_factor;
      }

      if(debug >= 2) {
        std::cout << "\tline search" << std::endl;
        std::cout << "\t\tprev_llik=" << prev_llik 
          << "\tllik=" << llik 
          << "\tstepsize=" << stepsize
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
          << std::endl;
      }
      /*
      if(debug) {
        std::cout << "\tline search" << std::endl;
        std::cout << "\t\tprev_llik=" << prev_llik 
          << "\tllik=" << llik 
          << "\tstepsize=" << stepsize
          << "\tdiff=" << llik - prev_llik 
          << "\tthreshold=" << threshold 
          << "\tdVL=" << dVL 
          << "\t1/dVL2=" << 1.0/dVL2
          << "\tdeltaVL=" << deltaVL
          << "\ttemp_VL=" << temp_VL
          << "\tcurr_VL=" << curr_VL
          << "\ttemp_VR=" << temp_VR
          << "\tcurr_VR=" << curr_VR
          << std::endl;
      }
      */

      size_t num_lineSearch_iters = 0;
      while(llik - prev_llik < threshold && step_norm > 1e-6 && fabs(dVL) > 1e-10 && fabs(dVR) > 1e-10){
        num_lineSearch_iters += 1;
        if( num_lineSearch_iters > 5000) {
          std::cout << "line search failed" << std::endl;
          std::cout << "\t\tprev_llik=" << prev_llik 
            << "\tllik=" << llik 
            << "\tstepsize=" << stepsize
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
            << std::endl;
          std::cout << "debug," << "curr.leaf" << "," << "y" << "," << "leaf.id" << "," << "v" << "," << "agent.id" << "," << "sample.id" << std::endl;
          for(auto a_id: node_agentIDs) {
            std::vector<size_t> a_sIDs = agentID_to_sampleIDs[a_id];
            for(auto s_id : a_sIDs) {
              size_t l_id     = sampleID_to_leafID[s_id];
              double response = data->get(s_id, dependent_varID);
              if( l_id == nodeID ) {
                const bool is_r = right_sIDs.find(s_id) != right_sIDs.end();
                if(is_r) {
                  std::cout << "debug," << "R" << "," << response << "," << l_id << "," << temp_VR << "," << a_id << "," << s_id << std::endl;
                } else {
                  std::cout << "debug," << "L" << "," << response << "," << l_id << "," << temp_VL << "," << a_id << "," << s_id << std::endl;
                }
              } else {
                std::cout << "debug," << "X" << "," << response << "," << l_id << "," << util[l_id] << "," << a_id << "," << s_id << std::endl;
              }
            }
          }
          exit(-1);
        }
        stepsize   = stepsize * beta;
        /*
        temp_VL = curr_VL - stepsize*deltaVL;
        temp_VR = V_star - temp_VL;
        */
        temp_VL = curr_VL + stepsize*delta_VL;
        temp_VR = curr_VR + stepsize*delta_VR;
        // compute current likelihood
        llik = 0;
        for(auto a_id: node_agentIDs) {
          std::vector<size_t> a_sIDs = agentID_to_sampleIDs[a_id];
          double Z_curr = agent_Z[a_id] - n_l[a_id]*exp(V_star) + n_l[a_id]*exp(temp_VL) - n_r[a_id]*exp(V_star) + n_r[a_id]*exp(temp_VR);
          //std::cout << "agentID=" << a_id << "\tZ_curr=" << Z_curr << std::endl;
          //TODO: only need the agent's sampleIDs with response = 1
          for(auto s_id : a_sIDs) {
            double response = data->get(s_id, dependent_varID);
            if( response != 0 ) {
              size_t l_id = sampleID_to_leafID[s_id];
              if( l_id == nodeID ) {
                const bool is_r = right_sIDs.find(s_id) != right_sIDs.end();
                if(is_r) {
                  llik += temp_VR - log(Z_curr);
                } else {
                  llik += temp_VL - log(Z_curr);
                }
              } else {
                llik += util[l_id] - log(Z_curr);
              }
              break;
            } 
          }
        }
      //threshold = -stepsize*alpha*dVL*deltaVL;
      //threshold = stepsize*alpha*grad_norm;
      if( nodeID == 0 ) {
        threshold = -stepsize*alpha*dVL*delta_VL;
      } else {
        double thresh_factor = dVL*delta_VL + dVR*delta_VR;
        threshold = stepsize*alpha*thresh_factor;
      }
      /*
        if( debug ) {
          std::cout << "\t\tnum_iters=" << num_lineSearch_iters 
            << "\tprev_llik=" << prev_llik 
            << "\tllik=" << llik 
            << "\tstepsize=" << stepsize
            << "\tdiff=" << llik - prev_llik 
            << "\tthreshold=" << threshold 
            << "\tdVL=" << dVL 
            << "\t1/dVL2=" << 1.0/dVL2
            << "\tdeltaVL=" << deltaVL
            << "\ttemp_VL=" << temp_VL
            << "\tcurr_VL=" << curr_VL
            << "\ttemp_VR=" << temp_VR
            << "\tcurr_VR=" << curr_VR
            << std::endl;
        }
        */
        if( debug >= 2 ) {
          std::cout << "\t\tnum_iters=" << num_lineSearch_iters 
            << "\tprev_llik=" << prev_llik 
            << "\tllik=" << llik 
            << "\tstepsize=" << stepsize
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
            << std::endl;
        }
      }

      if(threshold == 0 && dVL != 0) {
          std::cout << "line search failed" << std::endl;
          /*
        std::cout << "threshold zero erro" << std::endl;
        std::cout << std::fixed 
            << "\t\tnum_iters=" << num_lineSearch_iters 
            << "\titer_num=" << num_newton_iter 
            << "\tnodeID=" << nodeID
            << "\tprev_llik=" << prev_llik 
            << "\tllik=" << llik 
            << "\tstepsize=" << stepsize
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
            << std::endl;
        std::cout << "debug," << "curr.leaf" << "," << "y" << "," << "leaf.id" << "," << "v" << "," << "agent.id" << "," << "sample.id" << std::endl;
        for(auto a_id: node_agentIDs) {
          std::vector<size_t> a_sIDs = agentID_to_sampleIDs[a_id];
          for(auto s_id : a_sIDs) {
            size_t l_id     = sampleID_to_leafID[s_id];
            double response = data->get(s_id, dependent_varID);
            if( l_id == nodeID ) {
              const bool is_r = right_sIDs.find(s_id) != right_sIDs.end();
              if(is_r) {
                std::cout << "debug," << "R" << "," << response << "," << l_id << "," << temp_VR << "," << a_id << "," << s_id << std::endl;
              } else {
                std::cout << "debug," << "L" << "," << response << "," << l_id << "," << temp_VL << "," << a_id << "," << s_id << std::endl;
              }
            } else {
              std::cout << "debug," << "X" << "," << response << "," << l_id << "," << util[l_id] << "," << a_id << "," << s_id << std::endl;
            }
          }
        }
        exit(-1);
        */
      }

      // don't accept if we didn't cross threshold
      if( llik - prev_llik > threshold && ( threshold != 0 ) ){
        curr_VL = temp_VL;
        curr_VR = temp_VR;
      } else {
          llik = prev_llik;
          /*
          std::cout << "rejected this iteration of newton,threshold=" << threshold << ",step_norm=" << step_norm << ",dVL=" << dVL << ",dVR=" << dVR << ",agent_pure=" << agent_pure << std::endl;
          std::cout << "\t\tnum_iters=" << num_lineSearch_iters 
            << "\titer_num=" << num_newton_iter 
            << "\tnodeID=" << nodeID
            << "\tprev_llik=" << prev_llik 
            << "\tllik=" << llik 
            << "\tstepsize=" << stepsize
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
            << std::endl;
            */
      }
      
      if( debug >= 2 ) {
        std::cout << "\tline search complete" << std::endl;
        std::cout// << std::fixed << std::setprecision(15) 
          << "\tsplit_num=" << num_splits 
          << "\titer_num=" << num_newton_iter 
          << "\tc_l=" << c_l 
          << "\tc_r=" << c_r 
          << "\tdelta_VL=" << delta_VL 
          << "\tdelta_VR=" << delta_VR 
          << "\tdVL=" << dVL 
          << "\tdVR=" << dVR 
          << "\tdVL2=" << dVL2
          << "\tdVR2=" << dVR2 
          << "\tdVLVR=" << dVLVR
          << "\t1/dtmnt=" << dtmnt
          << "\tnew_llik=" << llik
          << "\tprev_llik=" << prev_llik 
          << "\tdelta_step=" << llik - prev_llik
          << "\tpre-split_llik=" << curr_llik
          << "\tdelta_llik=" << llik - curr_llik
          << "\tcurr_VL= " << curr_VL 
          << "\tcurr_VR= " << curr_VR << std::endl;
      }
      /*
      if( debug ) {
        std::cout << "\tline search complete" << std::endl;
        std::cout// << std::fixed << std::setprecision(15) 
          << "\tsplit_num=" << num_splits 
          << "\titer_num=" << num_newton_iter 
          << "\tc_l=" << c_l 
          << "\tc_r=" << c_r 
          << "\tdVL=" << dVL 
          << "\t1/dVL2=" << 1.0/dVL2
          << "\tdeltaVL=" << deltaVL
          << "\tnew_llik=" << llik
          << "\tprev_llik=" << prev_llik 
          << "\tdelta_step=" << llik - prev_llik
          << "\tpre-split_llik=" << curr_llik
          << "\tdelta_llik=" << llik - curr_llik
          << "\tcurr_VL= " << curr_VL 
          << "\tcurr_VR= " << curr_VR << std::endl;
      }
      */
   } while( (llik - prev_llik) > 1e-4  && step_norm > 1e-6); //TODO: need llik < curr_llik ?
    /*****************************************************************/
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
      if( debug >= 2 ) 
        std::cout << "best increase=" << best_increase << std::endl;
      child_util[0][nodeID] = curr_VL;
      child_util[1][nodeID] = curr_VR;
    } else {
      //std::cout << "index=" << i << " not good enough with increase=" << increase << std::endl;
    }
  }
}


void TreeDiscreteChoice::addImpurityImportance(size_t nodeID, size_t varID, double decrease) {

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
    double curr_llik = compute_log_likelihood(agent_Z, curr_util);

    // gradient descent
    double prev_llik;
    do {

        prev_llik       = curr_llik;
        compute_full_gradient(leafID_to_partial, agent_Z, curr_util, leafIDs);

        // armijo line search
        curr_llik = backtracking(leafID_to_partial, agent_Z, curr_util, leafIDs, prev_llik);

        if(debug >= 2)
            std::cout << "global adjustment,gradient descent,curr_llik=" << curr_llik << "\tprev_llik=" << prev_llik << "\tdiff=" << curr_llik - prev_llik << std::endl;

    } while ( curr_llik - prev_llik > 1e-5 );
    if(debug >= 2) 
        std::cout << "done with global adjustment,curr_llik=" << curr_llik << "\tprev_llik=" << prev_llik << "\tdiff=" << curr_llik - prev_llik << std::endl;
    util = curr_util;
    
    for(auto & l_id : leafIDs) {
      split_values[l_id] = util[l_id];
    }

    // tree statistics
    if(debug >= 1) {
        for(auto & l_id : leafIDs) {
            std::cout << "leaf_id=" << l_id 
                      << "\tdepth=" << node_depth[l_id] 
                      << "\tsize="  << sampleIDs[l_id].size() 
                      << "\tutil="  << util[l_id]
                      << std::endl;
        } 
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
    double curr_llik = compute_log_likelihood(agent_Z, temp_util);

    if( debug >= 2 ) {
      std::cout << std::fixed << "global adjustement," << "prev_llik=" << prev_llik << "\tcurr_llik=" << curr_llik << "\tstepsize=" << stepsize << "\tgrad_norm=" << grad_norm
        << "\tmax_grad_component=" << max_grad_component
        << "\tdiff=" << curr_llik - prev_llik << "\tthreshold=" << stepsize*alpha*grad_norm_squared << std::endl;
    }

    size_t num_iter = 0;
    while(curr_llik - prev_llik < alpha*grad_norm_squared*stepsize  && grad_norm > 1e-4) {
        num_iter += 1;
        temp_util  = curr_util;
        stepsize   = stepsize * beta;
        for( auto l_id : leafIDs ) {
            temp_util[l_id] += stepsize*leafID_to_partial.at(l_id); 
        }
        compute_partition_func(agent_Z, temp_util);
        curr_llik = compute_log_likelihood(agent_Z, temp_util);

        if( debug >= 2 ) { 
          std::cout << "global adjustement," << "prev_llik=" << prev_llik << "\tcurr_llik=" << curr_llik << "\tstepsize=" << stepsize << "\tgrad_norm=" << grad_norm
            << "\tdiff=" << curr_llik - prev_llik << "\tthreshold=" << stepsize*alpha*grad_norm_squared << std::endl;
        }
        if(num_iter > 1000) {
            std::cout << "global adjustment line search failed" << std::endl;
            std::cout << "debug,leaf.id,agent.id,sample.id,util,response" << std::endl;
            for(auto a_id: agentIDs) {
                std::vector<size_t> a_sIDs = agentID_to_sampleIDs[a_id];
                for(auto s_id : a_sIDs) {
                    size_t l_id     = sampleID_to_leafID[s_id];
                    double response = data->get(s_id, dependent_varID);
                    std::cout << std::fixed << std::setprecision(15)  << "debug," << l_id << "," << a_id << "," << s_id << "," << curr_util[l_id] << "," << response << std::endl;
                }
            }
            exit(-1);
        }
    }
    //TODO: why do we need this?
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

double TreeDiscreteChoice::compute_log_likelihood(const std::unordered_map<size_t, double>& agent_Z, const std::vector<double>& curr_util) {
    double llik = 0;
    size_t debug_num = 0;
    for(auto a_id: agentIDs) {
        for(auto s_id: agentID_to_sampleIDs[a_id]) {
          debug_num +=1;
            double response  = data->get(s_id, dependent_varID);
            size_t l_id = sampleID_to_leafID[s_id];
            double Z = agent_Z.at(a_id);
            llik  += response*(curr_util[l_id] - log(Z));
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
