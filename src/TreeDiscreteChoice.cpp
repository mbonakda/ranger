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
#include "TreeDiscreteChoice.h"
#include "Data.h"

TreeDiscreteChoice::TreeDiscreteChoice() :
    counter(0), sums(0) {
}

TreeDiscreteChoice::TreeDiscreteChoice(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_varIDs,
    std::vector<double>& split_values, std::vector<bool>* is_ordered_variable) :
    Tree(child_nodeIDs, split_varIDs, split_values, is_ordered_variable), counter(0), sums(0), dcrf_numItems(0), dcrf_numAgents(0) {
      num_splits = 0;
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

}

void TreeDiscreteChoice::post_bootstrap_init() {
  std::cout << "TreeDiscreteChoice::post_bootstrap_init()" << std::endl;
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
  llik.push_back(-dcrf_numAgents*log((double)dcrf_numItems));
  std::cout << "numAgents:\t" << dcrf_numAgents << std::endl;
  std::cout << "numItems:\t" << dcrf_numItems << std::endl;
  std::cout << "initial log-lik: " << llik[0] << std::endl;

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

bool TreeDiscreteChoice::splitNodeInternal(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {

  std::cout << "considered for split number " << num_splits << std::endl;

  //TODO: do better here
  size_t agentID_varID = data->getVariableID("agentID");
  possible_split_varIDs.erase(std::remove(possible_split_varIDs.begin(), possible_split_varIDs.end(), agentID_varID), possible_split_varIDs.end());

  // Check node size, stop if maximum reached
  if (sampleIDs[nodeID].size() <= min_node_size) {
    std::cout << "reached min node size" << std::endl;
    split_values[nodeID] = util[nodeID];
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
    split_values[nodeID] = util[nodeID];
    std::cout << "PURE NODE" << std::endl;
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
    std::cout << "no good splits" << std::endl;
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
  double best_VL = 0;
  double best_VR = 0;

  size_t c_star = 0, c_l = 0, c_r = 0; // number of positive choices in current leaf, and potential left/right leaf

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

    node_agentIDs.insert(agentID);

    auto itr = idx_to_sID.find(index);
    if( itr == idx_to_sID.end() ) {
      idx_to_sID.emplace(index, std::unordered_set<size_t>());
    }
    idx_to_sID[index].insert(sampleID);
    right_sIDs.insert(sampleID);
    ++counter[index];
  }

  
  // compute partition funcs for each agent
  for(auto a_id: node_agentIDs) {
    for(auto s_id: agentID_to_sampleIDs[a_id]) {
      size_t leaf_id = sampleID_to_leafID[s_id];
      agent_Z[a_id] += exp(util[leaf_id]);
    }
  }

  size_t n_left = 0;
  double sum_left = 0;

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


  std::cout << "curr likelihood = " << curr_llik << std::endl;
  double V_star  = util[nodeID];


  for (size_t i = 0; i < num_unique - 1; ++i) {

    std::cout << "progress: " << i << "/" << num_unique << std::endl;
    double curr_VL = util[nodeID];
    double curr_VR = V_star - curr_VL;
    std::cout << "start_VL= " << curr_VL << std::endl;
    std::cout << "start_VR= " << curr_VR << std::endl;

    // Stop if nothing here
    if (counter[i] == 0) {
      std::cout << "no samples at this index" << std::endl;
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
    n_left += counter[i];
    //sum_left += sums[i];
    size_t n_right = num_samples_node - n_left;
    if (n_right == 0) {
      break;
    }


    /*****************************************************************
    // Maximum Likelihood
    *****************************************************************/
    double deltaVL  = 0;
    double llik      = curr_llik;
    double prev_llik;
    size_t num_newton_iter = 0;
    do { 
      num_newton_iter += 1;
      prev_llik = llik;
      // compute entries for gradient + hessian
      double dVL = 0;
      double dVL2 = 0;

      dVL += static_cast<double>(c_l) - static_cast<double>(c_r);
      for(auto a_id: node_agentIDs) {

        double Z_curr = agent_Z[a_id] - n_l[a_id]*exp(V_star) + n_l[a_id]*exp(curr_VL) - n_r[a_id]*exp(V_star) + n_r[a_id]*exp(curr_VR);

        double   mm  = ( n_l[a_id]*exp(curr_VL) - n_r[a_id]*exp(curr_VR) );
        double   pp  = ( n_l[a_id]*exp(curr_VL) + n_r[a_id]*exp(curr_VR) );
        dVL      -=  mm / Z_curr;
        dVL2     -=  ( Z_curr*pp  - mm*mm ) / (Z_curr*Z_curr);

        /*
           std::cout << "agentID=" << a_id << "\tn_l=" << n_l[a_id] << "\tn_r=" << n_r[a_id] << "\tZ_curr=" << Z_curr 
           << "\tdVL=" << dVL  << "\tdVL2=" << dVL2 
           << "\tmm=" << mm << "\tpp=" << pp << std::endl;
           */

      }

      // newton step
      deltaVL = (1.0 / dVL2)*dVL;
      curr_VL = curr_VL - deltaVL;
      curr_VR = V_star - curr_VL;

      // compute current likelihood
      llik = 0;
      for(auto a_id: node_agentIDs) {
        std::vector<size_t> a_sIDs = agentID_to_sampleIDs[a_id];
        double Z_curr = agent_Z[a_id] - n_l[a_id]*exp(V_star) + n_l[a_id]*exp(curr_VL) - n_r[a_id]*exp(V_star) + n_r[a_id]*exp(curr_VR);
        //std::cout << "agentID=" << a_id << "\tZ_curr=" << Z_curr << std::endl;
        //TODO: only need the agent's sampleIDs with response = 1
        for(auto s_id : a_sIDs) {
          double response = data->get(s_id, dependent_varID);
          if( response != 0 ) {
            size_t l_id = sampleID_to_leafID[s_id];
            if( l_id == nodeID ) {
              const bool is_r = right_sIDs.find(s_id) != right_sIDs.end();
              if(is_r) {
                llik += curr_VR - log(Z_curr);
              } else {
                llik += curr_VL - log(Z_curr);
              }
            } else {
              llik += util[l_id] - log(Z_curr);
            }
            break;
          } 
        }
      }

      std::cout << "split_num=" << num_splits << "\titer_num=" << num_newton_iter << "\tc_l=" << c_l << "\tc_r=" << c_r 
      << "\tdVL=" << dVL 
      << "\t1/dVL2=" << 1.0/dVL2
      << "\tdeltaVL=" << deltaVL
      << "\tnew_llik=" << llik
      << "\tprev_llik=" << prev_llik 
      << "\tdelta_llik=" << llik - prev_llik
      << "\tcurr_VL= " << curr_VL 
      << "\tcurr_VR= " << curr_VR << std::endl;
   } while( fabs(llik - prev_llik) > 0.001 );// && abs(deltaVL) > 0.001); 
    /*****************************************************************/
    double increase = llik - curr_llik;
    // If better than before, use this
    if (increase > best_increase ) {
      std::cout << "new best increase=" << increase << "\tindex=" << i << std::endl;
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
      std::cout << "index=" << i << " not good enough with increase=" << increase << std::endl;
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

void TreeDiscreteChoice::reshape() {
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

  /*
  // 3. perform bottom-up over-constrained optimization
  for (auto& nn : sc_nodes) {
  over_constr_opt(nn, node_to_left[nn], node_to_right[nn]);
  }
  */


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

void TreeDiscreteChoice::goldilocks_opt(const std::set<size_t> & leaves, const std::vector<std::pair<size_t, size_t>> & id_edges) {

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

