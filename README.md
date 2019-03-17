## ranger: A Fast Implementation of Random Forests
* Marvin N. Wright, wright@imbs.uni-luebeck.de
* Extended by Matt Bonakdarpour

### Introduction
This fork of the ranger package contains implementations for the following two methods:
* Random Forest Reshaping (Bonakdarpour et al., 2018)
* Discrete Choice Random Forests

### Reshaping
The reshaping implementation allows the user to input a character vector corresponding to the desired reshaped predictor variables. The reshaping algorithm will enforce the constraint such that the predictions are monotonically increasing with respect to those input variables. 

The current reshaping functionality is only implemented for regression and probability trees. In the master branch, only the "over-constrained estimator" (Bonakdarpour et al, 2018) is implemented to minimize dependencies on external packages. The `reshape-only` branch also contains the "exact estimator" but requires updates to the `Makevars` file based on your local `mosek` installation path. 

Here is example usage for fitting a shape-constrained random forest with respect to an input variabled called `bmi`:
```r
rf                       <- ranger(dependent.variable.name = 'y',
                                   data                    = input.data.df,
                                   write.forest            = TRUE,
                                   num.trees               = 100,
                                   sc.variable.names       = c('bmi')
                                   )
```

### Discrete Choice Random Forests
The discrete choice random forest implementation allows the user to specify a `speedy` input parameter which implements an approximate split-finding algorithm. The dataframe is expect to be in "long" format as defined by the `mlogit` R package. The implementation expects a column labeled `agentID` which uniquely identifies the agent making the choice in the corresponding row. The dependent variable is assumed to be an integer (0 or 1) depending on whether or not the agent chose the corresponding item in that row. The remaining columns are assumed to be predictor variables. 

Example usage:
```r
rf                      <- ranger(dependent.variable.name = 'choice',
                                  data                = input.data.df,
                                  write.forest        = TRUE,
                                  num.trees           = 100,
                                  discrete.choice     = TRUE
                                 )
```

### References
* Wright, M. N. & Ziegler, A. (2017). ranger: A Fast Implementation of Random Forests for High Dimensional Data in C++ and R. Journal of Statistical Software 77:1-17. http://dx.doi.org/10.18637/jss.v077.i01.
* Schmid, M., Wright, M. N. & Ziegler, A. (2016). On the use of Harrell’s C for clinical risk prediction via random survival forests. Expert Systems with Applications 63:450-459. http://dx.doi.org/10.1016/j.eswa.2016.07.018.
* Wright, M. N., Dankowski, T. & Ziegler, A. (2017). Unbiased split variable selection for random survival forests using maximally selected rank statistics. Statistics in Medicine. http://dx.doi.org/10.1002/sim.7212.
* Breiman, L. (2001). Random forests. Machine learning 45:5-32.
* Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008). Random survival forests. The Annals of Applied Statistics 2:841-860.
* Malley, J. D., Kruppa, J., Dasgupta, A., Malley, K. G., & Ziegler, A. (2012). Probability machines: consistent probability estimation using nonparametric learning machines. Methods of Information in Medicine 51:74-81.
* Bonakdarpour, M., Chatterjee, S., Foygel Barber, R., Lafferty, J. (2018). ICML
