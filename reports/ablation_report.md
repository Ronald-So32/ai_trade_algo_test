# Ablation Test Results

**Total variants tested:** 119


## Drawdown Control

| variant                                  | is_baseline   |      sharpe |      calmar |   max_drawdown |   cdar_95 |   skewness |
|:-----------------------------------------|:--------------|------------:|------------:|---------------:|----------:|-----------:|
| time_series_momentum_binary_breaker      | True          |  0.39232    |  0.177896   |     -0.189404  | 0.175943  | -2.01557   |
| time_series_momentum_no_control          | False         |  0.449897   |  0.202963   |     -0.202521  | 0.183874  | -1.77024   |
| time_series_momentum_continuous_cdar     | False         |  0.39232    |  0.177896   |     -0.189404  | 0.175943  | -2.01557   |
| cross_sectional_momentum_binary_breaker  | True          |  0.349351   |  0.107225   |     -0.19939   | 0.174103  | -0.522869  |
| cross_sectional_momentum_no_control      | False         |  0.368124   |  0.131119   |     -0.182693  | 0.17439   | -0.494864  |
| cross_sectional_momentum_continuous_cdar | False         |  0.349351   |  0.107225   |     -0.19939   | 0.174103  | -0.522869  |
| mean_reversion_binary_breaker            | True          | -0.0441329  | -0.0143471  |     -0.0354022 | 0.0305411 | -0.753192  |
| mean_reversion_no_control                | False         | -0.0441329  | -0.0143471  |     -0.0354022 | 0.0305411 | -0.753192  |
| mean_reversion_continuous_cdar           | False         | -0.0441329  | -0.0143471  |     -0.0354022 | 0.0305411 | -0.753192  |
| distance_pairs_binary_breaker            | True          | -0.346246   | -0.0538523  |     -0.173047  | 0.173023  | -7.93485   |
| distance_pairs_no_control                | False         | -0.36439    | -0.0571843  |     -0.204654  | 0.204197  | -4.97269   |
| distance_pairs_continuous_cdar           | False         | -0.346246   | -0.0538523  |     -0.173047  | 0.173023  | -7.93485   |
| kalman_pairs_binary_breaker              | True          | -0.437614   | -0.0542392  |     -0.149958  | 0.149491  | -2.27879   |
| kalman_pairs_no_control                  | False         | -0.503524   | -0.0562801  |     -0.176788  | 0.175073  | -2.39758   |
| kalman_pairs_continuous_cdar             | False         | -0.437614   | -0.0542392  |     -0.149958  | 0.149491  | -2.27879   |
| volatility_breakout_binary_breaker       | True          |  0.262583   |  0.112534   |     -0.164003  | 0.156458  |  1.2216    |
| volatility_breakout_no_control           | False         |  0.301422   |  0.132968   |     -0.167299  | 0.155859  |  1.09821   |
| volatility_breakout_continuous_cdar      | False         |  0.262583   |  0.112534   |     -0.164003  | 0.156458  |  1.2216    |
| carry_binary_breaker                     | True          | -0.266492   | -0.0493062  |     -0.283338  | 0.283338  | -0.990259  |
| carry_no_control                         | False         | -0.0400671  | -0.0283696  |     -0.195178  | 0.195157  | -1.74142   |
| carry_continuous_cdar                    | False         | -0.266492   | -0.0493062  |     -0.283338  | 0.283338  | -0.990259  |
| factor_momentum_binary_breaker           | True          | -0.475656   | -0.0667193  |     -0.156438  | 0.156438  | -1.32816   |
| factor_momentum_no_control               | False         | -0.554011   | -0.0682642  |     -0.196986  | 0.196983  | -1.41011   |
| factor_momentum_continuous_cdar          | False         | -0.475656   | -0.0667193  |     -0.156438  | 0.156438  | -1.32816   |
| pca_stat_arb_binary_breaker              | True          | -0.0788986  | -0.0255194  |     -0.167096  | 0.166883  |  0.229904  |
| pca_stat_arb_no_control                  | False         | -0.0360807  | -0.0188321  |     -0.203486  | 0.16147   |  0.173757  |
| pca_stat_arb_continuous_cdar             | False         | -0.0788986  | -0.0255194  |     -0.167096  | 0.166883  |  0.229904  |
| vol_managed_binary_breaker               | True          |  0.961233   |  0.593081   |     -0.167156  | 0.150805  | -0.953462  |
| vol_managed_no_control                   | False         |  1.00838    |  0.673207   |     -0.158111  | 0.13599   | -0.913054  |
| vol_managed_continuous_cdar              | False         |  0.961233   |  0.593081   |     -0.167156  | 0.150805  | -0.953462  |
| pead_binary_breaker                      | True          |  0.34245    |  0.134853   |     -0.136794  | 0.130712  |  0.925606  |
| pead_no_control                          | False         |  0.368404   |  0.145328   |     -0.139575  | 0.129003  |  0.900813  |
| pead_continuous_cdar                     | False         |  0.34245    |  0.134853   |     -0.136794  | 0.130712  |  0.925606  |
| residual_momentum_binary_breaker         | True          | -0.292792   | -0.0518875  |     -0.211289  | 0.204017  | -0.538981  |
| residual_momentum_no_control             | False         |  0.0575241  |  0.00883633 |     -0.181202  | 0.153542  | -0.0591834 |
| residual_momentum_continuous_cdar        | False         | -0.292792   | -0.0518875  |     -0.211289  | 0.204017  | -0.538981  |
| low_risk_bab_binary_breaker              | True          |  0.0479283  |  0.00581421 |     -0.183073  | 0.183073  | -0.218422  |
| low_risk_bab_no_control                  | False         |  0.00350098 | -0.00915177 |     -0.224687  | 0.224627  | -0.0833819 |
| low_risk_bab_continuous_cdar             | False         |  0.0479283  |  0.00581421 |     -0.183073  | 0.183073  | -0.218422  |
| short_term_reversal_binary_breaker       | True          | -0.105431   | -0.0296496  |     -0.166988  | 0.166985  | -0.0917985 |
| short_term_reversal_no_control           | False         | -0.14357    | -0.0344951  |     -0.204975  | 0.204905  | -0.120803  |
| short_term_reversal_continuous_cdar      | False         | -0.105431   | -0.0296496  |     -0.166988  | 0.166985  | -0.0917985 |
| vol_risk_premium_binary_breaker          | True          |  0.203469   |  0.0683011  |     -0.182993  | 0.182986  |  2.20315   |
| vol_risk_premium_no_control              | False         |  0.194717   |  0.0536489  |     -0.235233  | 0.235108  |  2.25096   |
| vol_risk_premium_continuous_cdar         | False         |  0.203469   |  0.0683011  |     -0.182993  | 0.182986  |  2.20315   |



## Momentum Risk Management

| variant                              | is_baseline   |      sharpe |      calmar |   max_drawdown |   cdar_95 |   skewness |
|:-------------------------------------|:--------------|------------:|------------:|---------------:|----------:|-----------:|
| cross_sectional_momentum_raw         | True          |  0.368124   |  0.131119   |      -0.182693 |  0.17439  | -0.494864  |
| cross_sectional_momentum_vol_scaled  | False         |  0.498041   |  0.236767   |      -0.181    |  0.175394 |  0.0546072 |
| cross_sectional_momentum_crash_gated | False         |  0.500087   |  0.24772    |      -0.163525 |  0.157799 |  0.0481948 |
| time_series_momentum_raw             | True          |  0.449897   |  0.202963   |      -0.202521 |  0.183874 | -1.77024   |
| time_series_momentum_vol_scaled      | False         |  0.633006   |  0.394954   |      -0.159589 |  0.130759 | -0.803793  |
| time_series_momentum_crash_gated     | False         |  0.66111    |  0.396968   |      -0.159589 |  0.128224 | -0.811909  |
| residual_momentum_raw                | True          |  0.0575241  |  0.00883633 |      -0.181202 |  0.153542 | -0.0591834 |
| residual_momentum_vol_scaled         | False         |  0.0204466  | -0.0095747  |      -0.29916  |  0.273969 | -0.0725383 |
| residual_momentum_crash_gated        | False         |  0.00637709 | -0.0133889  |      -0.293208 |  0.267803 | -0.0336166 |
| factor_momentum_raw                  | True          | -0.554011   | -0.0682642  |      -0.196986 |  0.196983 | -1.41011   |
| factor_momentum_vol_scaled           | False         | -0.562128   | -0.0709006  |      -0.263566 |  0.263561 | -2.15325   |
| factor_momentum_crash_gated          | False         | -0.568656   | -0.0663076  |      -0.140015 |  0.140008 | -2.21892   |



## Covariance Estimator

| variant                | is_baseline   |   sharpe |    calmar |   max_drawdown |   cdar_95 |   skewness |
|:-----------------------|:--------------|---------:|----------:|---------------:|----------:|-----------:|
| sample_cov_risk_parity | True          | 0.277398 | 0.0567218 |     -0.0619274 | 0.0590825 |   -1.83547 |
| shrinkage_risk_parity  | False         | 0.277398 | 0.0567218 |     -0.0619274 | 0.0590825 |   -1.83547 |



## Cost Stress

| variant            | is_baseline   |    sharpe |     calmar |   max_drawdown |   cdar_95 |   skewness |
|:-------------------|:--------------|----------:|-----------:|---------------:|----------:|-----------:|
| portfolio_baseline | True          |  1.32417  |  0.987322  |     -0.0890837 | 0.063593  |  -0.187085 |
| portfolio_2x_costs | False         |  0.552132 |  0.305313  |     -0.112973  | 0.0959931 |  -0.187085 |
| portfolio_4x_costs | False         | -0.991947 | -0.0975061 |     -0.663634  | 0.65657   |  -0.187085 |



## Allocation Logic

| variant      | is_baseline   |   sharpe |    calmar |   max_drawdown |   cdar_95 |   skewness |
|:-------------|:--------------|---------:|----------:|---------------:|----------:|-----------:|
| equal_weight | True          | 0.629899 | 0.219344  |     -0.0601074 | 0.0558384 |   -1.72533 |
| risk_parity  | False         | 0.277398 | 0.0567218 |     -0.0619274 | 0.0590825 |   -1.83547 |
| herc_cdar    | False         | 0.78426  | 0.304957  |     -0.0537989 | 0.0457635 |   -1.30257 |



## Advanced Allocation

| variant               | is_baseline   |   sharpe |    calmar |   max_drawdown |   cdar_95 |   skewness |
|:----------------------|:--------------|---------:|----------:|---------------:|----------:|-----------:|
| equal_weight          | True          | 0.629899 | 0.219344  |     -0.0601074 | 0.0558384 |   -1.72533 |
| cvar_optimized        | False         | 0.19901  | 0.038739  |     -0.0462353 | 0.0443415 |   -1.21634 |
| downside_risk_parity  | False         | 0.210679 | 0.0396515 |     -0.0540887 | 0.0520654 |   -1.67034 |
| max_diversification   | False         | 0.200228 | 0.036759  |     -0.0593823 | 0.0568846 |   -1.71769 |
| blend_cvar_drp_maxdiv | False         | 0.206623 | 0.0387205 |     -0.0527244 | 0.0507126 |   -1.59764 |



## Systemic Risk Overlay

| variant            | is_baseline   |   sharpe |   calmar |   max_drawdown |   cdar_95 |   skewness |
|:-------------------|:--------------|---------:|---------:|---------------:|----------:|-----------:|
| no_overlay         | True          | 1.32417  | 0.987322 |     -0.0890837 | 0.063593  |  -0.187085 |
| turbulence_overlay | False         | 0.825929 | 0.291853 |     -0.0504019 | 0.0459315 |  -0.785731 |
| absorption_overlay | False         | 0.725448 | 0.229635 |     -0.0574245 | 0.0537899 |  -1.18396  |
| composite_overlay  | False         | 0.806108 | 0.300961 |     -0.0455129 | 0.0423378 |  -0.874806 |



## Adaptive Stops

| variant                                    | is_baseline   |      sharpe |       calmar |   max_drawdown |   cdar_95 |    skewness |
|:-------------------------------------------|:--------------|------------:|-------------:|---------------:|----------:|------------:|
| time_series_momentum_no_stops              | True          |  0.449897   |  0.202963    |     -0.202521  | 0.183874  | -1.77024    |
| time_series_momentum_adaptive_tight        | False         |  0.477881   |  0.21486     |     -0.188957  | 0.169057  | -1.59735    |
| time_series_momentum_adaptive_standard     | False         |  0.408118   |  0.171353    |     -0.21265   | 0.193883  | -1.80935    |
| cross_sectional_momentum_no_stops          | True          |  0.368124   |  0.131119    |     -0.182693  | 0.17439   | -0.494864   |
| cross_sectional_momentum_adaptive_tight    | False         |  0.178353   |  0.0458219   |     -0.210328  | 0.201782  | -0.778801   |
| cross_sectional_momentum_adaptive_standard | False         |  0.405721   |  0.136466    |     -0.192488  | 0.184017  | -0.506917   |
| mean_reversion_no_stops                    | True          | -0.0441329  | -0.0143471   |     -0.0354022 | 0.0305411 | -0.753192   |
| mean_reversion_adaptive_tight              | False         | -0.179529   | -0.0471972   |     -0.0380689 | 0.0305602 | -2.1023     |
| mean_reversion_adaptive_standard           | False         | -0.0441329  | -0.0143471   |     -0.0354022 | 0.0305411 | -0.753192   |
| distance_pairs_no_stops                    | True          | -0.36439    | -0.0571843   |     -0.204654  | 0.204197  | -4.97269    |
| distance_pairs_adaptive_tight              | False         | -0.0958446  | -0.032727    |     -0.0881159 | 0.0876504 | -0.640627   |
| distance_pairs_adaptive_standard           | False         | -0.342269   | -0.0549475   |     -0.191748  | 0.191303  | -5.7484     |
| kalman_pairs_no_stops                      | True          | -0.503524   | -0.0562801   |     -0.176788  | 0.175073  | -2.39758    |
| kalman_pairs_adaptive_tight                | False         | -0.37285    | -0.050514    |     -0.127351  | 0.125442  | -2.46516    |
| kalman_pairs_adaptive_standard             | False         | -0.451618   | -0.0541569   |     -0.154185  | 0.152913  | -2.58943    |
| volatility_breakout_no_stops               | True          |  0.301422   |  0.132968    |     -0.167299  | 0.155859  |  1.09821    |
| volatility_breakout_adaptive_tight         | False         |  0.149928   |  0.0520143   |     -0.172157  | 0.160855  |  1.17811    |
| volatility_breakout_adaptive_standard      | False         |  0.24171    |  0.102315    |     -0.162459  | 0.152715  |  1.15988    |
| carry_no_stops                             | True          | -0.0400671  | -0.0283696   |     -0.195178  | 0.195157  | -1.74142    |
| carry_adaptive_tight                       | False         | -0.139895   | -0.0491217   |     -0.244297  | 0.244279  | -2.00547    |
| carry_adaptive_standard                    | False         | -0.0779701  | -0.0391271   |     -0.206978  | 0.206957  | -1.82155    |
| factor_momentum_no_stops                   | True          | -0.554011   | -0.0682642   |     -0.196986  | 0.196983  | -1.41011    |
| factor_momentum_adaptive_tight             | False         | -0.559708   | -0.0670568   |     -0.165457  | 0.165456  | -1.32658    |
| factor_momentum_adaptive_standard          | False         | -0.492021   | -0.0672448   |     -0.17046   | 0.170457  | -1.09421    |
| pca_stat_arb_no_stops                      | True          | -0.0360807  | -0.0188321   |     -0.203486  | 0.16147   |  0.173757   |
| pca_stat_arb_adaptive_tight                | False         | -0.14404    | -0.0432319   |     -0.201896  | 0.173014  |  0.258152   |
| pca_stat_arb_adaptive_standard             | False         | -0.108911   | -0.0331226   |     -0.229419  | 0.185821  |  0.206836   |
| vol_managed_no_stops                       | True          |  1.00838    |  0.673207    |     -0.158111  | 0.13599   | -0.913054   |
| vol_managed_adaptive_tight                 | False         |  0.914381   |  0.484248    |     -0.183908  | 0.156284  | -0.994376   |
| vol_managed_adaptive_standard              | False         |  0.990039   |  0.652321    |     -0.158111  | 0.140832  | -0.933629   |
| pead_no_stops                              | True          |  0.368404   |  0.145328    |     -0.139575  | 0.129003  |  0.900813   |
| pead_adaptive_tight                        | False         |  0.288327   |  0.0764378   |     -0.187662  | 0.170683  |  0.705288   |
| pead_adaptive_standard                     | False         |  0.363456   |  0.142997    |     -0.139575  | 0.129003  |  0.902458   |
| residual_momentum_no_stops                 | True          |  0.0575241  |  0.00883633  |     -0.181202  | 0.153542  | -0.0591834  |
| residual_momentum_adaptive_tight           | False         |  0.115379   |  0.0317268   |     -0.165487  | 0.132292  |  0.00835937 |
| residual_momentum_adaptive_standard        | False         | -0.0583434  | -0.0297854   |     -0.198839  | 0.172587  | -0.0817955  |
| low_risk_bab_no_stops                      | True          |  0.00350098 | -0.00915177  |     -0.224687  | 0.224627  | -0.0833819  |
| low_risk_bab_adaptive_tight                | False         |  0.162949   |  0.0635614   |     -0.129519  | 0.129461  | -0.0480639  |
| low_risk_bab_adaptive_standard             | False         |  0.0349322  |  0.000578806 |     -0.213558  | 0.213488  | -0.0676527  |
| short_term_reversal_no_stops               | True          | -0.14357    | -0.0344951   |     -0.204975  | 0.204905  | -0.120803   |
| short_term_reversal_adaptive_tight         | False         | -0.122677   | -0.0281117   |     -0.193866  | 0.193805  | -0.241231   |
| short_term_reversal_adaptive_standard      | False         | -0.132123   | -0.0301586   |     -0.208209  | 0.208139  | -0.250913   |
| vol_risk_premium_no_stops                  | True          |  0.194717   |  0.0536489   |     -0.235233  | 0.235108  |  2.25096    |
| vol_risk_premium_adaptive_tight            | False         |  0.190848   |  0.0442918   |     -0.27145   | 0.271381  |  2.52421    |
| vol_risk_premium_adaptive_standard         | False         |  0.171481   |  0.0448211   |     -0.238404  | 0.238325  |  2.27507    |

